import os
from typing import Any, Dict, List, Tuple

import cv2
from tqdm import tqdm
from pathlib import Path
from app.clients.json_processor import JsonProcessor
from app.clients.video_processors.opencv_reader import OpenCVVideoReader
from app.pipeline.base_processor import BaseProcessor
from app.pipeline.people_detector import PeopleDetector
from app.pipeline.face_detector import FaceDetector
from app.utils.config_reader import ConfigReader
from app.pipeline.face_search import FaceSearchService


class AutomaticIdentificationPipeline:
    def __init__(self):
        """
        :param processors: список обработчиков в порядке их выполнения
        """
        self.config_reader = ConfigReader()

        general_config = self.config_reader.get_general_config()
        self.json_processor = JsonProcessor(
            output_root=general_config.output_dir, json_size=general_config.json_size
        )

        self.people_detector: PeopleDetector = PeopleDetector(
        )

        self.face_detector: FaceDetector = FaceDetector(
        )

        self.face_searcher: FaceSearchService = FaceSearchService()

    def process_video(
        self,
        video_path: str,
        target_fps: float = 10,
        show_video: bool = False,
        save_video: bool = False,
        save_in_json: bool = False,
        show_progress: bool = True,
    ):
        """
        Обработка видео с детекцией людей

        :param video_path: путь к видеофайлу
        :param target_fps: целевая частота кадров
        :param show_video: показывать ли видео в реальном времени
        :param save_in_json: сохранять ли результаты в JSON
        :param show_progress: показывать ли прогресс-бар
        :return: (список путей к JSON-файлам, информация о видео)
        """
        if save_video:
            output_dir = Path("results")
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"processed_{Path(video_path).name}"

        json_files = []
        with OpenCVVideoReader(video_path, target_fps=target_fps) as vp:
            video_info = vp.get_video_info()

            if save_video:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                frame_size = (video_info["width"], video_info["height"])
                video_writer = cv2.VideoWriter(
                    str(output_path), fourcc, target_fps, frame_size
                )

            total_frames = int(
                video_info["total_frames"]
                / (video_info["original_fps"] / video_info["target_fps"])
            )
            duration = video_info["duration"]

            print(f"\nProcessing video: {os.path.basename(video_path)}")
            print(
                f"Duration: {duration:.2f}s, Frames: {total_frames}, Target FPS: {target_fps}, Original FPS {video_info['original_fps']}"
            )

            pbar = tqdm(
                total=total_frames,
                desc="Processing frames",
                unit="frame",
                disable=not show_progress,
            )

            try:
                for frame, timestamp, frame_num in vp.frames():

                    result = self.process_frame(frame, show_video)

                    if save_video and result["frame"] is not None:
                        result["frame"] = self.draw_frame_number(
                            result["frame"], frame_num
                        )
                        video_writer.write(result["frame"])

                    # перед сохранением в json можно добавить постпроцесор
                    # который будет проходиться "скользящим окном"
                    # брать последние N кадров
                    result_file = self.json_processor.save_detections(
                        video_path=video_path,
                        frame_num=frame_num,
                        timestamp=timestamp,
                        people=result["result_dicts"],
                    )

                    if result_file and result_file not in json_files:
                        json_files.append(result_file)

                    if show_video and result["frame"] is not None:
                        cv2.imshow("Results", result["frame"])
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break

                    pbar.update(1)

            finally:
                pbar.close()
                if save_video and video_writer is not None:
                    video_writer.release()
                if save_in_json:
                    self.json_processor.finalize()
                if show_video:
                    cv2.destroyAllWindows()
        print("\nProcessing complete!")
        print(f"Video info: {video_info}")
        # print(f"Generated {len(json_files)} JSON files:")
        # for file_path in json_files:
        #     print(f"- {file_path}")
    
    def process_frame(self, frame, show_video):
        result_people_detector = self.people_detector.process(
            frame,
            show_video=show_video,
        )

        result_face_detector = self.face_detector.process(
            frame=result_people_detector["frame"],
            people_bboxes=result_people_detector["people_boxes"],
            show_video=show_video,
        )

        result_face_recognizer = self.face_searcher.process(
            frame=result_face_detector["frame"],
            result_dicts=result_face_detector["result_dicts"],
            show_video=show_video,
        )

        return result_face_recognizer


    def test_people_detector_video(self):
        self.process_video(
            "data/input_task.mp4", 60, show_video=True, save_in_json=True
        )

    def test_people_detector_video2(self):
        self.process_video(
            "datasets/videos/1080p_Видео_от_Записи_для_аналитики (5).mp4",
            30,
            show_video=True,
            save_in_json=True,
        )

    def draw_frame_number(self, frame, frame_num):
        """
        Рисует номер кадра в правом верхнем углу изображения

        :param frame: кадр видео (numpy.ndarray)
        :param frame_num: номер кадра (int)
        :return: кадр с нарисованным номером
        """
        # Параметры текста
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        font_thickness = 2
        font_color = (0, 255, 0)  # Зеленый цвет (BGR)

        # Позиция текста (правый верхний угол с отступом)
        text = f"Frame: {frame_num}"
        text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
        text_x = (
            frame.shape[1] - text_size[0] - 20
        )  # Отступ 20 пикселей от правого края
        text_y = text_size[1] + 20  # Отступ 20 пикселей от верхнего края

        # Рисуем текст на кадре
        cv2.putText(
            frame,
            text,
            (text_x, text_y),
            font,
            font_scale,
            font_color,
            font_thickness,
            cv2.LINE_AA,
        )

        return frame
