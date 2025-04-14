import os
from typing import Any, Dict, List, Tuple

import cv2
from tqdm import tqdm
from pathlib import Path
from app.clients.json_processor import JsonProcessor
from app.clients.video_processors.opencv_reader import OpenCVVideoReader
from app.pipline.base_processor import BaseProcessor
from app.pipline.people_detector import PeopleDetector
from app.pipline.face_detector import FaceDetector
from app.utils.config_reader import ConfigReader


class AutomaticIdentificationPipeline:
    def __init__(self, config_path: str):
        """
        :param processors: список обработчиков в порядке их выполнения
        """
        self.config_reader = ConfigReader(config_path=Path(config_path))

        general_config = self.config_reader.get_general_config()
        self.json_processor = JsonProcessor(
            output_root=general_config.output_dir, json_size=general_config.json_size
        )

        people_detector_config = self.config_reader.get_people_detector_config()
        self.people_detector: PeopleDetector = PeopleDetector(
            device=people_detector_config.device,
            confidence_threshold=people_detector_config.confidence_threshold,
        )

        face_detector_config = self.config_reader.get_face_detector_config()
        self.face_detector: FaceDetector = FaceDetector(
            model_path=face_detector_config.model_path,
            score_threshold=face_detector_config.score_threshold,
            nms_threshold=face_detector_config.nms_threshold,
            top_k=face_detector_config.top_k,
            device=face_detector_config.device,
            confidence_threshold=face_detector_config.confidence_threshold,
        )

    def process_video(
        self,
        video_path: str,
        target_fps: float = 10,
        show_video: bool = False,
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
        json_files = []
        with OpenCVVideoReader(video_path, target_fps=target_fps) as vp:
            video_info = vp.get_video_info()

            total_frames = int(
                video_info["total_frames"]
                / (video_info["original_fps"] / video_info["target_fps"])
            )
            duration = video_info["duration"]

            print(f"\nProcessing video: {os.path.basename(video_path)}")
            print(
                f"Duration: {duration:.2f}s, Frames: {total_frames}, Target FPS: {target_fps}, Original FPS {video_info["original_fps"]}"
            )

            pbar = tqdm(
                total=total_frames,
                desc="Processing frames",
                unit="frame",
                disable=not show_progress,
            )

            try:
                for frame, timestamp, frame_num in vp.frames():
                    result_people_detector = self.people_detector.process(
                        frame,
                        show_video=show_video,
                    )

                    result_face_detector = self.face_detector.process(
                        frame=result_people_detector["frame"],
                        people_bboxes=result_people_detector["people_boxes"],
                        show_video=show_video,
                    )

                    result = result_face_detector

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
                if save_in_json:
                    self.json_processor.finalize()
                if show_video:
                    cv2.destroyAllWindows()

        print("\nProcessing complete!")
        print(f"Video info: {video_info}")
        # print(f"Generated {len(json_files)} JSON files:")
        # for file_path in json_files:
        #     print(f"- {file_path}")

    def test_people_detector_video(self):
        self.process_video(
            "data/input_task.mp4", 60, show_video=True, save_in_json=True
        )

    def test_people_detector_video2(self):
        self.process_video(
            "data/1080p_Видео_от_Записи_для_аналитики (5).mp4",
            30,
            show_video=True,
            save_in_json=True,
        )
