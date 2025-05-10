import datetime
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
from tqdm import tqdm

from app.clients.db_manager import DatabaseManager
from app.clients.json_processor import JsonProcessor
from app.pipeline.base_processor import BaseProcessor
from app.pipeline.face_detector import FaceDetector
from app.pipeline.face_search import FaceSearchService
from app.pipeline.people_detector import PeopleDetector
from app.pipeline.postprocessor import PersonIDPostprocessor
from app.utils.config_reader import ConfigReader


class AutomaticIdentificationPipeline:
    def __init__(self, config: Optional[ConfigReader] = None):
        if config is None:
            config = ConfigReader()

        general_config = config.get_general_config()
        self.json_processor = JsonProcessor(
            output_root=general_config.output_dir, json_size=general_config.json_size
        )
        self.output_dir = general_config.output_dir
        self.people_detector = PeopleDetector(config)
        self.face_detector = FaceDetector(config)
        self.face_searcher = FaceSearchService(config)
        self.postprocessor = PersonIDPostprocessor(window_size=30)

        self.db_manager = DatabaseManager()

    def analyze_face_sizes(self, video_path, section_duration=10):
        """
        Анализирует размеры лиц в начале, середине и конце видео, используя моду (наиболее частые значения).
        Мода вычисляется вручную без использования дополнительных библиотек.
        
        Параметры:
            video_path (str): Путь к видеофайлу
            section_duration (int): Длительность каждой анализируемой секции в секундах
        
        Возвращает:
            bool: True если модальные значения ширины и высоты больше пороговых, иначе False
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print("Ошибка: Не удалось открыть видеофайл")
            return False

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps

        if fps == 0:
            print("Ошибка: Не удалось получить FPS видео")
            return False

        if duration < section_duration * 3:
            print(f"Ошибка: Видео слишком короткое. Нужно минимум {section_duration*3} секунд")
            return False

        frames_per_section = int(section_duration * fps)
        sections = {
            "start": (0, frames_per_section),
            "middle": (
                (total_frames // 2) - (frames_per_section // 2),
                (total_frames // 2) + (frames_per_section // 2),
            ),
            "end": (total_frames - frames_per_section, total_frames),
        }

        width_counts = {}
        height_counts = {}

        for section_name, (start_frame, end_frame) in sections.items():
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            frames_processed = 0
            while frames_processed < (end_frame - start_frame):
                ret, frame = cap.read()
                if not ret:
                    break

                result_people_detector = self.people_detector.process(
                    frame,
                    show_video=False,
                )

                result_face_detector = self.face_detector.process(
                    frame=result_people_detector["frame"],
                    people_bboxes=result_people_detector["people_boxes"],
                    show_video=False,
                )
                
                for person in result_face_detector["result_dicts"]:
                    if person["face_bbox"] is not None:
                        w = abs(person["face_bbox"]["x1"] - person["face_bbox"]["x2"])
                        h = abs(person["face_bbox"]["y1"] - person["face_bbox"]["y2"])
                        
                        if w in width_counts:
                            width_counts[w] += 1
                        else:
                            width_counts[w] = 1
                        
                        if h in height_counts:
                            height_counts[h] += 1
                        else:
                            height_counts[h] = 1

                frames_processed += 1

        cap.release()
        
        if not width_counts or not height_counts:
            return False
        
        def find_mode(value_counts):
            max_count = 0
            mode_value = None
            for value, count in value_counts.items():
                if count > max_count:
                    max_count = count
                    mode_value = value
            return mode_value
        
        modal_width = find_mode(width_counts)
        modal_height = find_mode(height_counts)
        
        return modal_width >= 130 and modal_height >= 200

    def process_video(
        self,
        video_path: str,
        show_video: bool = False,
        save_video: bool = False,
        save_in_json: bool = False,
        show_progress: bool = True,
    ):
        """Обработка видео с сохранением оригинального FPS"""
        analyze_result = self.analyze_face_sizes(video_path)
        if analyze_result == False:
            self.db_manager.insert_video_info(
                {
                    "video_name": str(Path(video_path).name),
                    "original_fps": original_fps,
                    "width": width,
                    "height": height,
                    "total_frames": total_frames,
                    "duration": duration,
                    "processing_date": datetime.datetime.now().isoformat(),
                    "status_of_processing": "ERROR: Sizes of faces in video are too small"
                }
            )
            return None
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        original_fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / original_fps

        print(f"\nProcessing video: {os.path.basename(video_path)}")
        print(
            f"Duration: {duration:.2f}s, Frames: {total_frames}, FPS: {original_fps:.2f}"
        )

        if save_video:
            output_dir = Path(f"{self.output_dir}/{Path(video_path).stem}")
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"processed_{Path(video_path).name}"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_writer = cv2.VideoWriter(
                str(output_path), fourcc, original_fps, (width, height)
            )

        json_files = []
        pbar = tqdm(
            total=total_frames,
            desc="Processing frames",
            unit="frame",
            disable=not show_progress,
        )

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

                result = self.process_frame(frame, show_video)

                if save_video and result["frame"] is not None:
                    result["frame"] = self.draw_frame_number(result["frame"], frame_num)
                    video_writer.write(result["frame"])

                self.db_manager.save_frame_results(
                    video_name=str(Path(video_path).name),
                    frame_number=frame_num,
                    timestamp=timestamp,
                    detections=result["result_dicts"],
                )

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
            cap.release()
            if save_video:
                video_writer.release()
            if save_in_json:
                self.json_processor.finalize()
            if show_video:
                cv2.destroyAllWindows()
            self.db_manager.insert_video_info(
                {
                    "video_name": str(Path(video_path).name),
                    "original_fps": original_fps,
                    "width": width,
                    "height": height,
                    "total_frames": total_frames,
                    "duration": duration,
                    "processing_date": datetime.datetime.now().isoformat(),
                    "status_of_processing": "DONE"
                }
            )
        print("\nProcessing complete!")
        return json_files

    def process_frame(self, frame, show_video):
        """Обработка одного кадра (остается без изменений)"""
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

        result_face_recognizer["result_dicts"] = self.postprocessor.update(
            result_face_recognizer["result_dicts"]
        )

        result_face_recognizer["frame"] = self.postprocessor.draw_id(
            result_face_recognizer["frame"], result_face_recognizer["result_dicts"]
        )

        return result_face_recognizer

    def draw_frame_number(self, frame, frame_num):
        """Рисует номер кадра (остается без изменений)"""
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        font_thickness = 2
        font_color = (0, 255, 0)

        text = f"Frame: {frame_num}"
        text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
        text_x = frame.shape[1] - text_size[0] - 20
        text_y = text_size[1] + 20

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
