import json
import os
from datetime import datetime
from typing import Any, Dict, List


class JsonProcessor:
    def __init__(self, output_root: str = "detection_results", json_size: int = 512 * 1024):
        """
        Инициализация процессора для работы с JSON

        :param output_root: корневая директория для сохранения результатов
        """
        self.output_root = output_root
        os.makedirs(self.output_root, exist_ok=True)

        self.json_size = json_size
        self.current_file_path = None
        self.current_file_size = 0
        self.current_detections = []
        self.current_start_time = None
        self.current_end_time = None
        self.video_name = None

    def _get_video_output_dir(self, video_path: str) -> str:
        """Создает и возвращает путь к папке с именем видео"""
        self.video_name = os.path.splitext(os.path.basename(video_path))[0]
        output_dir = os.path.join(self.output_root, self.video_name)
        os.makedirs(output_dir, exist_ok=True)
        return output_dir

    def _create_new_file(self, timestamp: float) -> str:
        """Создает новый файл для записи"""
        self.current_start_time = timestamp
        self.current_end_time = timestamp
        filename = f"{timestamp:.1f}.json"

        output_dir = self._get_video_output_dir(self.current_video_path)
        self.current_file_path = os.path.join(output_dir, filename)
        self.current_file_size = 0
        self.current_detections = []

        return self.current_file_path

    def _save_current_file(self):
        """Сохраняет текущие данные в файл"""
        if not self.current_detections:
            return

        data = {
            "video_path": self.current_video_path,
            "start_time": self.current_start_time,
            "end_time": self.current_end_time,
            "processing_date": datetime.now().isoformat(),
            "frames": self.current_detections,
        }

        with open(self.current_file_path, "w") as f:
            json.dump(data, f, indent=2)

    def _estimate_size(self, detection_data: Dict[str, Any]) -> int:
        """Оценивает размер данных в байтах"""
        return len(json.dumps(detection_data).encode("utf-8"))

    def save_detections(
        self, video_path: str, frame_num: int, timestamp: float, people
    ) -> str:
        """
        Сохраняет данные детектирования в JSON файлы (не более 10MB каждый)

        :param video_path: путь к видеофайлу
        :param frame_num: номер кадра
        :param timestamp: временная метка в секундах
        :param boxes_with_scores: список детекций (bbox, score)
        :param frame_size: размер кадра (width, height)
        :return: путь к последнему сохраненному файлу
        """
        self.current_video_path = video_path

        detection_data = {
            "frame_number": frame_num,
            "timestamp": round(timestamp, 3),
            "people": [
                {
                    "person": {
                        "x1": int(person["person_bbox"]["x1"]),
                        "y1": int(person["person_bbox"]["y1"]),
                        "x2": int(person["person_bbox"]["x2"]),
                        "y2": int(person["person_bbox"]["y2"]),
                        "confidence": round(float(person["person_bbox"]["score"]), 5)
                    },
                    "face": {
                        "x1": int(person["face_bbox"]["x1"]),
                        "y1": int(person["face_bbox"]["y1"]),
                        "x2": int(person["face_bbox"]["x2"]),
                        "y2": int(person["face_bbox"]["y2"]),
                        "confidence": round(float(person["face_bbox"]["score"]), 5)
                    } if person.get("face_bbox") is not None else None,
                }
                for person in people
            ],
        }

        new_data_size = self._estimate_size(detection_data)

        if (
            self.current_file_path is None
            or self.current_file_size + new_data_size > self.json_size
        ):

            if self.current_file_path is not None:
                self._save_current_file()

            self._create_new_file(timestamp)
            self.current_file_size = 0

        self.current_detections.append(detection_data)
        self.current_file_size += new_data_size
        self.current_end_time = timestamp

        return self.current_file_path

    def finalize(self):
        """Завершает запись, сохраняя оставшиеся данные"""
        if self.current_detections:
            self._save_current_file()
        self.current_file_path = None
        self.current_detections = []
