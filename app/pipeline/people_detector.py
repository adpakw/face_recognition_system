import os
from typing import Any, Dict, List, Tuple, Optional

import cv2
from tqdm import tqdm
from pathlib import Path

from app.clients.json_processor import JsonProcessor
from app.clients.video_processors.opencv_reader import OpenCVVideoReader
from app.utils.config_reader import ConfigReader

from app.models.ssd import SSD
from app.models.yolo import YOLODetector

class PeopleDetector:
    def __init__(self, config: Optional[ConfigReader] = None):
        """
        Трекер людей на основе BYTETracker
        
        Args:
            config: Конфигурация трекера. Если None, используются значения по умолчанию.
        """
        if config is None:
            config = ConfigReader()

        # Получаем конфиг для этапа people_detector
        detector_config = config.get_pipeline_step_config("people_detector")
        
        # Динамически выбираем модель на основе конфига
        model_name = config.get_config().pipeline["people_detector"].model
        self.detection_model = self._init_model(model_name, detector_config)
        self.confidence_threshold = detector_config["cfg"]["confidence_threshold"]

    def _init_model(self, model_name: str, config: Dict[str, Any]):
        """Инициализирует модель детекции на основе имени и конфига"""
        if model_name == "SSD":
            return SSD(device=config['cfg']["device"])
        elif model_name == "Yolo":
            return YOLODetector(device=config['cfg']["device"])
        else:
            raise ValueError(f"Unsupported people detection model: {model_name}")

    def draw_detections(self, image, detections):
        """Отрисовка bounding boxes и скоров на изображении"""
        for detection in detections:
            cv2.rectangle(image, (detection['x1'], detection['y1']), (detection['x2'], detection['y2']), (0, 255, 0), 2)
            label = f"{detection['score']:.2f}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)

            cv2.rectangle(image, (detection['x1'], detection['y1'] - 20), (detection['x1'] + w, detection['y1']), (0, 255, 0), -1)

            cv2.putText(
                image, label, (detection['x1'], detection['y1'] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1
            )

        return image

    def process(self, frame, show_video=False):
        """Обработка одного кадра"""
        result = {"frame": None, "people_boxes": None}

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result["people_boxes"] = self.detection_model.detect_people(
            frame_rgb, 
            confidence_threshold=self.confidence_threshold
        )

        if show_video:
            result_frame = self.draw_detections(frame.copy(), result["people_boxes"])
            result["frame"] = result_frame

        return result