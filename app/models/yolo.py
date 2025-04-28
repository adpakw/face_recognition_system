import cv2
import numpy as np
import torch
from ultralytics import YOLO
from typing import List, Dict

class YOLODetector:
    def __init__(self, 
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Инициализация детектора людей с YOLOv8

        Args:
            model_path (str): Путь к файлу весов модели (.pt)
            device (str): Устройство для вычислений ('cuda' или 'cpu')
        """
        self.device = device
        self.model = YOLO("app/models/weights/yolov8n.pt").to(self.device)
        print(f"YOLO detector initialized on \"{self.device.upper()}\" device")

    def detect_people(self, image: np.ndarray, confidence_threshold: float = 0.5) -> List[Dict]:
        """
        Детектирование людей на изображении

        Args:
            image (np.ndarray): Входное изображение (BGR)
            confidence_threshold (float): Порог уверенности для детекции

        Returns:
            List[Dict]: Список словарей с bounding boxes людей в формате:
                       [{"x1": x1, "y1": y1, "x2": x2, "y2": y2, "score": score}, ...]
        """
        # Конвертируем BGR в RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Выполняем детекцию
        results = self.model(image_rgb, verbose=False, conf=confidence_threshold)
        
        people_boxes = []
        for result in results:
            # Фильтруем только класс 'person' (обычно class_id=0)
            boxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)
            
            for box, score, class_id in zip(boxes, scores, class_ids):
                if class_id == 0:  # Класс 'person'
                    x1, y1, x2, y2 = map(int, box[:4])
                    people_boxes.append({
                        "x1": x1, "y1": y1, 
                        "x2": x2, "y2": y2, 
                        "score": float(score)
                    })
        
        return people_boxes