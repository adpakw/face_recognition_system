import cv2
import numpy as np
import torch
from ultralytics import YOLO
from typing import List, Dict

class YOLODetector:
    def __init__(self, 
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 tracker_config: str = "app/configs/bytetrack.yaml"):
        """
        Инициализация детектора людей с YOLOv8 + ByteTrack (встроенный в Ultralytics)
        
        Args:
            device (str): 'cuda' или 'cpu'
            tracker_config (str): Конфиг трекера (по умолчанию 'bytetrack.yaml')
        """
        self.device = device
        self.model = YOLO("app/models/weights/yolov8n.pt").to(self.device)
        
        self.tracker_config = tracker_config
        
        print(f"YOLOv8 + ByteTrack initialized on \"{self.device.upper()}\"")

    def detect_people(self, image: np.ndarray, confidence_threshold: float = 0.5) -> List[Dict]:
        """
        Детектирование и трекинг людей на изображении
        
        Args:
            image (np.ndarray): Входное изображение (BGR)
            confidence_threshold (float): Порог уверенности для детекции
            
        Returns:
            List[Dict]: Список словарей с bounding boxes и ID треков:
                       [{"x1": x1, "y1": y1, "x2": x2, "y2": y2, 
                         "score": score, "track_id": id}, ...]
        """
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        results = self.model.track(
            image_rgb, 
            persist=True,
            conf=confidence_threshold,
            classes=[0], 
            tracker=self.tracker_config,
            verbose=False
        )
        
        people_boxes = []
        
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            scores = results[0].boxes.conf.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            
            for box, score, track_id in zip(boxes, scores, track_ids):
                x1, y1, x2, y2 = map(int, box[:4])
                people_boxes.append({
                    "x1": x1, "y1": y1,
                    "x2": x2, "y2": y2,
                    "score": float(score),
                    "track_id": int(track_id)
                })
        
        return people_boxes