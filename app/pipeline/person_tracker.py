import numpy as np
from typing import Dict, List, Optional, Union
from collections import defaultdict
from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.utils import yaml_load
from app.utils.config_reader import ConfigReader
import cv2



class PersonTracker:
    def __init__(self, config: Optional[ConfigReader] = None):
        """
        Трекер людей на основе BYTETracker
        
        Args:
            config: Конфигурация трекера. Если None, используются значения по умолчанию.
        """
        if config is None:
            config = ConfigReader()
        
        cfg = yaml_load("app/pipeline/bytetrack.yaml")
        # self.tracker = BYTETracker(args=cfg
        # )
        
        self.track_history = defaultdict(list)
        
        self.box_color = (0, 255, 0)  # Зеленый
        self.text_color = (0, 0, 0)   # Черный
        self.track_color = (0, 255, 255)  # Желтый

    def process(
        self, 
        frame: np.ndarray, 
        people_bboxes: Optional[List[Dict]] = None, 
        show_video: bool = False
    ) -> Dict[str, Union[np.ndarray, List[Dict]]]:
        """
        Обрабатывает кадр с детекциями людей и возвращает треки
        
        Args:
            frame: Входной кадр (BGR)
            people_bboxes: Список детекций в формате [{"x1": x1, "y1": y1, "x2": x2, "y2": y2, "score": score}]
            show_video: Флаг отображения результата
            
        Returns:
            Словарь с результатами:
            {
                "frame": кадр с отрисованными треками (если show_video=True),
                "people_boxes": список треков в формате [{"x1": x1, "y1": y1, "x2": x2, "y2": y2, "track_id": id}]
            }
        """
        result = {"frame": None, "people_boxes": []}
        
        if people_bboxes is None or len(people_bboxes) == 0:
            return result
        
        detections = np.array([
            [bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"], bbox.get("score", 0.5)]
            for bbox in people_bboxes
        ], dtype=np.float32)
        
        tracked_objects = self.tracker.update(detections, frame.shape[:2])
        
        tracked_boxes = []
        for obj in tracked_objects:
            x1, y1, x2, y2, track_id = obj[:5]
            tracked_boxes.append({
                "x1": x1, "y1": y1, 
                "x2": x2, "y2": y2,
                "track_id": int(track_id)
            })
            
            center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
            self.track_history[int(track_id)].append(center)
            if len(self.track_history[int(track_id)]) > 30:
                self.track_history[int(track_id)].pop(0)
        
        result["people_boxes"] = tracked_boxes
        
        if show_video:
            result["frame"] = self._draw_tracks(frame.copy(), tracked_boxes)
            
        return result

    def _draw_tracks(self, frame: np.ndarray, tracked_boxes: List[Dict]) -> np.ndarray:
        """
        Отрисовывает треки на кадре
        
        Args:
            frame: Входной кадр (BGR)
            tracked_boxes: Список треков
            
        Returns:
            Кадр с отрисованными треками
        """
        for box in tracked_boxes:
            x1, y1, x2, y2 = map(int, [box["x1"], box["y1"], box["x2"], box["y2"]])
            track_id = box["track_id"]
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), self.box_color, 2)
            
            cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.text_color, 2)
            
            history = self.track_history.get(track_id, [])
            for i in range(1, len(history)):
                cv2.line(frame, history[i-1], history[i], self.track_color, 2)
        
        return frame