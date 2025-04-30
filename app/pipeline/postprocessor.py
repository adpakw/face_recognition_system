from collections import defaultdict, deque
from typing import Dict, List, Optional
import numpy as np
import cv2

class PersonIDPostprocessor:
    def __init__(self, window_size: int = 10):
        """
        Постпроцессор для стабилизации идентификации персонажей
        
        Args:
            window_size: Размер скользящего окна (количество кадров)
        """
        self.window_size = window_size
        self.track_history = defaultdict(lambda: deque(maxlen=window_size))
        
    def update(self, detections: List[Dict]) -> List[Dict]:
        """
        Обновляет историю треков и возвращает стабилизированные идентификации
        
        Args:
            detections: Список обнаружений текущего кадра
            
        Returns:
            Список обнаружений с обновленными идентификациями
        """
        for det in detections:
            if ('person_bbox' in det and 
                isinstance(det['person_bbox'], dict) and 
                'track_id' in det['person_bbox']):
                
                track_id = det['person_bbox']['track_id']
                person_id = 'unknown'
                
                if ('person_id' in det and 
                    isinstance(det['person_id'], dict) and 
                    'name' in det['person_id']):
                    
                    person_id = det['person_id']['name']
                
                self.track_history[track_id].append(person_id)
        
        for det in detections:
            if ('person_bbox' in det and 
                isinstance(det['person_bbox'], dict) and 
                'track_id' in det['person_bbox']):
                
                track_id = det['person_bbox']['track_id']
                history = self.track_history[track_id]
                
                if not history:
                    continue
                    
                counts = defaultdict(int)
                for pid in history:
                    counts[pid] += 1
                    
                most_common = max(counts.items(), key=lambda x: x[1])[0]
                
                # current_id = 'unknown'
                if ('person_id' in det and 
                    isinstance(det['person_id'], dict) and 
                    'name' in det['person_id']):
                    
                    current_id = det['person_id']['name']
                
                if det['person_id'] is not None and current_id != most_common:
                    if 'person_id' not in det:
                        det['person_id'] = {}
                    
                    det['person_id']['name'] = most_common
        
        return detections
    
    def draw_id(self, frame: np.ndarray, result_dicts) -> np.ndarray:
        for person in result_dicts:
            if person["person_id"] is None or person["face_bbox"] is None:
                continue

            p_x1, p_y1 = person["person_bbox"]["x1"], person["person_bbox"]["y1"]
            p_x2, p_y2 = person["person_bbox"]["x2"], person["person_bbox"]["y2"]
            text_color = (0, 0, 0)

            person_id = person["person_id"]
            if person_id["match"]:
                text = f"{person_id['name']}"
            else:
                text = "Unknown"

            (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            
            cv2.putText(
                frame,
                text,
                (p_x2 - text_w, p_y1+text_h),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                text_color,
                2,
                cv2.LINE_AA
            )

        return frame