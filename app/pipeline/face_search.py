import cv2
import numpy as np
from typing import Dict, List, Optional, Union
from app.clients.image_dataset import ImageDataset
from app.pipeline.face_recognizer import FaceRecognizer
from app.utils.config_reader import ConfigReader


class FaceSearchService:
    def __init__(self, config: Optional[ConfigReader] = None):
        """
        Сервис для поиска, обработки и отрисовки лиц
        
        Args:
            config: Экземпляр ConfigReader. Если None, создаст новый
        """
        if config is None:
            config = ConfigReader()
        self.dataset = ImageDataset(config)
        self.recognizer = FaceRecognizer(config)
        
        recognizer_config = config.get_pipeline_step_config("face_recognizer")
        self.threshold = recognizer_config["cfg"]["confidence_threshold"]

    def process(
        self, 
        frame: np.ndarray, 
        result_dicts: List[Dict], 
        show_video: bool = False
    ) -> Dict[str, Union[np.ndarray, List[Dict]]]:
        """
        Обрабатывает кадр и идентифицирует лица
        
        Args:
            frame: Входной кадр (BGR)
            result_dicts: Список словарей с обнаруженными людьми и лицами
            show_video: Флаг отображения результата
            
        Returns:
            Словарь с результатами:
            {
                "frame": изображение с отрисованными результатами (если show_video=True),
                "result_dicts": список словарей с информацией о распознанных лицах
            }
        """
        result = {"frame": None, "result_dicts": []}
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        output_frame = frame.copy()

        for person in result_dicts:
            person_data = {"person_bbox": person["person_bbox"], "face_bbox": person["face_bbox"]}
            
            if person["face_bbox"] is None:
                person_data["person_id"] = None
                result["result_dicts"].append(person_data)
                continue

            face_emb = self._extract_face_embedding(frame_rgb, person["face_bbox"])
            
            scores, _, names = self.dataset.search(face_emb, threshold=self.threshold)
            
            person_data["person_id"] = {
                "name": names[0] if len(names) > 0 else "unknown",
                "score": float(scores[0]) if len(scores) > 0 else 0.0,
                "match": len(names) > 0
            }
            result["result_dicts"].append(person_data)

            if show_video:
                output_frame = self._draw_person_info(output_frame, person_data)

        if show_video:
            result["frame"] = output_frame

        return result

    def _extract_face_embedding(self, frame_rgb: np.ndarray, bbox: Dict[str, int]) -> np.ndarray:
        """Извлекает эмбеддинг лица из bounding box"""
        x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
        face_img = frame_rgb[y1:y2, x1:x2]
        
        return self.recognizer.extract_embedding(face_img)

    def _draw_person_info(self, frame: np.ndarray, person: Dict) -> np.ndarray:
        """
        Отрисовывает информацию о персоне на кадре
        
        Args:
            frame: Исходное изображение (BGR)
            person: Словарь с информацией о персоне
            
        Returns:
            Изображение с отрисованной информацией
        """
        if person["person_id"] is None or person["face_bbox"] is None:
            return frame

        p_x1, p_y1 = person["person_bbox"]["x1"], person["person_bbox"]["y1"]
        p_x2, p_y2 = person["person_bbox"]["x2"], person["person_bbox"]["y2"]
        f_x1, f_y1 = person["face_bbox"]["x1"], person["face_bbox"]["y1"]
        f_x2, f_y2 = person["face_bbox"]["x2"], person["face_bbox"]["y2"]

        
        person_color = (0, 255, 0)
        face_color = (0, 165, 255)
        text_color = (0, 0, 0)   
        text_bg = (255, 255, 255)

        cv2.rectangle(frame, (p_x1, p_y1), (p_x2, p_y2), person_color, 2)
        
        cv2.rectangle(frame, (f_x1, f_y1), (f_x2, f_y2), face_color, 2)

        person_id = person["person_id"]
        if person_id["match"]:
            text = f"{person_id['name']} ({person_id['score']:.2f})"
        else:
            text = "Unknown"

        (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        text_y = max(p_y1 - 10, text_h + 5)
        
        cv2.putText(
            frame,
            text,
            (p_x2 - text_w, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            text_color,
            2,
            cv2.LINE_AA
        )

        return frame