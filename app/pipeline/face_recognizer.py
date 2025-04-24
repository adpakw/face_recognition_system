# app/pipeline/face_recognizer.py
from typing import Dict, Any
import numpy as np
from app.utils.config_reader import ConfigReader
from app.models.arcface import ArcFace
from app.models.vgg_face import VGG_16
from app.models.sface import SFace


class FaceRecognizer:
    def __init__(self):
        """
        Класс только для извлечения эмбеддингов лиц
        """
        config = ConfigReader()
        recognizer_config = config.get_pipeline_step_config("face_recognizer")
        model_name = config.get_config().pipeline["face_recognizer"].model
        self._init_model(model_name, recognizer_config)

    def _init_model(self, model_name: str, config: Dict[str, Any]):
        device = config['cfg']["device"]
        
        if model_name == "ArcFace":
            self.model = ArcFace(device)
        elif model_name == "VGG-Face":
            self.model = VGG_16(device)
        elif model_name == "SFace":
            self.model = SFace(device)
        else:
            raise ValueError(f"Unsupported face recognition model: {model_name}")

    def extract_embedding(self, frame_rgb: np.ndarray) -> np.ndarray:
        """Извлекает эмбеддинг лица из bounding box"""

        return self.model.get_embeddings(frame_rgb)