from typing import Dict, Any,Optional, Union, Tuple
import numpy as np
from app.utils.config_reader import ConfigReader
from app.models.facenet import FaceNet128dClient, FaceNet512dClient
# from app.models.vgg_face import VGG_16
from app.models.sface import SFace
# from deepface import DeepFace
import cv2

class FaceRecognizer:
    def __init__(self, config: Optional[ConfigReader] = None):
        """
        Класс только для извлечения эмбеддингов лиц
        """
        if config is None:
            config = ConfigReader()
        recognizer_config = config.get_pipeline_step_config("face_recognizer")
        self.model_name = config.get_config().pipeline["face_recognizer"].model
        self._init_model(self.model_name, recognizer_config)
        models = [
            "VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace",
            "DeepID", "ArcFace", "Dlib", "SFace", "GhostFaceNet",
            "Buffalo_L",
        ]

    def _init_model(self, model_name: str, config: Dict[str, Any]):
        device = config['cfg']["device"]
        
        if model_name == "Facenet":
            self.model = FaceNet128dClient()
        elif model_name == "Facenet512":
            self.model = FaceNet512dClient()
        elif model_name == "VGG-Face":
            # self.model = VGG_16(device)
            pass
        elif model_name == "SFace":
            self.model = SFace(device)
        else:
            raise ValueError(f"Unsupported face recognition model: {model_name}")

    def extract_embedding(self, frame_rgb: np.ndarray) -> np.ndarray:
        """Извлекает эмбеддинг лица из bounding box"""

        # return self.model.get_embeddings(frame_rgb)
        return self.model.get_embeddings(frame_rgb)
    
    def align_img_wrt_eyes(self, 
        img: np.ndarray,
        left_eye: Optional[Union[list, tuple]],
        right_eye: Optional[Union[list, tuple]],
    ) -> Tuple[np.ndarray, float]:
        """
        Align a given image horizantally with respect to their left and right eye locations
        Args:
            img (np.ndarray): pre-loaded image with detected face
            left_eye (list or tuple): coordinates of left eye with respect to the person itself
            right_eye(list or tuple): coordinates of right eye with respect to the person itself
        Returns:
            img (np.ndarray): aligned facial image
        """
        # if eye could not be detected for the given image, return image itself
        if left_eye is None or right_eye is None:
            return img, 0

        # sometimes unexpectedly detected images come with nil dimensions
        if img.shape[0] == 0 or img.shape[1] == 0:
            return img, 0

        angle = float(np.degrees(np.arctan2(left_eye[1] - right_eye[1], left_eye[0] - right_eye[0])))

        (h, w) = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        img = cv2.warpAffine(
            img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0)
        )

        return img, angle