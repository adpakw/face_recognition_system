import cv2
import numpy as np
from typing import Optional

class SFace:
    def __init__(self, device: str = "cuda"):
        """
        Инициализация модели SFace для извлечения эмбеддингов лиц.
        
        Args:
            device (str): Устройство для выполнения вычислений ('cuda' или 'cpu')
        """
        self.device = device.lower()
        self.model = self._load_model()
        
    def _load_model(self):
        """Загрузка предварительно обученной модели SFace"""
        # Укажите правильный путь к модели SFace
        model_path = "app/models/weights/face_recognition_sface_2021dec.onnx"  # или другой путь
        
        try:
            # Определяем backend и target в зависимости от устройства
            if self.device == "cuda":
                if not cv2.cuda.getCudaEnabledDeviceCount():
                    print("CUDA не доступна, используется CPU")
                    self.device = "cpu"
                    backend = cv2.dnn.DNN_BACKEND_OPENCV
                    target = cv2.dnn.DNN_TARGET_CPU
                else:
                    backend = cv2.dnn.DNN_BACKEND_CUDA
                    target = cv2.dnn.DNN_TARGET_CUDA
            else:
                backend = cv2.dnn.DNN_BACKEND_OPENCV
                target = cv2.dnn.DNN_TARGET_CPU

            model = cv2.FaceRecognizerSF.create(
                model=model_path,
                config="",
                backend_id=backend,
                target_id=target
            )
            return model
        except Exception as e:
            raise RuntimeError(f"Failed to load SFace model: {str(e)}. Please check if model file exists at {model_path}")
    
    def get_embeddings(self, face_image: np.ndarray, bbox: tuple = None) -> np.ndarray:
        """
        Извлечение эмбеддингов лица из изображения.
        
        Args:
            face_image (np.ndarray): Изображение лица в RGB формате
            bbox (tuple): Координаты bounding box (x1, y1, x2, y2)
            
        Returns:
            np.ndarray: Вектор эмбеддинга лица (1D массив)
        """
        if not hasattr(self, 'model') or self.model is None:
            raise RuntimeError("SFace model not initialized")
            
        # Преобразование изображения
        face_img = cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR)
        
        # Если передан bounding box, вырезаем область лица
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            face_img = face_img[y1:y2, x1:x2]
        
        # Выравнивание лица и извлечение признаков
        face_align = self.model.alignCrop(face_img, np.zeros(1))  # Второй аргумент - landmarks
        embedding = self.model.feature(face_align)
        return embedding.flatten()