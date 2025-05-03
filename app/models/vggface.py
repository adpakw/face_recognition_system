import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchfile
import torchvision
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    Convolution2D,
    ZeroPadding2D,
    MaxPooling2D,
    Flatten,
    Dropout,
    Activation,
)
from typing import Any, Dict, Optional, Union, List, Tuple

import numpy as np

class VggFaceClient:
    """
    VGG-Face model class with preprocessing
    """

    def __init__(self):
        self.model = base_model()
        self.model.load_weights("app/models/weights/vgg_face_weights.h5")
        self.model_name = "VGG-Face"
        self.input_shape = (224, 224)
        self.output_shape = 4096
        
        # Параметры нормализации для VGG-Face (BGR-формат)
        self.mean = np.array([91.4953, 103.8827, 131.0912], dtype=np.float32)
        self.scale = 1.0 / 255  # Опционально, зависит от тренировочного пайплайна

    def preprocess(self, img: np.ndarray) -> np.ndarray:
        """
        Препроцессинг изображения для VGG-Face:
        1. Ресайз до (224, 224)
        2. Конвертация цветового пространства (если нужно)
        3. Нормализация
        4. Добавление batch-размерности
        """
        # Ресайз с сохранением пропорций (опционально)
        img = cv2.resize(img, self.input_shape, interpolation=cv2.INTER_LINEAR)
        
        # Конвертация RGB -> BGR (если изображение загружено через PIL)
        if img.shape[-1] == 3:  # Проверка на RGB
            img = img[..., ::-1]  # Конвертация в BGR
        
        # Нормализация
        img = img.astype(np.float32)
        img -= self.mean  # Вычитание средних значений
        img *= self.scale  # Масштабирование (если используется)
        
        # Добавление batch-размерности [1, H, W, C]
        return np.expand_dims(img, axis=0)

    def get_embeddings(self, img: np.ndarray) -> List[float]:
        """
        Генерация эмбеддингов с препроцессингом:
        """
        # Препроцессинг
        processed_img = self.preprocess(img)
        
        # Инференс модели
        embedding = self.model.predict(processed_img, verbose=0)[0]
        
        # L2-нормализация эмбеддингов
        return self.l2_normalize(embedding).tolist()

    def l2_normalize(self,
        x: Union[np.ndarray, list], axis: Union[int, None] = None, epsilon: float = 1e-10
    ) -> np.ndarray:
        """L2-нормализация векторов"""
        x = np.asarray(x)
        norm = np.linalg.norm(x, axis=axis, keepdims=True)
        return x / (norm + epsilon)


def base_model() -> Sequential:
    """
    Base model of VGG-Face being used for classification - not to find embeddings
    Returns:
        model (Sequential): model was trained to classify 2622 identities
    """
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(224, 224, 3)))
    model.add(Convolution2D(64, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, (3, 3), activation="relu"))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, (3, 3), activation="relu"))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, (3, 3), activation="relu"))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation="relu"))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation="relu"))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Convolution2D(4096, (7, 7), activation="relu"))
    model.add(Dropout(0.5))
    model.add(Convolution2D(4096, (1, 1), activation="relu"))
    model.add(Dropout(0.5))
    model.add(Convolution2D(2622, (1, 1)))
    model.add(Flatten())
    model.add(Activation("softmax"))

    return model


