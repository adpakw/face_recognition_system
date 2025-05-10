from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from tensorflow.keras.layers import (Activation, Convolution2D, Dropout,
                                     Flatten, MaxPooling2D, ZeroPadding2D)
from tensorflow.keras.models import Model, Sequential


class VggFaceClient:
    """
    VGG-Face model class with preprocessing
    """

    def __init__(self):
        self.model = base_model()
        self.model.load_weights("app/models/weights/vgg_face_weights.h5")
        self.input_shape = (224, 224)

        self.mean = np.array([91.4953, 103.8827, 131.0912], dtype=np.float32)

    def preprocess(self, img: np.ndarray) -> np.ndarray:
        img = cv2.resize(img, self.input_shape, interpolation=cv2.INTER_LINEAR)

        img = img.astype(np.float32)
        img -= self.mean
        return np.expand_dims(img, axis=0)

    def get_embeddings(self, img: np.ndarray) -> List[float]:
        processed_img = self.preprocess(img)
        embedding = self.model.predict(processed_img, verbose=0)[0]
        return self.l2_normalize(embedding).tolist()

    def l2_normalize(
        self,
        x: Union[np.ndarray, list],
        axis: Union[int, None] = None,
        epsilon: float = 1e-10,
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
