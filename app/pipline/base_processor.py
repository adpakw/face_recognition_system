from abc import ABC, abstractmethod
from typing import Any, Dict

import numpy as np


class BaseProcessor(ABC):
    """Абстрактный базовый класс для всех обработчиков"""

    @abstractmethod
    def process(self, frame: np.ndarray, detections: Dict[str, Any]) -> Dict[str, Any]:
        """Основной метод обработки кадра"""
        pass

    @abstractmethod
    def draw(self, frame: np.ndarray, results: Dict[str, Any]) -> np.ndarray:
        """Метод для визуализации результатов"""
        pass
