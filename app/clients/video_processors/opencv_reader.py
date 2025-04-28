from typing import Iterator, Optional, Tuple

import cv2
import numpy as np


class OpenCVVideoReader:
    def __init__(self, video_path: str, target_fps: Optional[float] = None):
        """
        Инициализация видеопроцессора

        :param video_path: путь к видеофайлу
        :param target_fps: целевая частота кадров
                          (None - оригинальная частота,
                          если target_fps > original_fps - используется original_fps)
        """
        self.video_path = video_path
        self.target_fps = target_fps
        self.cap = None
        self.original_fps = 0
        self.frame_skip = 0
        self.current_frame_pos = 0
        self.current_time = 0.0

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()

    def open(self):
        """Открытие видеофайла и настройка параметров"""
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise IOError(f"Не удалось открыть видео: {self.video_path}")

        self.original_fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.original_fps <= 0:
            self.original_fps = 30.0  # fallback

        if self.target_fps is not None:
            if self.target_fps < self.original_fps:
                self.frame_skip = int(self.original_fps / self.target_fps) - 1
            else:
                self.target_fps = self.original_fps
                self.frame_skip = 0
        else:
            self.frame_skip = 0

    def release(self):
        """Освобождение ресурсов"""
        if self.cap is not None:
            self.cap.release()
        self.cap = None

    def frames(self) -> Iterator[Tuple[np.ndarray, float, int]]:
        """
        Генератор кадров видео

        :return: итератор кортежей (кадр, timestamp в секундах, номер кадра)
        """
        if self.cap is None:
            self.open()

        self.current_frame_pos = 0
        self.current_time = 0.0
        frame_interval = 1.0 / self.original_fps

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            if self.current_frame_pos % (self.frame_skip + 1) != 0:
                self.current_frame_pos += 1
                self.current_time += frame_interval
                continue

            yield frame, self.current_time, self.current_frame_pos + 1

            self.current_frame_pos += 1
            self.current_time += frame_interval

    def get_video_info(self) -> dict:
        """Получение информации о видео"""
        if self.cap is None:
            self.open()

        return {
            "width": int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "original_fps": self.original_fps,
            "total_frames": int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "duration": int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) / self.original_fps,
            "target_fps": self.target_fps if self.target_fps else self.original_fps,
            "frame_skip": self.frame_skip,
        }
