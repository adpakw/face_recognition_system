import os
from typing import Dict, List, Optional

import cv2
import numpy as np
from tqdm import tqdm

from app.clients.frame_finder import FrameFinder


class OpenCVVideoWriter:
    def __init__(self, output_dir: str = "output_videos"):
        """
        Инициализация видео-райтера

        :param output_dir: директория для сохранения результирующих видео
        """
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def get_json_files_from_folder(self, folder_path: str) -> List[str]:
        """
        Получает все JSON файлы из указанной папки

        :param folder_path: путь к папке для поиска JSON файлов
        :return: список полных путей к JSON файлам
        """
        json_files = []

        # Проверяем, существует ли папка
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Папка не найдена: {folder_path}")

        # Проверяем, что это действительно папка
        if not os.path.isdir(folder_path):
            raise NotADirectoryError(
                f"Указанный путь не является папкой: {folder_path}"
            )

        # Проходим по всем файлам в папке
        for filename in os.listdir(folder_path):
            full_path = os.path.join(folder_path, filename)

            # Проверяем, что это файл и имеет расширение .json
            if os.path.isfile(full_path) and filename.lower().endswith(".json"):
                json_files.append(full_path)

        # Сортируем файлы по имени (хронологически)
        json_files.sort()

        return json_files

    def _draw_people_detections(
        self, frame: np.ndarray, data: List[Dict]
    ) -> np.ndarray:
        """Отрисовка детекций на кадре"""
        for el in data:
            bbox = el["person"]
            x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]

            # Рисуем bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Добавляем подпись с confidence
            label = f"Person: {el['confidence']:.2f}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(frame, (x1, y1 - 20), (x1 + w, y1), (0, 255, 0), -1)
            cv2.putText(
                frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1
            )

        return frame

    def get_result_video(
        self,
        video_path: str,
        json_path: str,
        output_video_name: Optional[str] = None,
        show_progress: bool = True,
        preview: bool = False,
    ) -> str:
        """
        Создание видео с отрисованными результатами детекции

        :param json_path: путь к JSON файлу с результатами
        :param output_video_name: имя выходного видеофайла (None - автоматическое)
        :param show_progress: показывать прогресс-бар
        :param preview: показывать видео в реальном времени
        :return: путь к сохраненному видеофайлу
        """
        json_files = self.get_json_files_from_folder(json_path)
        frame_finder = FrameFinder(json_files)

        # Создаем имя для выходного файла
        if output_video_name is None:
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            output_video_name = f"{video_name}_result.mp4"
        output_path = os.path.join(self.output_dir, output_video_name)

        # Открываем исходное видео
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Не удалось открыть видео: {video_path}")

        # Получаем параметры видео
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Создаем VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        try:
            # Инициализируем прогресс-бар
            pbar = tqdm(
                total=total_frames, desc="Rendering video", disable=not show_progress
            )

            frame_idx = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_idx += 1

                frame_data = frame_finder.find_frames(frame_idx)["people"]

                # Отрисовываем детекции
                frame_with_people_boxes = self._draw_people_detections(
                    frame.copy(), frame_data
                )

                frame_result = frame_with_people_boxes

                # Записываем кадр в выходное видео
                out.write(frame_result)

                # Показываем превью если нужно
                if preview:
                    cv2.imshow("Preview", frame_result)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                pbar.update(1)

        finally:
            # Освобождаем ресурсы
            pbar.close()
            cap.release()
            out.release()
            if preview:
                cv2.destroyAllWindows()

        print(f"\nРезультат сохранен в: {output_path}")
        return output_path
