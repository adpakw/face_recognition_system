import json
from typing import Dict, List, Optional


class FrameFinder:
    def __init__(self, json_files: List[str]):
        """
        Инициализация поисковика детекций с кэшированием структуры файлов

        :param json_files: список путей к JSON файлам с результатами
        """
        self.json_files = sorted(json_files)
        self.file_index = []  # (start_frame, end_frame, file_path)
        self._build_index()

    def _build_index(self):
        """Построение индекса файлов для быстрого поиска"""
        for file_path in self.json_files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    frames = data.get("frames", [])
                    if frames:
                        start_frame = frames[0]["frame_number"]
                        end_frame = frames[-1]["frame_number"]
                        self.file_index.append((start_frame, end_frame, file_path))
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Ошибка при индексации файла {file_path}: {str(e)}")

    def find_frames(self, target_frame: int) -> Optional[List[Dict]]:
        """
        Быстрый поиск детекций для указанного кадра

        :param target_frame: номер кадра для поиска
        :return: список детекций или None, если кадр не найден
        """
        for start, end, file_path in self.file_index:
            if start <= target_frame <= end:
                break

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                for frame_data in data['frames']:
                    if frame_data["frame_number"] == target_frame:
                        return frame_data
                    if frame_data["frame_number"] > target_frame:
                        break
        except (json.JSONDecodeError, KeyError):
            return None
        return None
