from app.clients.postgres import PostgreSQLClient
import os
from typing import List, Dict


class DatabaseManager:
    def execute_sql_files_from_folder(self) -> None:
        """
        Чтение и выполнение всех SQL-файлов из указанной папки
        """
        folder_path = "migrations"
        if not os.path.isdir(folder_path):
            raise ValueError(f"Папка не найдена: {folder_path}")
        
        sql_files = [f for f in os.listdir(folder_path) 
                    if f.endswith('.sql') and os.path.isfile(os.path.join(folder_path, f))]
        
        if not sql_files:
            print(f"В папке {folder_path} не найдено .sql файлов")
            return
        
        sql_files.sort()
    
        for sql_file in sql_files:
            file_path = os.path.join(folder_path, sql_file)
            print(f"Выполнение файла: {sql_file}")
            try:
                with PostgreSQLClient() as client:
                    client.execute_sql_file(file_path)
            except Exception as e:
                print(f"Ошибка при выполнении файла {sql_file}: {e}")
                raise

    def save_frame_results(self, video_name: str, frame_number: int, 
                         timestamp: float, detections: List[Dict]) -> None:
        """
        Сохраняет результаты анализа кадра в БД
        
        :param video_name: имя видеофайла
        :param frame_number: номер кадра
        :param timestamp: временная метка в секундах
        :param detections: список обнаружений людей (результат из process_frame)
        """
        records = []
        
        for detection in detections:
            # Подготовка данных для вставки
            record = {
                'video_name': video_name,
                'frame_number': frame_number,
                'timestamp': timestamp,
                
                # Данные о человеке
                'person_bbox_x1': detection['person_bbox']['x1'],
                'person_bbox_y1': detection['person_bbox']['y1'],
                'person_bbox_x2': detection['person_bbox']['x2'],
                'person_bbox_y2': detection['person_bbox']['y2'],
                'person_detection_conf': float(detection['person_bbox']['score']),
                
                # Данные о лице (может быть None)
                'face_bbox_x1': detection['face_bbox']['x1'] if detection['face_bbox'] else None,
                'face_bbox_y1': detection['face_bbox']['y1'] if detection['face_bbox'] else None,
                'face_bbox_x2': detection['face_bbox']['x2'] if detection['face_bbox'] else None,
                'face_bbox_y2': detection['face_bbox']['y2'] if detection['face_bbox'] else None,
                'face_detection_conf': float(detection['face_bbox']['score']) if detection['face_bbox'] else None,
                
                # Данные идентификации
                'person_name': detection['person_id']['name'] if detection['person_id'] else None,
                'person_identification_conf': float(detection['person_id']['score']) if detection['person_id'] else None
            }
            records.append(record)
        
        # Вставка данных в БД
        with PostgreSQLClient() as client:
            client.insert('video_analysis_results', records)