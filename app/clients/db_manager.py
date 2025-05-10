import os
from typing import Dict, List

from app.clients.postgres import PostgreSQLClient
from typing import List, Tuple, Optional
from pydantic import BaseModel
from pandas import DataFrame

class TimeInterval(BaseModel):
    video_name: str
    start_time: float
    end_time: float
    frame_count: int
    first_frame: int
    last_frame: int

class PersonIntervalsResponse(BaseModel):
    person_name: str
    intervals: List[TimeInterval]
    total_intervals: int
    total_videos: int
    total_frames: int

class DatabaseManager:
    def execute_sql_files_from_folder(self) -> None:
        """
        Чтение и выполнение всех SQL-файлов из указанной папки
        """
        folder_path = "migrations"
        if not os.path.isdir(folder_path):
            raise ValueError(f"Папка не найдена: {folder_path}")

        sql_files = [
            f
            for f in os.listdir(folder_path)
            if f.endswith(".sql") and os.path.isfile(os.path.join(folder_path, f))
        ]

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

    def save_frame_results(
        self,
        video_name: str,
        frame_number: int,
        timestamp: float,
        detections: List[Dict],
    ) -> None:
        """
        Сохраняет результаты анализа кадра в БД

        :param video_name: имя видеофайла
        :param frame_number: номер кадра
        :param timestamp: временная метка в секундах
        :param detections: список обнаружений людей (результат из process_frame)
        """
        records = []

        for detection in detections:
            record = {
                "video_name": video_name,
                "frame_number": frame_number,
                "timestamp": timestamp,
                "person_bbox_x1": detection["person_bbox"]["x1"],
                "person_bbox_y1": detection["person_bbox"]["y1"],
                "person_bbox_x2": detection["person_bbox"]["x2"],
                "person_bbox_y2": detection["person_bbox"]["y2"],
                "person_detection_conf": float(detection["person_bbox"]["score"]),
                "face_bbox_x1": (
                    detection["face"]["x1"] if detection["face"] else None
                ),
                "face_bbox_y1": (
                    detection["face"]["y1"] if detection["face"] else None
                ),
                "face_bbox_x2": (
                    detection["face"]["x2"] if detection["face"] else None
                ),
                "face_bbox_y2": (
                    detection["face"]["y2"] if detection["face"] else None
                ),
                "face_detection_conf": (
                    float(detection["face"]["score"])
                    if detection["face"]
                    else None
                ),
                "person_name": (
                    detection["person_id"]["name"] if detection["person_id"] else None
                ),
                "person_identification_conf": (
                    float(detection["person_id"]["score"])
                    if detection["person_id"]
                    else None
                ),
            }
            records.append(record)

        with PostgreSQLClient() as client:
            client.insert("video_analysis_results", records)

    def insert_new_persons(self, person_ids) -> None:
        records = []
        with PostgreSQLClient() as client:
            person_ids_in_db = client.execute_query("SELECT * FROM people")["person_id"]
        new_persons = list(set(person_ids) - set(person_ids_in_db))
        for person in new_persons:
            records.append({"person_id": person})

        with PostgreSQLClient() as client:
            client.insert("people", records)

    def insert_video_info(self, video_info) -> None:
        with PostgreSQLClient() as client:
            client.insert("videos", [video_info])
    
    def get_new_videos(self):
        with PostgreSQLClient() as client:
            videos_in_db = client.execute_query("SELECT * FROM videos")["video_name"]

            all_files = os.listdir("videos")

            videos = [file for file in all_files if file.endswith(".mp4")]
            new_videos = list(set(videos) - set(videos_in_db))
            return new_videos
    
    def get_all_people(self):
        with PostgreSQLClient() as client:
            people = client.execute_query(query="""SELECT person_id FROM people""", return_df=True)
            return people
    
    def get_all_videos(self):
        with PostgreSQLClient() as client:
            videos = client.execute_query(query="""SELECT * FROM videos""", return_df=True)
            return videos
    
    def get_top_10(self):
        with PostgreSQLClient() as client:
            top_10 = client.execute_query(query="""SELECT count(person_name), person_name FROM video_analysis_results
                 WHERE person_name <> 'None' and person_name <> 'unknown'
                 GROUP BY person_name
                 ORDER BY count(person_name) DESC
                 LIMIT 10""", return_df=True)
            top_10['rank'] = top_10.index + 1
            return top_10



    def find_continuous_intervals(self, person_name: str) -> Tuple[List[TimeInterval], int, int]:
        with PostgreSQLClient() as db:
            query = """
                SELECT 
                    video_name, 
                    frame_number, 
                    timestamp
                FROM video_analysis_results
                WHERE person_name = %s
                ORDER BY video_name, frame_number
            """
            data = db.execute_query(query, (person_name,))
        if data.empty:
            return [], 0, 0
        
        intervals = []
        current_video = None
        current_start = None
        current_start_frame = None
        current_end = None
        current_end_frame = None
        expected_next_frame = None
        
        for _, row in data.iterrows():
            video = row['video_name']
            frame = row['frame_number']
            timestamp = row['timestamp']
            
            if video != current_video:
                if current_start is not None:
                    interval = TimeInterval(
                        video_name=current_video,
                        start_time=current_start,
                        end_time=current_end,
                        frame_count=current_end_frame - current_start_frame + 1,
                        first_frame=current_start_frame,
                        last_frame=current_end_frame
                    )
                    intervals.append(interval)
                
                current_video = video
                current_start = timestamp
                current_start_frame = frame
                current_end = timestamp
                current_end_frame = frame
                expected_next_frame = frame + 1
            else:
                if frame == expected_next_frame:
                    current_end = timestamp
                    current_end_frame = frame
                    expected_next_frame = frame + 1
                else:
                    if current_start is not None:
                        interval = TimeInterval(
                            video_name=current_video,
                            start_time=current_start,
                            end_time=current_end,
                            frame_count=current_end_frame - current_start_frame + 1,
                            first_frame=current_start_frame,
                            last_frame=current_end_frame
                        )
                        intervals.append(interval)
                    
                    current_start = timestamp
                    current_start_frame = frame
                    current_end = timestamp
                    current_end_frame = frame
                    expected_next_frame = frame + 1
        
        if current_start is not None:
            interval = TimeInterval(
                video_name=current_video,
                start_time=current_start,
                end_time=current_end,
                frame_count=current_end_frame - current_start_frame + 1,
                first_frame=current_start_frame,
                last_frame=current_end_frame
            )
            intervals.append(interval)
        
        unique_videos = {interval.video_name for interval in intervals}
        total_frames = sum(interval.frame_count for interval in intervals)
        
        return intervals, len(unique_videos), total_frames