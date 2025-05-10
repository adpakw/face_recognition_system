import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile,Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.broker import check_videos_task, prepare_dataset_task
from app.clients.db_manager import DatabaseManager

IMAGE_STORAGE_PATH = "datasets/new_persons"
Path(IMAGE_STORAGE_PATH).mkdir(exist_ok=True)

VIDEO_STORAGE_PATH = "videos"
Path(VIDEO_STORAGE_PATH).mkdir(exist_ok=True)
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


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


@asynccontextmanager
async def lifespan(app: FastAPI):
    dbm = DatabaseManager()
    dbm.execute_sql_files_from_folder()

    prepare_dataset_task.delay()

    # thread = threading.Thread(target=check_database_periodically, daemon=True)
    # thread.start()
    yield
    pass


app = FastAPI(lifespan=lifespan)


def check_database_periodically():
    while True:
        print("Проверяю БД...")
        check_videos_task.delay()
        time.sleep(1 * 60 * 60)


@app.post("/api/upload-photo/")
async def upload_photo(name: str = Form(...), photo: UploadFile = File(...)):
    try:
        existing_files = list(Path(IMAGE_STORAGE_PATH).glob("*.*"))
        if not existing_files:
            max_num = 0
        max_num = -1
        for file in existing_files:
            try:
                num = int(file.stem)
                if num > max_num:
                    max_num = num
            except ValueError:
                continue

        file_ext = Path(photo.filename).suffix
        filename = f"{max_num + 1}{file_ext}"
        Path(f"{IMAGE_STORAGE_PATH}/{name}").mkdir(exist_ok=True)
        file_path = f"{IMAGE_STORAGE_PATH}/{name}/{filename}"

        with open(file_path, "wb") as buffer:
            buffer.write(await photo.read())

        prepare_dataset_task.delay()

        return JSONResponse(
            status_code=200,
            content={
                "status": "SUCCESS",
                "message": "Photo uploaded successfully",
                "name": name,
            },
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/upload-video/")
async def upload_video(
    video: UploadFile = File(...),
):
    try:
        if not Path(video.filename).suffix.lower() in ALLOWED_VIDEO_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid video format. Allowed formats: {', '.join(ALLOWED_VIDEO_EXTENSIONS)}",
            )

        file_path = os.path.join(VIDEO_STORAGE_PATH, video.filename)

        with open(file_path, "wb") as buffer:
            buffer.write(await video.read())

        response_data = {
            "status": "SUCCESS",
            "message": "Video uploaded successfully",
            "file_size": os.path.getsize(file_path),
        }

        return JSONResponse(status_code=200, content=response_data)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Video upload failed: {str(e)}")


@app.get("/api/people/")
async def get_people():
    try:
        service = DatabaseManager()
        result = service.get_all_people()

        if isinstance(result, pd.DataFrame):
            if not result.empty:
                return result.to_dict("records")
            return []

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка при получении списка людей известных системе: {str(e)}",
        )


@app.get("/api/videos/")
async def get_videos():
    try:
        service = DatabaseManager()
        result = service.get_all_videos()

        if isinstance(result, pd.DataFrame):
            if not result.empty:
                return result.to_dict("records")
            return []

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка при получении списка обработанных видео: {str(e)}",
        )


@app.get("/api/top-faces/", response_model=List[Dict[str, Any]])
async def get_top_faces():
    try:
        service = DatabaseManager()
        result = service.get_top_10()

        if isinstance(result, pd.DataFrame):
            if not result.empty:
                return result.to_dict("records")
            return []

        return [{"count": count, "person_name": name} for count, name in result]

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Ошибка при получении топ-10 лиц: {str(e)}"
        )


@app.get("/api/person-intervals/", response_model=PersonIntervalsResponse)
async def get_person_intervals(person_name: str = Query(..., description="Имя человека для поиска")
):
    try:
        service = DatabaseManager()
        intervals, total_videos, total_frames = service.find_continuous_intervals(
            person_name
        )
        intervals_dicts = [
            {
                "video_name": interval.video_name,
                "start_time": interval.start_time,
                "end_time": interval.end_time,
                "frame_count": interval.frame_count,
                "first_frame": interval.first_frame,
                "last_frame": interval.last_frame
            }
            for interval in intervals
        ]
        
        return PersonIntervalsResponse(
            person_name=person_name,
            intervals=intervals_dicts,
            total_intervals=len(intervals),
            total_videos=total_videos,
            total_frames=total_frames
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# @app.get("/api/videos-nas/", response_model=List[Dict[str, Union[str, int, float]]])
# async def list_videos_in_folder(folder_path: str = ""):
#     """
#     Возвращает список видеофайлов в указанной папке на NAS
    
#     Параметры:
#     - folder_path: относительный путь к папке внутри NAS
    
#     Возвращает:
#     - Список словарей с информацией о видеофайлах:
#       [{
#         "name": "video1.mp4",
#         "path": "/mnt/smb_share/path/video1.mp4",
#         "size": 1024567,  # размер в байтах (число)
#         "modified": 1708594981.4404233  # timestamp (число с плавающей точкой)
#       }, ...]
#     """
#     try:
#         nas_root = "/mnt/smb_share/[02] Проекты/[05] Тестовые видео"
        
#         # Проверяем, что NAS доступен
#         if not os.path.exists(nas_root):
#             raise HTTPException(
#                 status_code=503,
#                 detail="NAS не доступен. Проверьте подключение."
#             )
        
#         # Формируем полный путь
#         full_path = os.path.join(nas_root, folder_path)
        
#         # Проверяем существование папки
#         if not os.path.exists(full_path):
#             raise HTTPException(
#                 status_code=404,
#                 detail=f"Папка {full_path} не существует"
#             )
        
#         if not os.path.isdir(full_path):
#             raise HTTPException(
#                 status_code=400,
#                 detail=f"{full_path} не является папкой"
#             )
        
#         # Собираем информацию о видеофайлах
#         video_files = []
#         for filename in os.listdir(full_path):
#             file_ext = os.path.splitext(filename)[1]
#             if file_ext in ALLOWED_VIDEO_EXTENSIONS:
#                 file_path = os.path.join(full_path, filename)
#                 try:
#                     stat = os.stat(file_path)
                    
#                     video_files.append({
#                         "name": filename,
#                         "path": file_path,
#                         "size": stat.st_size,  # размер в байтах
#                         "modified": stat.st_mtime  # timestamp
#                     })
#                 except Exception as e:
#                     print(f"Ошибка при обработке файла {filename}: {str(e)}")
#                     continue
        
#         return video_files
    
#     except PermissionError as e:
#         raise HTTPException(
#             status_code=403,
#             detail=f"Нет доступа к папке: {str(e)}"
#         )
#     except Exception as e:
#         raise HTTPException(
#             status_code=500,
#             detail=f"Ошибка при получении списка видео: {str(e)}"
#         )

if __name__ == "__main__":
    load_dotenv(".env")

    uvicorn.run(app, host=os.getenv("API_HOST"), port=int(os.getenv("API_PORT", 8000)))
