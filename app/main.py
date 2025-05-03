from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from app.clients.db_manager import DatabaseManager
from dotenv import load_dotenv
import os
from pydantic import BaseModel
from typing import Optional
import uuid
import uvicorn

import os
import time
from app.broker import process_video_task
@asynccontextmanager
async def lifespan(app: FastAPI):
    dbm =DatabaseManager()
    dbm.execute_sql_files_from_folder()
    
    yield
    pass

app = FastAPI(lifespan=lifespan)

class TaskResponse(BaseModel):
    task_id: str
    status: str
    message: str

@app.post("/process-video/", response_model=TaskResponse)
async def process_video():
    # try:
        # Запускаем Celery задачу
    print("tgvb ")
    task = process_video_task.delay("cuts/parsa.mp4")
    print("jm")
    
    return JSONResponse(
        status_code=202,
        content={
            "task_id": task.id,
            "status": "PENDING",
            "message": "Video processing started"
        }
    )
    # except Exception as e:
    #     print("dfghj")
    #     raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    load_dotenv(".env")

    uvicorn.run(app, host=os.getenv('API_HOST'), port=int(os.getenv('API_PORT', 8000)))