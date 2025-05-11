import os
from multiprocessing import set_start_method
from pathlib import Path
import shutil
import torch
from celery import Celery

set_start_method("spawn", force=True)
torch.multiprocessing.set_start_method("spawn", force=True)

IMAGE_STORAGE_PATH = "datasets/new_persons"
Path(IMAGE_STORAGE_PATH).mkdir(exist_ok=True)
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

app = Celery(
    "tasks_broker",
    broker="amqp://localhost:5672",
    backend="redis://localhost:6379/0",
    include=["app.broker"],
)

app.conf.update(worker_concurrency=1, task_acks_late=True, worker_prefetch_multiplier=1)


@app.task(name="app.broker.prepare_dataset")
def prepare_dataset_task():
    try:
        from app.clients.image_dataset import ImageDataset
        from app.utils.config_reader import ConfigReader

        config = ConfigReader("app/configs/pipeline_conf.yaml")
        image_dataset = ImageDataset(config)
        image_dataset.add_persons()

    except RuntimeError as e:
        raise


@app.task(name="app.broker.check_videos")
def check_videos_task():
    try:
        from app.clients.db_manager import DatabaseManager
        from app.pipeline.pipeline import AutomaticIdentificationPipeline
        from app.utils.config_reader import ConfigReader

        config = ConfigReader("app/configs/pipeline_conf.yaml")
        pipeline = AutomaticIdentificationPipeline(config)

        db_manager = DatabaseManager()
        new_videos = db_manager.get_new_videos()
        for video in new_videos:
            pipeline.process_video(f"videos/{video}", False, True, True, True)

    except RuntimeError as e:
        raise

@app.task(name="app.broker.check_videos_in_nas")
def check_videos_in_nas_task():
    nas_root = "/mnt/smb_share/[02] Проекты/[05] Тестовые видео/для пети"
    local_root = "videos"
    
    if not os.path.exists(nas_root):
        raise RuntimeError("NAS не доступен. Проверьте подключение."
        )
    
    if not os.path.exists(nas_root):
        raise RuntimeError(f"Папка {nas_root} не существует"
        )
    
    if not os.path.isdir(nas_root):
        raise RuntimeError(f"{nas_root} не является папкой"
        )
    
    video_files = []
    for filename in os.listdir(nas_root):
        file_ext = os.path.splitext(filename)[1]
        if file_ext in ALLOWED_VIDEO_EXTENSIONS:
            file_path = os.path.join(nas_root, filename)
            video_files.append(filename)
    
    local_videos = os.listdir(local_root)

    videos_to_download = [x for x in video_files if x not in local_videos]
    print(videos_to_download)
    print()

    for filename in videos_to_download:
        src = os.path.join(nas_root, filename)
        dst = os.path.join(local_root, filename)
        print(filename)
        
        shutil.copy2(src, dst)

if __name__ == "__main__":
    app.worker_main(
        argv=["worker", "--pool=processes", "--concurrency=1", "--loglevel=info"]
    )
