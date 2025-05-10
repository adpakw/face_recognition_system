import os
from multiprocessing import set_start_method
from pathlib import Path

import torch
from celery import Celery

set_start_method("spawn", force=True)
torch.multiprocessing.set_start_method("spawn", force=True)

IMAGE_STORAGE_PATH = "datasets/new_persons"
Path(IMAGE_STORAGE_PATH).mkdir(exist_ok=True)

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


if __name__ == "__main__":
    app.worker_main(
        argv=["worker", "--pool=processes", "--concurrency=1", "--loglevel=info"]
    )
