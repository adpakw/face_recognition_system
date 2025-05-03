import os
from multiprocessing import set_start_method
import torch
from celery import Celery

set_start_method('spawn', force=True)
torch.multiprocessing.set_start_method('spawn', force=True)


app = Celery(
    'tasks_broker',
    broker='amqp://localhost:5672',
    backend='redis://localhost:6379/0',
    include=['app.broker']
)

app.conf.update(
    worker_concurrency=1,
    task_acks_late=True,
    worker_prefetch_multiplier=1
)

@app.task(
    name='app.broker.process_video_task',
    # bind=True,
    # autoretry_for=(RuntimeError,),
    # max_retries=3
)
def process_video_task(video_path: str):
    try:
        from app.utils.config_reader import ConfigReader
        from app.pipeline.pipeline import AutomaticIdentificationPipeline

        baseline_cfg = ConfigReader("app/configs/pipeline_conf.yaml")

        pipeline = AutomaticIdentificationPipeline(baseline_cfg)
        pipeline.process_video(video_path, False, True, True, True)

    except RuntimeError as e:
        raise

if __name__ == '__main__':
    app.worker_main(
        argv=['worker', '--pool=processes', '--concurrency=1', '--loglevel=info']
    )