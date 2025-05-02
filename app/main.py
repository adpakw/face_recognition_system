from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.clients.db_manager import DatabaseManager
from celery import Celery

@asynccontextmanager
async def lifespan(app: FastAPI):
    dbm =DatabaseManager()
    dbm.execute_sql_files_from_folder()
    
    yield
    pass

app = FastAPI(lifespan=lifespan)



appc = Celery(
    "tasks",
    broker="amqp://rabbitmq:5672",
    backend="redis://redis:6379/0"
)
