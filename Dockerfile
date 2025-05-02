FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

# Установка зависимостей
RUN apt-get update && apt-get install -y wget && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh

ENV PATH="/opt/conda/bin:$PATH"

# Копируем Conda env
# COPY ENV.yml .
# RUN conda env create -n ENVNAME --file ENV.yml
COPY conda_environment.yml .
RUN conda env create -f conda_environment.yml python=3.11

# Активируем окружение
ENV PATH /opt/conda/envs/video_gpu/bin:$PATH

# Копируем код и миграции
WORKDIR /app
COPY ./app .
COPY ./migrations ./migrations


CMD ["uvicorn", "main:app", "--host", "${API_HOST}", "--port", "${API_PORT}"]