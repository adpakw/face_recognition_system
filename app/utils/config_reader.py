from pathlib import Path
from typing import Dict, Literal, Optional

import yaml
from pydantic import BaseModel, Field 


# Модели для валидации конфига
class PeopleDetectorConfig(BaseModel):
    device: Literal["cuda", "cpu"]
    confidence_threshold: float

class FaceDetectorConfig(BaseModel):
    model_path: Path
    score_threshold: float
    nms_threshold: float
    top_k: int
    device: Literal["cuda", "cpu"]
    confidence_threshold: float

class FaceRecognizerConfig(BaseModel):
    model: Literal["ArcFace", "VGG-Face", "SFace"]
    device: Literal["cuda", "cpu"]
    confidence_threshold: float

class GeneralConfig(BaseModel):
    output_dir: Path
    json_size: int

class PipelineConfig(BaseModel):
    people_detector: PeopleDetectorConfig
    face_detector: FaceDetectorConfig
    face_recognizer: FaceRecognizerConfig

class AppConfig(BaseModel):
    pipline: PipelineConfig
    general: GeneralConfig

class ConfigReader:
    def __init__(self, config_path: Path = Path("app/configs/pipline_conf.yaml")):
        """
        Инициализация читателя конфигурации
        
        Args:
            config_path: Путь к YAML конфигурационному файлу
        """
        self.config_path = config_path
        self._config = self._load_and_validate()

    def _load_and_validate(self) -> AppConfig:
        """Загружает и валидирует конфигурацию"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with open(self.config_path, 'r') as f:
            raw_config = yaml.safe_load(f)
        
        # Конвертируем строковые пути в Path объекты
        if 'pipline' in raw_config and 'face_detector' in raw_config['pipline']:
            raw_config['pipline']['face_detector']['model_path'] = Path(
                raw_config['pipline']['face_detector']['model_path']
            )
        
        if 'general' in raw_config and 'output_dir' in raw_config['general']:
            raw_config['general']['output_dir'] = Path(
                raw_config['general']['output_dir']
            )
        
        return AppConfig(**raw_config)

    def get_people_detector_config(self) -> PeopleDetectorConfig:
        """Возвращает конфигурацию детектора людей"""
        return self._config.pipline.people_detector

    def get_face_detector_config(self) -> FaceDetectorConfig:
        """Возвращает конфигурацию детектора лиц"""
        return self._config.pipline.face_detector

    def get_face_recognizer_config(self) -> FaceRecognizerConfig:
        """Возвращает конфигурацию детектора лиц"""
        return self._config.pipline.face_recognizer

    def get_general_config(self) -> GeneralConfig:
        """Возвращает общую конфигурацию"""
        return self._config.general

    def get_config(self) -> AppConfig:
        """Возвращает всю валидированную конфигурацию"""
        return self._config