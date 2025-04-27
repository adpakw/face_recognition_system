from pathlib import Path
from typing import Dict, Literal, Optional, Union, Any
from enum import Enum
import yaml
from pydantic import BaseModel


class ModelConfig(BaseModel):
    device: Literal["cuda", "cpu"]
    confidence_threshold: float


class YuNetConfig(ModelConfig):
    model_path: Path
    score_threshold: float
    nms_threshold: float
    top_k: int
    face_padding: float


class PipelineStep(BaseModel):
    model: str


class GeneralConfig(BaseModel):
    output_dir: Path
    json_size: int

    backup_dir: Path
    add_persons_dir: Path
    faces_save_dir: Path
    features_dir: Path
    gpu_id: int


class AppConfig(BaseModel):
    pipeline: Dict[str, PipelineStep]
    general: GeneralConfig
    models: Dict[str, Any]


class ConfigReader:
    def __init__(self, config_path: str = "app/configs/baseline_pipeline_conf.yaml"):
        self.config_path = Path(config_path)
        self._config = self._load_and_validate()

    def _load_and_validate(self) -> AppConfig:
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with open(self.config_path, 'r') as f:
            raw_config = yaml.safe_load(f)
        
        if 'models' in raw_config:
            for model_name, model_config in raw_config['models'].items():
                if 'model_path' in model_config:
                    raw_config['models'][model_name]['model_path'] = Path(
                        model_config['model_path']
                    )
        
        if 'general' in raw_config and 'output_dir' in raw_config['general']:
            raw_config['general']['output_dir'] = Path(
                raw_config['general']['output_dir']
            )
        
        return AppConfig(**raw_config)

    def get_pipeline_step_config(self, step_name: str):
        """Возвращает конфиг для конкретного этапа пайплайна"""
        step = self._config.pipeline[step_name]
        model_config = self._config.models[step.model].copy()
        
        return {"name": step.model, "cfg": model_config}

    def get_general_config(self) -> GeneralConfig:
        return self._config.general

    def get_config(self) -> AppConfig:
        return self._config