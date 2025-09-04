import os
import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field


class Settings(BaseModel):
    app_name: str = Field(default="Music Genre Bot")
    app_version: str = Field(default="1.0.0")
    debug: bool = Field(default=False)
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    api_reload: bool = Field(default=False)
    audio_model_name: str = Field(default="dima806/music_genres_classification")
    text_model_name: str = Field(default="google/flan-t5-large")
    model_cache_dir: str = Field(default="./model_cache")
    audio_sample_rate: int = Field(default=16000)
    max_audio_duration: int = Field(default=300)
    max_file_size: int = Field(default=50 * 1024 * 1024)
    allowed_audio_formats: list = Field(default=["audio/mpeg", "audio/mp3", "audio/wav"])
    log_level: str = Field(default="INFO")
    log_format: str = Field(default="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    log_file: Optional[str] = Field(default=None)
    torch_num_threads: int = Field(default=4)
    device_map: str = Field(default="auto")

    class Config:
        case_sensitive = False

    @classmethod
    def from_env(cls):
        import os

        kwargs = {}
        for field_name in cls.model_fields.keys():
            env_name = field_name.upper()
            if env_name in os.environ:
                env_value = os.environ[env_name]
                if field_name in ['debug', 'api_reload']:
                    kwargs[field_name] = env_value.lower() in ('true', '1', 'yes')
                elif field_name in ['api_port', 'audio_sample_rate', 'max_audio_duration', 'max_file_size', 'torch_num_threads']:
                    kwargs[field_name] = int(env_value)
                elif field_name == 'allowed_audio_formats':
                    kwargs[field_name] = env_value.split(',')
                else:
                    kwargs[field_name] = env_value
        return cls(**kwargs)


class LoggingConfig:
    @staticmethod
    def setup_logging(settings: Settings) -> None:
        if settings.log_file:
            log_path = Path(settings.log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)

        logging.basicConfig(
            level=getattr(logging, settings.log_level.upper()),
            format=settings.log_format,
            handlers=LoggingConfig._get_handlers(settings)
        )

        LoggingConfig._configure_third_party_loggers()

    @staticmethod
    def _get_handlers(settings: Settings) -> list:
        handlers = []

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, settings.log_level.upper()))
        console_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        console_handler.setFormatter(console_formatter)
        handlers.append(console_handler)

        if settings.log_file:
            file_handler = logging.FileHandler(settings.log_file)
            file_handler.setLevel(logging.DEBUG)
            file_formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
            )
            file_handler.setFormatter(file_formatter)
            handlers.append(file_handler)

        return handlers

    @staticmethod
    def _configure_third_party_loggers() -> None:
        logging.getLogger("transformers").setLevel(logging.WARNING)
        logging.getLogger("transformers.configuration_utils").setLevel(logging.ERROR)
        logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
        logging.getLogger("torch").setLevel(logging.WARNING)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


class AppConfig:
    def __init__(self):
        self.settings = Settings.from_env()
        self._setup_environment()
        LoggingConfig.setup_logging(self.settings)

    def _setup_environment(self) -> None:
        Path(self.settings.model_cache_dir).mkdir(parents=True, exist_ok=True)

    def get_audio_model_config(self) -> Dict[str, Any]:
        return {
            "model_name": self.settings.audio_model_name,
            "cache_dir": self.settings.model_cache_dir,
            "device_map": self.settings.device_map,
            "torch_num_threads": self.settings.torch_num_threads,
            "trust_remote_code": False,
            "local_files_only": False,
            "resume_download": True,
            "force_download": False,
            "proxies": None,
            "use_auth_token": False,
            "revision": "main",
            "timeout": 60,
            "max_retries": 3
        }

    def get_text_model_config(self) -> Dict[str, Any]:
        return {
            "model_name": self.settings.text_model_name,
            "cache_dir": self.settings.model_cache_dir,
            "device_map": "cpu",
            "torch_num_threads": self.settings.torch_num_threads,
            "trust_remote_code": False,
            "local_files_only": False,
            "resume_download": True,
            "force_download": False,
            "proxies": None,
            "use_auth_token": False,
            "revision": "main",
            "timeout": 60,
            "max_retries": 3
        }

    def get_model_config(self) -> Dict[str, Any]:
        return self.get_audio_model_config()

    def get_api_config(self) -> Dict[str, Any]:
        return {
            "host": self.settings.api_host,
            "port": self.settings.api_port,
            "reload": self.settings.api_reload,
            "debug": self.settings.debug
        }

    def get_file_config(self) -> Dict[str, Any]:
        return {
            "max_file_size": self.settings.max_file_size,
            "allowed_formats": self.settings.allowed_audio_formats,
            "max_duration": self.settings.max_audio_duration,
            "sample_rate": self.settings.audio_sample_rate
        }


config = AppConfig()


def get_config() -> AppConfig:
    return config


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"music_genre_bot.{name}")