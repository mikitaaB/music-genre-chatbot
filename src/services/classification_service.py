import io
from typing import Dict, Optional, Tuple, Any

import librosa
import torch
import numpy as np
from transformers import pipeline, Pipeline

from src.config import get_config
from src.services.dynamic_recommendation_service import get_dynamic_recommendation_service
from src.schemas.response import AudioMetadata, ClassificationResponse

config = get_config()


class ModelManager:
    def __init__(self):
        self._genre_classifier: Optional[Pipeline] = None
        self._model_config = config.get_audio_model_config()
        self._model_loaded = False
        self._load_model()

    def _load_model(self) -> None:
        try:
            torch.set_num_threads(self._model_config['torch_num_threads'])
            self._genre_classifier = pipeline(
                "audio-classification",
                model=self._model_config['model_name'],
                device_map=self._model_config['device_map'],
                model_kwargs={
                    "cache_dir": self._model_config['cache_dir'],
                    "dtype": torch.float32,
                }
            )
            self._model_loaded = True
        except Exception as e:
            self._model_loaded = False
            raise RuntimeError(f"Model initialization failed: {e}")

    def is_model_loaded(self) -> bool:
        return self._model_loaded and self._genre_classifier is not None

    def get_classifier(self) -> Pipeline:
        if not self.is_model_loaded():
            raise RuntimeError("Classification model is not loaded")
        return self._genre_classifier


class AudioProcessor:
    def __init__(self):
        self._file_config = config.get_file_config()
        self.sample_rate = self._file_config['sample_rate']

    def process_audio_file(self, file_bytes: bytes) -> Tuple[Dict[str, Any], AudioMetadata]:
        if not file_bytes:
            raise ValueError("Empty audio file")

        try:
            buffer = io.BytesIO(file_bytes)
            waveform, _ = librosa.load(buffer, sr=self.sample_rate, mono=True, dtype=np.float32)

            self._validate_audio_quality(waveform)

            metadata = AudioMetadata(
                duration=len(waveform) / self.sample_rate,
                sample_rate=self.sample_rate,
                channels=1,
                file_size=len(file_bytes),
                format="mp3"
            )

            audio_dict = {
                "raw": waveform,
                "sampling_rate": self.sample_rate
            }

            return audio_dict, metadata

        except Exception as e:
            raise ValueError(f"Unable to process audio file: {e}")

    def _validate_audio_quality(self, waveform: np.ndarray) -> None:
        if len(waveform) == 0:
            raise ValueError("Audio file contains no audio data")

        duration = len(waveform) / self.sample_rate
        max_duration = self._file_config['max_duration']
        if duration > max_duration:
            raise ValueError(f"Audio too long: {duration:.1f}s (max: {max_duration}s)")


class ClassificationService:
    def __init__(self):
        self.model_manager = ModelManager()
        self.audio_processor = AudioProcessor()
        self.dynamic_recommendation_service = get_dynamic_recommendation_service()

    def classify_with_recommendations(self, file_bytes: bytes) -> ClassificationResponse:
        try:
            audio_dict, _ = self.audio_processor.process_audio_file(file_bytes)
            genre = self._classify_genre(audio_dict)

            if self.dynamic_recommendation_service.is_available():
                recommendations = self.dynamic_recommendation_service.generate_dynamic_recommendations(genre)
            else:
                recommendations = []

            return ClassificationResponse(
                genre=genre,
                recommendations=recommendations
            )

        except Exception as e:
            raise RuntimeError(f"Classification service error: {e}")

    def _classify_genre(self, audio_dict: Dict[str, Any]) -> str:
        try:
            classifier = self.model_manager.get_classifier()
            results = classifier(audio_dict)

            if not results:
                raise RuntimeError("No classification results")

            primary_result = results[0]
            genre = primary_result["label"].lower().strip()

            return genre

        except Exception as e:
            raise RuntimeError(f"Model inference error: {e}")


_classification_service: Optional[ClassificationService] = None


def get_classification_service() -> ClassificationService:
    global _classification_service
    if _classification_service is None:
        _classification_service = ClassificationService()
    return _classification_service