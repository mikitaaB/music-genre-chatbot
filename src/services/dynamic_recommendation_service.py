import logging
from typing import List, Optional
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

from src.config import get_config, get_logger

logger = get_logger("dynamic_recommendation_service")
config = get_config()


class DynamicRecommendationService:
    def __init__(self):
        self.tokenizer: Optional[T5Tokenizer] = None
        self.model: Optional[T5ForConditionalGeneration] = None
        self._load_text_model()

    def _load_text_model(self) -> None:
        try:
            self.tokenizer = T5Tokenizer.from_pretrained(
                "google/flan-t5-large",
                cache_dir=config.settings.model_cache_dir
            )

            self.model = T5ForConditionalGeneration.from_pretrained(
                "google/flan-t5-large",
                dtype=torch.float32,
                cache_dir=config.settings.model_cache_dir,
                low_cpu_mem_usage=True
            )

        except Exception as e:
            logger.error(f"Failed to load T5 model: {e}")
            self.tokenizer = None
            self.model = None

    def generate_dynamic_recommendations(self, genre: str) -> List[str]:
        if not self.tokenizer or not self.model:
            return []

        try:
            prompt = self._build_prompt(genre)
            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids

            outputs = self.model.generate(
                input_ids,
                max_new_tokens=200,
                temperature=0.7,
                do_sample=True,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return self._process_generated_text(generated_text, prompt)

        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return []

    def _build_prompt(self, genre: str) -> str:
        prompts_by_genre = {
            "rock": "Please suggest 2-3 similar bands or songs in the rock genre in bullet points.",
            "pop": "Please recommend popular artists in the pop genre in bullet points.",
            "hip-hop": "Please suggest popular playlists or artists in the hip-hop genre in bullet points.",
            "classical": "Please share an interesting fact about a composer or a piece in classical music.",
            "jazz": "Please suggest a relaxing evening playlist in the jazz genre in bullet points.",
            "electronic": "Please recommend famous music festivals or top DJs in the electronic music genre in bullet points.",
            "electro": "Please recommend famous music festivals or top DJs in the electronic music genre in bullet points."
        }

        prompt = prompts_by_genre.get(genre.lower(), f"Please recommend music related to {genre} genre.")
        return prompt

    def _process_generated_text(self, generated_text: str, prompt: str) -> List[str]:
        try:
            if prompt in generated_text:
                recommendations_text = generated_text.replace(prompt, "").strip()
            else:
                recommendations_text = generated_text.strip()

            return [recommendations_text] if recommendations_text else []

        except Exception:
            return []

    def is_available(self) -> bool:
        return self.tokenizer is not None and self.model is not None


_dynamic_recommendation_service: Optional[DynamicRecommendationService] = None


def get_dynamic_recommendation_service() -> DynamicRecommendationService:
    global _dynamic_recommendation_service
    if _dynamic_recommendation_service is None:
        _dynamic_recommendation_service = DynamicRecommendationService()
    return _dynamic_recommendation_service
