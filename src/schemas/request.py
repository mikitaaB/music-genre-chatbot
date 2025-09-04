import io
import logging
from typing import Optional
from fastapi import UploadFile, File, HTTPException, status

from src.config import get_config

logger = logging.getLogger("music_genre_bot.schemas.request")
config = get_config()


class AudioFileValidator:
    @staticmethod
    def validate_file_size(file_size: int) -> None:
        max_size = config.settings.max_file_size
        if file_size > max_size:
            size_mb = max_size / (1024 * 1024)
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File too large. Maximum size: {size_mb:.1f}MB"
            )

    @staticmethod
    def validate_content_type(content_type: Optional[str]) -> None:
        if not content_type:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Unable to determine file type"
            )

        allowed_types = config.settings.allowed_audio_formats
        if content_type not in allowed_types:
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail=f"Unsupported format: {content_type}"
            )

    @staticmethod
    def validate_file_content(file_content: bytes) -> None:
        if not file_content:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Empty file received"
            )


async def audio_file(file: UploadFile = None) -> UploadFile:
    if file is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No file provided"
        )

    try:
        AudioFileValidator.validate_content_type(file.content_type)

        file_content = await file.read()
        file_size = len(file_content)

        AudioFileValidator.validate_file_size(file_size)
        AudioFileValidator.validate_file_content(file_content)

        file.file = io.BytesIO(file_content)
        return file

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"File validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to process uploaded file"
        )
