from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File

from src.schemas.request import audio_file
from src.schemas.response import ClassificationResponse
from src.services.classification_service import get_classification_service

router = APIRouter(
    prefix="/api/v1",
    tags=["Music Genre Classification"]
)

classification_service = get_classification_service()


@router.post(
    "/classify",
    response_model=ClassificationResponse
)
async def classify_music(
    file: UploadFile = File(...)
) -> ClassificationResponse:
    try:
        validated_file = await audio_file(file)
        file_content = await validated_file.read()
        result = classification_service.classify_with_recommendations(
            file_bytes=file_content
        )
        return result

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except RuntimeError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Classification service temporarily unavailable"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred"
        )
