# Music Genre Classification Bot

## Описание проекта

Чат-бот для распознавания стиля музыкальных композиций. Принимает MP3 аудиофайлы, классифицирует их по жанру и выдает персонализированные рекомендации.

## Состав архива

```
music-genre-bot/src/main.py - основной файл приложения
music-genre-bot/src/config.py - конфигурация
music-genre-bot/src/api/v1/endpoints.py - API эндпоинты
music-genre-bot/src/services/classification_service.py - сервис классификации жанров
music-genre-bot/src/services/dynamic_recommendation_service.py - сервис рекомендаций
music-genre-bot/src/schemas/request.py - модели запросов
music-genre-bot/src/schemas/response.py - модели ответов
music-genre-bot/requirements.txt - Python зависимости
music-genre-bot/.env.example - пример конфигурации для прода
```

## Инструкции к проверке результата

### 1. Установка зависимостей

```
cd music-genre-bot
pip install -r requirements.txt
```

### 2. Запуск сервера

**Локальный запуск:**
```
uvicorn src.main:app --host 0.0.0.0 --port 8000
```

**Прод:**

Скопируйте пример конфигурации:
```
cp .env.example .env
```
Отредактировать .env файл для прода.

Запуск сервера:
```
python -m src.main
```

### 3. Тестирование API

Сервер запустится на http://localhost:8000

**Интерактивная документация:** http://localhost:8000/docs

**Пример запроса через curl:**
```
curl -X POST "http://localhost:8000/api/v1/classify" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@test_song.mp3"
```

**Ручное тестирование образцов:**
```
curl -X POST "http://localhost:8000/api/v1/classify" \
     -F "file=@rock_sample.mp3"
```

**Пример ответа:**
```json
{
  "genre": "rock",
  "recommendations": [
    "Led Zeppelin, Queen, and The Beatles"
  ]
}
```

## Технические детали

### Используемые модели

1. **Классификация жанров:** `dima806/music_genres_classification` (HuggingFace)

2. **Генерация рекомендаций:** `google/flan-t5-large` (HuggingFace)

### Поддерживаемые жанры и типы рекомендаций

- **Rock:** Похожие группы и композиции
- **Pop:** Популярные исполнители жанра
- **Hip-Hop:** Плейлисты и артисты
- **Classical:** Интересные факты о композиторах
- **Jazz:** Расслабляющие плейлисты для вечера
- **Electronic:** Фестивали и топовые диджеи

### Требования к файлам

- **Формат:** MP3
- **Максимальный размер:** 50MB
- **Максимальная длительность:** 5 минут

### Архитектура

- **Framework:** FastAPI
- **Audio Processing:** librosa
- **ML Models:** HuggingFace Transformers
- **Configuration:** Environment variables with defaults
