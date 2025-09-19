# Music Genre Classification Bot

## Project Description

A chatbot for recognizing the style of musical compositions. It accepts MP3 audio files, classifies them by genre, and provides personalized recommendations.


## Instructions for Checking Results

### 1. Installing Dependencies

```
python -m venv venv # For Windows
python3 -m venv venv  # For Linux/MacOS

.\venv\Scripts\activate # Windows
source venv/bin/activate # Linux/MacOS

pip install -r requirements.txt
```

### 2. Running the Server

**Local run:**
```
uvicorn src.main:app --host 0.0.0.0 --port 8000
```

**Production:**

Copy the configuration example:
```
cp .env.example .env
```
Edit the .env file for production.

Starting the server:
```
python -m src.main
```

### 3. API Testing

The server will start at http://localhost:8000

**Interactive documentation:** http://localhost:8000/docs

**Example request via curl:**
```
curl -X POST "http://localhost:8000/api/v1/classify" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@test_song.mp3"
```

**Manual testing of samples:**
```
curl -X POST "http://localhost:8000/api/v1/classify" \
     -F "file=@rock_sample.mp3"
```

**Example response:**
```json
{
  "genre": "rock",
  "recommendations": [
    "Led Zeppelin, Queen, and The Beatles"
  ]
}
```

## Technical Details

### Models Used

1. **Genre Classification:** `dima806/music_genres_classification` (HuggingFace)

2. **Recommendation Generation:** `google/flan-t5-large` (HuggingFace)

### Supported Genres and Recommendation Types

- **Rock:** Similar bands and compositions
- **Pop:** Popular genre artists
- **Hip-Hop:** Playlists and artists
- **Classical:** Interesting facts about composers
- **Jazz:** Relaxing evening playlists
- **Electronic:** Festivals and top DJs

### File Requirements

- **Format:** MP3
- **Maximum size:** 50MB
- **Maximum duration:** 5 minutes

### Architecture

- **Framework:** FastAPI
- **Audio Processing:** librosa
- **ML Models:** HuggingFace Transformers
- **Configuration:** Environment variables with defaults