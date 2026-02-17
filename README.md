# Music Recommendation Engine

A modular Python project that builds a **hybrid music recommendation system** using:
- **Collaborative filtering** (PyTorch matrix factorization embeddings)
- **Content-based similarity** (TF-IDF + cosine similarity)
- **FastAPI** for serving recommendations

## Project Structure

```text
.
├── data/
│   ├── users.csv
│   ├── songs.csv
│   └── listen_history.csv
├── src/music_recommender/
│   ├── api.py
│   ├── data.py
│   ├── models.py
│   └── pipeline.py
├── tests/
│   └── test_pipeline.py
├── example_usage.py
├── train.py
└── requirements.txt
```

## Features

1. Loads and preprocesses users, songs, and listening history datasets.
2. Trains user/song embeddings with a PyTorch matrix factorization model.
3. Builds a content-based model using song metadata (`title`, `artist`, `genre`).
4. Combines both signals into a hybrid recommendation score.
5. Exposes a FastAPI endpoint for real-time recommendations.
6. Includes unit tests and offline example usage.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Dataset Format

### `users.csv`
Required columns:
- `user_id` (int)

Optional columns can include user metadata, e.g. `name`, `age`, `country`.

### `songs.csv`
Required columns:
- `song_id` (int)
- `title` (str)
- `artist` (str)
- `genre` (str)

### `listen_history.csv`
Required columns:
- `user_id` (int)
- `song_id` (int)
- `play_count` (numeric)

`play_count` is converted to an implicit rating used to train the embedding model.

## Train the Recommender

```bash
PYTHONPATH=src python train.py
```

This writes trained artifacts to `./artifacts`.

## API Server

Run the recommendation API:

```bash
PYTHONPATH=src uvicorn music_recommender.api:app --reload --port 8000
```

Example request:

```bash
curl "http://127.0.0.1:8000/recommendations?user_id=1&top_k=5"
```

Health check:

```bash
curl "http://127.0.0.1:8000/health"
```

## Example Offline Usage

```bash
PYTHONPATH=src python example_usage.py
```

## Tests

```bash
PYTHONPATH=src pytest -q
```

## Notes

- For production, you can swap matrix factorization with deeper neural recommenders.
- Add richer song metadata (tempo, language, mood, tags) to improve content signals.
- Integrate user profile features for stronger cold-start performance.
