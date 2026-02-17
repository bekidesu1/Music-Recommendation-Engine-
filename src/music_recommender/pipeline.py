"""Training pipeline and hybrid recommendation orchestration."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pickle
import numpy as np
import pandas as pd
import torch

from .data import DatasetBundle, load_dataset, preprocess_listen_history
from .models import ContentSimilarityModel, MatrixFactorizationModel, train_matrix_factorization


@dataclass
class RecommenderArtifacts:
    model: MatrixFactorizationModel
    content_model: ContentSimilarityModel
    users: pd.DataFrame
    songs: pd.DataFrame
    interactions: pd.DataFrame
    user_to_index: dict[int, int]
    song_to_index: dict[int, int]
    index_to_song: dict[int, int]


class HybridRecommender:
    """Hybrid recommender blending collaborative and content scores."""

    def __init__(self, artifacts: RecommenderArtifacts, alpha: float = 0.7):
        self.artifacts = artifacts
        self.alpha = alpha

    def _normalize_scores(self, arr: np.ndarray) -> np.ndarray:
        min_val, max_val = arr.min(), arr.max()
        if max_val - min_val < 1e-8:
            return np.zeros_like(arr)
        return (arr - min_val) / (max_val - min_val)

    def recommend_for_user(self, user_id: int, top_k: int = 10) -> pd.DataFrame:
        if user_id not in self.artifacts.user_to_index:
            raise ValueError(f"Unknown user_id: {user_id}")

        user_idx = self.artifacts.user_to_index[user_id]
        num_songs = len(self.artifacts.song_to_index)

        song_indices = torch.arange(num_songs, dtype=torch.long)
        user_indices = torch.full((num_songs,), user_idx, dtype=torch.long)

        self.artifacts.model.eval()
        with torch.no_grad():
            collab_scores = self.artifacts.model(user_indices, song_indices).numpy()

        listened_song_ids = (
            self.artifacts.interactions.loc[self.artifacts.interactions["user_id"] == user_id, "song_id"]
            .drop_duplicates()
            .tolist()
        )
        content_scores = self.artifacts.content_model.score_songs_for_user(listened_song_ids)

        collab_norm = self._normalize_scores(collab_scores)
        content_norm = self._normalize_scores(content_scores)
        hybrid_scores = self.alpha * collab_norm + (1 - self.alpha) * content_norm

        listened = set(listened_song_ids)
        song_ids = [self.artifacts.index_to_song[i] for i in range(num_songs)]
        result = pd.DataFrame(
            {
                "song_id": song_ids,
                "hybrid_score": hybrid_scores,
                "collaborative_score": collab_norm,
                "content_score": content_norm,
            }
        )
        result = result[~result["song_id"].isin(listened)]
        result = result.sort_values("hybrid_score", ascending=False).head(top_k)

        return result.merge(self.artifacts.songs, on="song_id", how="left")


def train_pipeline(
    data_dir: str | Path,
    embedding_dim: int = 32,
    epochs: int = 20,
    alpha: float = 0.7,
) -> HybridRecommender:
    """Load data, engineer features, train embedding model, and build hybrid recommender."""
    bundle: DatasetBundle = load_dataset(data_dir)
    interactions = preprocess_listen_history(bundle)

    user_ids = sorted(interactions["user_id"].unique().tolist())
    song_ids = sorted(bundle.songs["song_id"].unique().tolist())

    user_to_index = {int(uid): idx for idx, uid in enumerate(user_ids)}
    song_to_index = {int(sid): idx for idx, sid in enumerate(song_ids)}
    index_to_song = {idx: sid for sid, idx in song_to_index.items()}

    mf_model = train_matrix_factorization(
        interactions=interactions,
        user_to_index=user_to_index,
        song_to_index=song_to_index,
        embedding_dim=embedding_dim,
        epochs=epochs,
    )
    content_model = ContentSimilarityModel.fit(bundle.songs)

    artifacts = RecommenderArtifacts(
        model=mf_model,
        content_model=content_model,
        users=bundle.users,
        songs=bundle.songs,
        interactions=interactions,
        user_to_index=user_to_index,
        song_to_index=song_to_index,
        index_to_song=index_to_song,
    )

    return HybridRecommender(artifacts=artifacts, alpha=alpha)


def save_recommender(recommender: HybridRecommender, artifact_dir: str | Path) -> None:
    """Persist trained artifacts for API serving."""
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    torch.save(recommender.artifacts.model.state_dict(), artifact_dir / "mf_model.pt")
    with (artifact_dir / "content_model.pkl").open("wb") as f:
        pickle.dump(recommender.artifacts.content_model, f)
    recommender.artifacts.users.to_csv(artifact_dir / "users.csv", index=False)
    recommender.artifacts.songs.to_csv(artifact_dir / "songs.csv", index=False)
    recommender.artifacts.interactions.to_csv(artifact_dir / "interactions.csv", index=False)

    metadata = {
        "user_to_index": recommender.artifacts.user_to_index,
        "song_to_index": recommender.artifacts.song_to_index,
        "alpha": recommender.alpha,
        "embedding_dim": recommender.artifacts.model.user_embeddings.embedding_dim,
    }
    (artifact_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")


def load_recommender(artifact_dir: str | Path) -> HybridRecommender:
    """Load persisted recommender artifacts."""
    artifact_dir = Path(artifact_dir)
    metadata = json.loads((artifact_dir / "metadata.json").read_text(encoding="utf-8"))

    user_to_index = {int(k): int(v) for k, v in metadata["user_to_index"].items()}
    song_to_index = {int(k): int(v) for k, v in metadata["song_to_index"].items()}
    index_to_song = {idx: sid for sid, idx in song_to_index.items()}

    users = pd.read_csv(artifact_dir / "users.csv")
    songs = pd.read_csv(artifact_dir / "songs.csv")
    interactions = pd.read_csv(artifact_dir / "interactions.csv")
    with (artifact_dir / "content_model.pkl").open("rb") as f:
        content_model = pickle.load(f)

    mf_model = MatrixFactorizationModel(
        num_users=len(user_to_index),
        num_songs=len(song_to_index),
        embedding_dim=int(metadata["embedding_dim"]),
    )
    mf_model.load_state_dict(torch.load(artifact_dir / "mf_model.pt", map_location="cpu"))

    artifacts = RecommenderArtifacts(
        model=mf_model,
        content_model=content_model,
        users=users,
        songs=songs,
        interactions=interactions,
        user_to_index=user_to_index,
        song_to_index=song_to_index,
        index_to_song=index_to_song,
    )
    return HybridRecommender(artifacts=artifacts, alpha=float(metadata["alpha"]))
