"""Embedding and similarity models for hybrid recommendation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from torch import nn
from torch.utils.data import DataLoader, Dataset


class InteractionDataset(Dataset):
    """PyTorch dataset for user-song interactions."""

    def __init__(self, user_indices: np.ndarray, song_indices: np.ndarray, ratings: np.ndarray):
        self.user_indices = torch.tensor(user_indices, dtype=torch.long)
        self.song_indices = torch.tensor(song_indices, dtype=torch.long)
        self.ratings = torch.tensor(ratings, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.ratings)

    def __getitem__(self, idx: int):
        return self.user_indices[idx], self.song_indices[idx], self.ratings[idx]


class MatrixFactorizationModel(nn.Module):
    """Simple latent-factor model for collaborative filtering."""

    def __init__(self, num_users: int, num_songs: int, embedding_dim: int = 32):
        super().__init__()
        self.user_embeddings = nn.Embedding(num_users, embedding_dim)
        self.song_embeddings = nn.Embedding(num_songs, embedding_dim)
        self.user_bias = nn.Embedding(num_users, 1)
        self.song_bias = nn.Embedding(num_songs, 1)

        nn.init.normal_(self.user_embeddings.weight, std=0.01)
        nn.init.normal_(self.song_embeddings.weight, std=0.01)

    def forward(self, user_idx: torch.Tensor, song_idx: torch.Tensor) -> torch.Tensor:
        user_vec = self.user_embeddings(user_idx)
        song_vec = self.song_embeddings(song_idx)
        interaction = (user_vec * song_vec).sum(dim=1)
        bias = self.user_bias(user_idx).squeeze(-1) + self.song_bias(song_idx).squeeze(-1)
        return interaction + bias


@dataclass
class ContentSimilarityModel:
    """TF-IDF content-based model for songs."""

    vectorizer: TfidfVectorizer
    tfidf_matrix: np.ndarray
    song_id_to_index: dict[int, int]

    @classmethod
    def fit(cls, songs: pd.DataFrame) -> "ContentSimilarityModel":
        features = (songs["title"] + " " + songs["artist"] + " " + songs["genre"]).tolist()
        vectorizer = TfidfVectorizer(stop_words="english")
        tfidf_matrix = vectorizer.fit_transform(features)
        mapping = {int(song_id): idx for idx, song_id in enumerate(songs["song_id"].tolist())}
        return cls(vectorizer=vectorizer, tfidf_matrix=tfidf_matrix, song_id_to_index=mapping)

    def user_profile_vector(self, listened_song_ids: list[int]) -> np.ndarray:
        valid_indices = [self.song_id_to_index[sid] for sid in listened_song_ids if sid in self.song_id_to_index]
        if not valid_indices:
            return np.asarray(self.tfidf_matrix.mean(axis=0))
        user_matrix = self.tfidf_matrix[valid_indices]
        return np.asarray(user_matrix.mean(axis=0))

    def score_songs_for_user(self, listened_song_ids: list[int]) -> np.ndarray:
        profile = self.user_profile_vector(listened_song_ids)
        return cosine_similarity(profile, self.tfidf_matrix).ravel()


def train_matrix_factorization(
    interactions: pd.DataFrame,
    user_to_index: dict[int, int],
    song_to_index: dict[int, int],
    embedding_dim: int = 32,
    epochs: int = 20,
    batch_size: int = 256,
    lr: float = 1e-2,
) -> MatrixFactorizationModel:
    """Train collaborative filtering model on implicit-feedback ratings."""
    user_indices = interactions["user_id"].map(user_to_index).to_numpy()
    song_indices = interactions["song_id"].map(song_to_index).to_numpy()
    ratings = interactions["rating"].to_numpy(dtype=np.float32)

    dataset = InteractionDataset(user_indices, song_indices, ratings)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = MatrixFactorizationModel(
        num_users=len(user_to_index),
        num_songs=len(song_to_index),
        embedding_dim=embedding_dim,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    criterion = nn.MSELoss()

    model.train()
    for _ in range(epochs):
        for batch_user, batch_song, batch_rating in loader:
            optimizer.zero_grad()
            preds = model(batch_user, batch_song)
            loss = criterion(preds, batch_rating)
            loss.backward()
            optimizer.step()

    return model
