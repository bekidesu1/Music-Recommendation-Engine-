"""Data loading and preprocessing utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class DatasetBundle:
    """Container for project datasets."""

    users: pd.DataFrame
    songs: pd.DataFrame
    listen_history: pd.DataFrame


REQUIRED_USER_COLUMNS = {"user_id"}
REQUIRED_SONG_COLUMNS = {"song_id", "title", "artist", "genre"}
REQUIRED_HISTORY_COLUMNS = {"user_id", "song_id", "play_count"}


def _validate_columns(df: pd.DataFrame, required: set[str], name: str) -> None:
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{name} is missing required columns: {sorted(missing)}")


def load_dataset(data_dir: str | Path) -> DatasetBundle:
    """Load users, songs, and listening history CSV files from data directory."""
    data_dir = Path(data_dir)
    users = pd.read_csv(data_dir / "users.csv")
    songs = pd.read_csv(data_dir / "songs.csv")
    listen_history = pd.read_csv(data_dir / "listen_history.csv")

    _validate_columns(users, REQUIRED_USER_COLUMNS, "users.csv")
    _validate_columns(songs, REQUIRED_SONG_COLUMNS, "songs.csv")
    _validate_columns(listen_history, REQUIRED_HISTORY_COLUMNS, "listen_history.csv")

    return DatasetBundle(users=users, songs=songs, listen_history=listen_history)


def preprocess_listen_history(bundle: DatasetBundle) -> pd.DataFrame:
    """Prepare implicit feedback ratings from play counts and merge song metadata."""
    history = bundle.listen_history.copy()
    history["play_count"] = history["play_count"].clip(lower=0)
    history["rating"] = 1.0 + history["play_count"].apply(lambda x: min(4.0, x / 10.0))

    merged = history.merge(bundle.songs, on="song_id", how="left")
    if merged[["title", "artist", "genre"]].isnull().any().any():
        raise ValueError("Found songs in listen history without metadata in songs.csv")

    return merged
