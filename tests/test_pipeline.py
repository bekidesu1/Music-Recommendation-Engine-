import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("numpy")
pytest.importorskip("sklearn")
pytest.importorskip("torch")

from music_recommender.data import load_dataset, preprocess_listen_history
from music_recommender.pipeline import train_pipeline


def test_data_loading_and_preprocessing():
    bundle = load_dataset("data")
    interactions = preprocess_listen_history(bundle)

    assert not interactions.empty
    assert {"user_id", "song_id", "rating", "title", "artist", "genre"}.issubset(interactions.columns)


def test_train_and_recommend():
    recommender = train_pipeline(data_dir="data", embedding_dim=8, epochs=3)
    recs = recommender.recommend_for_user(user_id=1, top_k=3)

    assert len(recs) == 3
    assert "hybrid_score" in recs.columns
    assert recs["song_id"].nunique() == 3
