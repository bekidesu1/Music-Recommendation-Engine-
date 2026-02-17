"""Simple offline usage example."""

from music_recommender.pipeline import train_pipeline

if __name__ == "__main__":
    recommender = train_pipeline(data_dir="data", embedding_dim=8, epochs=10)
    recommendations = recommender.recommend_for_user(user_id=1, top_k=5)
    print(recommendations[["song_id", "title", "artist", "hybrid_score"]])
