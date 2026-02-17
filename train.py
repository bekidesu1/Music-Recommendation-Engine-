"""Train and save the hybrid recommender."""

from music_recommender.pipeline import save_recommender, train_pipeline

if __name__ == "__main__":
    recommender = train_pipeline(data_dir="data", embedding_dim=16, epochs=25, alpha=0.7)
    save_recommender(recommender, artifact_dir="artifacts")
    print("Training complete. Artifacts saved in ./artifacts")
