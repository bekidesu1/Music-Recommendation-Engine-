"""FastAPI application for serving song recommendations."""

from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .pipeline import HybridRecommender, load_recommender


class RecommendationResponse(BaseModel):
    user_id: int
    recommendations: list[dict]


def create_app(artifact_dir: str | None = None) -> FastAPI:
    app = FastAPI(title="Music Recommendation API", version="1.0.0")
    model_dir = artifact_dir or os.getenv("ARTIFACT_DIR", "artifacts")

    recommender: HybridRecommender = load_recommender(model_dir)


    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/recommendations", response_model=RecommendationResponse)
    def recommendations(user_id: int, top_k: int = 10):
        try:
            recs = recommender.recommend_for_user(user_id=user_id, top_k=top_k)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        payload = recs[["song_id", "title", "artist", "genre", "hybrid_score"]].to_dict("records")
        return RecommendationResponse(user_id=user_id, recommendations=payload)

    return app


app = create_app()
