"""
API routes for the Movie Recommendation System.

Endpoints:
  GET  /api/movies          — list all movie titles (for autocomplete)
  GET  /api/movies/search   — search movies by prefix
  POST /api/recommend/content
  POST /api/recommend/collaborative
  POST /api/recommend/hybrid
  GET  /api/stats           — dashboard stats
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from fastapi import APIRouter, Request, HTTPException, Query
from app.schemas import (
    ContentRequest, CollaborativeRequest, HybridRequest,
    RecommendationResponse, MovieResult, MovieSearchResult, DashboardStats,
)
from src.content_based  import content_recommend
from src.collaborative  import collaborative_recommend
from src.hybrid         import hybrid_recommend

router = APIRouter()


# ── helper ────────────────────────────────────────────────────────────────────

def _df_to_results(df, score_col: str) -> list[MovieResult]:
    if df is None:
        return []
    results = []
    for _, row in df.iterrows():
        score = None
        if score_col in row.index:
            try:
                score = float(row[score_col])
            except (TypeError, ValueError):
                pass
        results.append(MovieResult(
            title     = str(row.get("title", "")) or None,
            genre_str = str(row.get("genre_str", "")) or None,
            score     = score,
        ))
    return results


# ── movie list / search ───────────────────────────────────────────────────────

@router.get("/movies", response_model=MovieSearchResult)
async def list_movies(request: Request):
    return MovieSearchResult(titles=request.app.state.movie_titles[:500])


@router.get("/movies/search", response_model=MovieSearchResult)
async def search_movies(request: Request, q: str = Query("", min_length=1)):
    q_lower = q.lower()
    matches = [t for t in request.app.state.movie_titles if q_lower in t.lower()][:30]
    return MovieSearchResult(titles=matches)


# ── content-based ─────────────────────────────────────────────────────────────

@router.post("/recommend/content", response_model=RecommendationResponse)
async def recommend_content(body: ContentRequest, request: Request):
    s = request.app.state
    df = content_recommend(
        body.title,
        s.tfidf_matrix,
        s.content,
        s.indices,
        n=body.n,
    )
    if df is None:
        raise HTTPException(status_code=404, detail=f"Movie '{body.title}' not found.")
    return RecommendationResponse(
        method  = "content-based",
        query   = body.title,
        results = _df_to_results(df, "similarity_score"),
    )


# ── collaborative ─────────────────────────────────────────────────────────────

@router.post("/recommend/collaborative", response_model=RecommendationResponse)
async def recommend_collaborative(body: CollaborativeRequest, request: Request):
    s = request.app.state
    df = collaborative_recommend(
        body.user_id,
        s.user_factors,
        s.item_factors,
        s.user_pos,
        s.movie_ids_lookup,
        s.user_item_sparse,
        s.global_mean,
        s.links,          # ← real links table from state
        s.content,
        n=body.n,
    )
    if df is None:
        raise HTTPException(status_code=404, detail=f"User {body.user_id} not found in the dataset.")
    return RecommendationResponse(
        method  = "collaborative-svd",
        query   = f"user_{body.user_id}",
        results = _df_to_results(df, "predicted_rating"),
    )


# ── hybrid ────────────────────────────────────────────────────────────────────

@router.post("/recommend/hybrid", response_model=RecommendationResponse)
async def recommend_hybrid(body: HybridRequest, request: Request):
    s = request.app.state
    df = hybrid_recommend(
        body.user_id,
        body.title,
        s.tfidf_matrix,
        s.content,
        s.indices,
        s.user_factors,
        s.item_factors,
        s.user_pos,
        s.movie_ids_lookup,
        s.global_mean,
        s.links,          # ← real links table from state
        n=body.n,
        w_content=body.w_content,
        w_collab=body.w_collab,
    )
    if df is None:
        raise HTTPException(status_code=404, detail=f"Movie '{body.title}' not found.")
    return RecommendationResponse(
        method  = "hybrid",
        query   = f"user_{body.user_id} × {body.title}",
        results = _df_to_results(df, "hybrid_score"),
    )


# ── stats ─────────────────────────────────────────────────────────────────────

@router.get("/stats", response_model=DashboardStats)
async def get_stats(request: Request):
    s = request.app.state
    return DashboardStats(
        total_movies   = len(s.content),
        total_users    = len(s.user_pos),
        svd_components = s.user_factors.shape[1],
        vocab_size     = s.tfidf_matrix.shape[1],
        model_types    = ["Content-Based (TF-IDF)", "Collaborative (SVD)", "Hybrid"],
    )