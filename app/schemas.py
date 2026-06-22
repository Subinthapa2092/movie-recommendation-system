from pydantic import BaseModel, Field
from typing import List, Optional


class ContentRequest(BaseModel):
    title: str = Field(..., example="The Dark Knight")
    n: int = Field(10, ge=1, le=50)


class CollaborativeRequest(BaseModel):
    user_id: int = Field(..., example=1)
    n: int = Field(10, ge=1, le=50)


class HybridRequest(BaseModel):
    user_id: int  = Field(..., example=1)
    title:   str  = Field(..., example="The Dark Knight")
    n:       int  = Field(10, ge=1, le=50)
    w_content: float = Field(0.6, ge=0.0, le=1.0)
    w_collab:  float = Field(0.4, ge=0.0, le=1.0)


class MovieResult(BaseModel):
    title:      Optional[str]
    genre_str:  Optional[str]
    score:      Optional[float]


class RecommendationResponse(BaseModel):
    method:  str
    query:   str
    results: List[MovieResult]


class MovieSearchResult(BaseModel):
    titles: List[str]


class DashboardStats(BaseModel):
    total_movies:     int
    total_users:      int
    svd_components:   int
    vocab_size:       int
    model_types:      List[str]