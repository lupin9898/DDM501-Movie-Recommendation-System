"""Pydantic v2 request/response schemas for the recommendation API."""

from pydantic import BaseModel, Field


class RecommendRequest(BaseModel):
    """Request body for single-user recommendations."""

    user_id: int
    top_k: int = Field(default=10, ge=1, le=500)
    exclude_seen: bool = True


class MovieRecommendation(BaseModel):
    """A single movie recommendation with metadata."""

    movie_id: int
    title: str
    score: float
    genres: list[str]


class RecommendResponse(BaseModel):
    """Response containing recommendations for one user."""

    user_id: int
    recommendations: list[MovieRecommendation]
    model_version: str
    latency_ms: float


class BatchRecommendRequest(BaseModel):
    """Request body for batch recommendations across multiple users."""

    user_ids: list[int] = Field(..., min_length=1, max_length=100)
    top_k: int = Field(default=10, ge=1, le=500)


class BatchRecommendResponse(BaseModel):
    """Response containing recommendations for multiple users."""

    results: list[RecommendResponse]
    latency_ms: float


class SimilarMoviesResponse(BaseModel):
    """Response containing movies similar to a given movie."""

    movie_id: int
    similar: list[MovieRecommendation]


class HealthResponse(BaseModel):
    """Health-check response with service status."""

    status: str
    model_version: str
    uptime: str
    model_type: str | None = None
    trained_at: str | None = None
    mlflow_run_id: str | None = None
    git_sha: str | None = None
