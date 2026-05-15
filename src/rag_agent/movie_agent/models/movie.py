"""Pydantic models for movie data."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class PersonRecord(BaseModel):
    """Director, writer, or actor record."""

    nconst: str | None = None
    primaryName: str | None = None
    birthYear: int | None = None
    deathYear: int | None = None
    primaryProfession: str | None = None


class RatingRecord(BaseModel):
    """Movie rating record."""

    averageRating: float | None = None
    numVotes: int | None = None


class MovieRecord(BaseModel):
    """Canonical movie record used for vector ingestion and prompt context."""

    movie_id: str = Field(..., description="Primary movie identifier shared by SQL/vector/graph")
    titleType: str | None = None
    primaryTitle: str | None = None
    originalTitle: str | None = None
    isAdult: bool | None = None
    startYear: int | None = None
    endYear: int | None = None
    runtimeMinutes: int | None = None
    genres: str | None = None
    description: str | None = None
    narrative: str | None = None
    directors: list[PersonRecord] = Field(default_factory=list)
    writers: list[PersonRecord] = Field(default_factory=list)
    actors: list[PersonRecord] = Field(default_factory=list)
    rating: RatingRecord | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def vector_content(self) -> str:
        """Return the text payload embedded into pgvector."""
        pieces = [
            self.primaryTitle or self.originalTitle or self.movie_id,
            f"Genres: {self.genres}" if self.genres else "",
            f"Year: {self.startYear}" if self.startYear else "",
            f"Description: {self.description}" if self.description else "",
            f"Narrative: {self.narrative}" if self.narrative else "",
        ]
        return "\n".join(piece for piece in pieces if piece)


class MovieSearchResult(BaseModel):
    """Vector search result."""

    movie_id: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    score: float | None = None
