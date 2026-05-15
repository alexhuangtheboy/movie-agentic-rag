"""Compact TOON-style formatting for movie context."""

from __future__ import annotations

import json
from typing import Any


def _escape(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, (list, tuple)):
        return ";".join(_escape(item) for item in value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    text = str(value)
    if "," in text or "\n" in text or '"' in text:
        return '"' + text.replace('"', '\\"') + '"'
    return text


def convert_to_toon_format(movie: dict[str, Any]) -> str:
    """Convert one movie dictionary to a compact TOON block."""
    fields = [
        "movie_id",
        "primaryTitle",
        "startYear",
        "genres",
        "averageRating",
        "numVotes",
        "directors",
        "writers",
        "actors",
        "description",
    ]
    present = [field for field in fields if movie.get(field) not in (None, "", [])]
    header = f"movie[1]{{{','.join(present)}}}:"
    row = "  " + ",".join(_escape(movie.get(field)) for field in present)
    return "\n".join([header, row])


def convert_movies_to_toon(movies: list[dict[str, Any]]) -> str:
    """Convert multiple movie dictionaries to a compact TOON table."""
    if not movies:
        return "movies[0]{}:"
    fields = [
        "movie_id",
        "primaryTitle",
        "startYear",
        "genres",
        "averageRating",
        "numVotes",
        "directors",
        "actors",
    ]
    header = f"movies[{len(movies)}]{{{','.join(fields)}}}:"
    rows = ["  " + ",".join(_escape(movie.get(field)) for field in fields) for movie in movies]
    return "\n".join([header] + rows)
