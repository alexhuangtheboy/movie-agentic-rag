"""DashScope OpenAI-compatible chat and embedding adapters."""

from __future__ import annotations

import os
from typing import List

from langchain_core.embeddings import Embeddings
from langchain_openai import ChatOpenAI
from openai import OpenAI

from rag_agent.constants import (
    get_embedding_api_key,
    get_embedding_endpoint,
    get_embedding_model,
    get_llm_api_key,
    get_llm_endpoint,
    get_llm_model,
)


class DashScopeEmbeddings(Embeddings):
    """Minimal LangChain embedding adapter for DashScope compatible-mode APIs."""

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
    ) -> None:
        self.base_url = base_url or get_embedding_endpoint()
        self.api_key = api_key or get_embedding_api_key()
        self.model = model or get_embedding_model()
        self.timeout = timeout or float(os.getenv("EMBEDDING_TIMEOUT_SECONDS", "30"))
        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key, timeout=self.timeout)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of documents."""
        if not texts:
            return []
        response = self.client.embeddings.create(model=self.model, input=texts)
        return [item.embedding for item in response.data]

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        response = self.client.embeddings.create(model=self.model, input=[text])
        return response.data[0].embedding


def get_chat_model(model_name: str | None = None) -> ChatOpenAI:
    """Create a deterministic DashScope OpenAI-compatible chat model."""
    return ChatOpenAI(
        model=model_name or get_llm_model(),
        base_url=get_llm_endpoint(),
        api_key=get_llm_api_key(),
        temperature=0,
        timeout=float(os.getenv("LLM_TIMEOUT_SECONDS", "60")),
    )


def get_embeddings() -> DashScopeEmbeddings:
    """Create the configured DashScope embedding adapter."""
    return DashScopeEmbeddings()
