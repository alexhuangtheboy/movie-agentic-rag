"""LLM and embedding adapters."""

from .dashscope import DashScopeEmbeddings, get_chat_model, get_embeddings

__all__ = ["DashScopeEmbeddings", "get_chat_model", "get_embeddings"]
