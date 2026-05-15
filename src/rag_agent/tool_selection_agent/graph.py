"""Tool selection LangGraph."""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from rag_agent.tool_selection_agent.nodes import initialize_tool_selection_node, select_tool_node
from rag_agent.tool_selection_agent.states import ToolSelectionState

builder = StateGraph(ToolSelectionState)
builder.add_node("initialize_tool_selection", initialize_tool_selection_node)
builder.add_node("select_tool", select_tool_node)
builder.add_edge(START, "initialize_tool_selection")
builder.add_edge("initialize_tool_selection", "select_tool")
builder.add_edge("select_tool", END)

tool_selection_graph = builder.compile(name="Movie Tool Selection")

__all__ = ["tool_selection_graph"]
