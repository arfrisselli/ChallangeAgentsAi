"""
LangGraph state: messages, plan, tool results, and context for multi-step flow.
"""
from typing import Annotated, Optional, Sequence, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class GraphState(TypedDict):
    """State passed between nodes. messages is the conversation; plan and tool_results used by Planner/Executor."""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    plan: Optional[str]
    tool_results: Optional[str]
    need_web_fallback: Optional[bool]
    is_conversational: Optional[bool]
    is_weather_query: Optional[bool]
    context: Optional[str]
