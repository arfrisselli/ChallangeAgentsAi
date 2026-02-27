"""
LangGraph StateGraph: nodes, edges, and routing for conversation, weather, fallback, and execution.
Compiled graph is the main entry for the backend.
"""
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from graph.state import GraphState
from graph.nodes import (
    user_node,
    planner_node,
    conversation_node,
    weather_node,
    fallback_search_node,
    executor_node,
    memory_node,
)


def route_after_planner(state: GraphState) -> str:
    """
    Conditional edge: route based on query type.
    - conversational: go to conversation_node
    - weather: go to weather_node (NO LLM!)
    - web fallback: go to fallback_search_node
    - default: go to executor_node
    """
    if state.get("is_conversational"):
        return "conversation"
    if state.get("is_weather_query"):
        return "weather"
    if state.get("need_web_fallback"):
        return "fallback_search"
    return "executor"


def build_graph():
    """Build and compile the graph. User -> Planner -> (Conversation | Weather | FallbackSearch | Executor) -> Memory -> END."""
    builder = StateGraph(GraphState)

    builder.add_node("user", user_node)
    builder.add_node("planner", planner_node)
    builder.add_node("conversation", conversation_node)
    builder.add_node("weather", weather_node)
    builder.add_node("fallback_search", fallback_search_node)
    builder.add_node("executor", executor_node)
    builder.add_node("memory", memory_node)

    builder.set_entry_point("user")
    builder.add_edge("user", "planner")
    builder.add_conditional_edges(
        "planner",
        route_after_planner,
        {
            "conversation": "conversation",
            "weather": "weather",
            "fallback_search": "fallback_search",
            "executor": "executor"
        }
    )
    builder.add_edge("conversation", "memory")
    builder.add_edge("weather", "memory")
    builder.add_edge("fallback_search", "memory")
    builder.add_edge("executor", "memory")
    builder.add_edge("memory", END)

    memory = MemorySaver()
    return builder.compile(checkpointer=memory)


# Singleton compiled graph for the app
_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph
