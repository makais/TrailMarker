"""
LangGraph workflow construction for the TrailMarker portfolio agent.
"""

from langgraph.graph import StateGraph, END
from trailmarker.state import PortfolioAgentState
from trailmarker import nodes


def route_by_intent(state: PortfolioAgentState) -> str:
    """
    Router function that decides which node to go to based on intent.

    Returns the name of the next node.
    """
    intent = state["intent"]

    if intent == "query":
        return "handle_query"
    elif intent == "add":
        return "handle_add"
    elif intent == "remove":
        return "handle_remove"
    elif intent == "load":
        return "handle_load"
    else:
        return "handle_unknown"


def create_graph():
    """
    Create and compile the LangGraph workflow.

    Returns:
        Compiled graph ready for invocation
    """
    # Create graph
    graph = StateGraph(PortfolioAgentState)

    # Define all nodes
    handlers = {
        "classify_intent": nodes.classify_intent,
        "handle_query": nodes.handle_query,
        "handle_add": nodes.handle_add,
        "handle_remove": nodes.handle_remove,
        "handle_unknown": nodes.handle_unknown,
        "handle_load": nodes.handle_load,
        "finalize": nodes.finalize,  # cleanup node
    }

    # Add all nodes
    for name, func in handlers.items():
        graph.add_node(name, func)

    # Start at classification
    graph.set_entry_point("classify_intent")

    # Route after classification
    graph.add_conditional_edges(
        "classify_intent",
        route_by_intent,
        {
            "handle_query": "handle_query",
            "handle_add": "handle_add",
            "handle_remove": "handle_remove",
            "handle_load": "handle_load",
            "handle_unknown": "handle_unknown"
        }
    )

    # All handlers go to finalize
    for name in handlers:
        if name not in ("classify_intent", "finalize"):
            graph.add_edge(name, "finalize")

    # Finalize goes to END
    graph.add_edge("finalize", END)

    # Compile and return
    return graph.compile()