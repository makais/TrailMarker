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
    elif intent == "save":
        return "handle_save"
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
        "handle_save": nodes.handle_save,
        "handle_unknown": nodes.handle_unknown,
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
            "handle_save": "handle_save",
            "handle_unknown": "handle_unknown"
        }
    )

    # All handlers go to END
    for name in handlers:
        if name != "classify_intent":
            graph.add_edge(name, END)

    # Compile and return
    return graph.compile()