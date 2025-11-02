"""
LangGraph node functions for the TrailMarker portfolio agent.
"""

from typing import Dict, Any
from trailmarker.state import PortfolioAgentState
from trailmarker.portfolio import Portfolio


def classify_intent(state: PortfolioAgentState) -> Dict[str, Any]:
    """
    Classify the user's intent using keyword matching.

    Returns dict with 'intent' key set to one of:
    - "add", "remove", "query", "save", "unknown"
    """
    user_input = state["user_input"].lower()

    # Keyword matching (simple and reliable)
    if any(word in user_input for word in ["add", "buy", "include", "purchase"]):
        intent = "add"
    elif any(word in user_input for word in ["remove", "delete", "drop", "sell"]):
        intent = "remove"
    elif any(word in user_input for word in ["show", "list", "display", "status", "what"]):
        intent = "query"
    elif "save" in user_input or "persist" in user_input:
        intent = "save"
    else:
        intent = "unknown"

    return {"intent": intent}


def handle_query(state: PortfolioAgentState) -> Dict[str, Any]:
    """Handle portfolio query requests."""
    portfolio = state["portfolio"]

    # Use the to_dataframe() method
    df = portfolio.to_dataframe()

    if df.empty:
        response = f"Portfolio '{portfolio.name}' is empty."
    else:
        tickers = ', '.join(df['ticker'].tolist())
        response = f"Portfolio '{portfolio.name}' has {len(df)} stocks: {tickers}"

    return {"response": response}


def handle_add(state: PortfolioAgentState) -> Dict[str, Any]:
    """Handle adding stocks (stub for now)."""
    response = "ADD functionality coming soon (would parse ticker from input)"
    return {"response": response}


def handle_remove(state: PortfolioAgentState) -> Dict[str, Any]:
    """Handle removing stocks (stub for now)."""
    response = "REMOVE functionality coming soon (would parse ticker from input)"
    return {"response": response}


def handle_save(state: PortfolioAgentState) -> Dict[str, Any]:
    """Handle saving portfolio."""
    portfolio = state["portfolio"]

    if not portfolio.dirty:
        response = f"Portfolio '{portfolio.name}' has no unsaved changes."
    else:
        portfolio.save()
        response = f"Portfolio '{portfolio.name}' saved successfully!"

    return {"response": response}


def handle_unknown(state: PortfolioAgentState) -> Dict[str, Any]:
    """Handle unknown intents."""
    response = "I don't understand that command. Try: 'show portfolio', 'save', etc."
    return {"response": response}