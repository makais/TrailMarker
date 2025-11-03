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
    - "add", "remove", "query", "load", "unknown"
    """
    user_input = state["user_input"].lower()

    # Keyword matching (simple and reliable)
    if any(word in user_input for word in ["add", "buy", "include", "purchase"]):
        intent = "add"
    elif any(word in user_input for word in ["remove", "delete", "drop", "sell"]):
        intent = "remove"
    elif any(word in user_input for word in ["show", "list", "display", "status", "what"]):
        intent = "query"
    elif any(word in user_input for word in ["load", "open", "fetch", "get", "retrieve"]):
        intent = "load"
    else:
        intent = "unknown"

    return {"intent": intent}

def handle_query(state: PortfolioAgentState) -> Dict[str, Any]:
    """Query needs a portfolio - load or fail gracefully."""
    
    # Reuse if already loaded, otherwise load from disk
    # If None return status not_found and response message
    portfolio = state["portfolio"]
    if portfolio is None:
        if not Portfolio.exists(state["portfolio_name"], state["data_dir"]):
            return {
                "status": "not_found",
                "response": f"Portfolio '{state['portfolio_name']}' doesn't exist yet. Create it first!"
            }
        
        portfolio = Portfolio.load(state["portfolio_name"], state["data_dir"])
    
    # Use the portfolio
    df = portfolio.to_dataframe()
    if df.empty:
        return {
            "portfolio": portfolio,
            "status": "empty",
            "response": f"Portfolio '{portfolio.name}' is empty."
        }
    
    tickers = ', '.join(df['ticker'].tolist())
    return {
        "portfolio": portfolio,
        "status": "success",
        "response": f"Portfolio '{portfolio.name}' has {len(df)} stocks: {tickers}"
    }

def handle_add(state: PortfolioAgentState) -> Dict[str, Any]:
    """Handle adding stocks - create empty portfolio if needed."""

    # Reuse if already loaded
    # If None, create one then add
    portfolio = state["portfolio"]
    if portfolio is None:
        # For 'add', create empty portfolio if it doesn't exist
        if Portfolio.exists(state["portfolio_name"], state["data_dir"]):
            portfolio = Portfolio.load(state["portfolio_name"], state["data_dir"])
        else:
            portfolio = Portfolio.create_empty(state["portfolio_name"], state["data_dir"])
            # Note: create_empty sets dirty=True, so finalize node will auto-save dirty portfolios

    response = "ADD functionality coming soon (would parse ticker from input)"
    return {
    "portfolio": portfolio,
    "status": "success",
    "response": response
    }

def handle_remove(state: PortfolioAgentState) -> Dict[str, Any]:
    """Handle removing stocks (stub for now). Requires existing portfolio."""

    # Reuse if already loaded
    portfolio = state["portfolio"]
    if portfolio is None:
        if not Portfolio.exists(state["portfolio_name"], state["data_dir"]):
            return {
                "status": "not_found",
                "response": f"Portfolio '{state['portfolio_name']}' doesn't exist yet. Nothing to remove!"
            }
        
        portfolio = Portfolio.load(state["portfolio_name"], state["data_dir"])
    
    response = "REMOVE functionality coming soon (would parse ticker from input)"
    return {"portfolio": portfolio, "response": response}

def handle_unknown(state: PortfolioAgentState) -> Dict[str, Any]:
    """Handle unknown intents."""
    return {
        "status": "error",
        "response": "I don't understand that command. Try: 'show portfolio', 'load', etc."
    }

def handle_load(state: PortfolioAgentState) -> Dict[str, Any]:
    """Load a portfolio - needs a portfolio - load or respond with warning and status"""
    
    # Reuse if already loaded, otherwise load from disk
    # If None return status not_found and response message
    portfolio = state["portfolio"]
    if portfolio is None:
        if not Portfolio.exists(state["portfolio_name"], state["data_dir"]):
            return {
                "status": "not_found",
                "response": f"Portfolio '{state['portfolio_name']}' doesn't exist yet. Create it first!"
            }
        
        portfolio = Portfolio.load(state["portfolio_name"], state["data_dir"])
    
    # Use the portfolio
    df = portfolio.to_dataframe()
    return {
        "portfolio": portfolio,
        "status": "success",
        "response": f"Portfolio '{portfolio.name}' loaded with {len(df)} stocks."
    }

def finalize(state: PortfolioAgentState) -> Dict[str, Any]:
    """
    Final node before END - auto-save dirty portfolios.
    
    This ensures each CLI invocation is a complete unit of work:
    load → modify → save.
    """
    portfolio = state["portfolio"]
    
    # If no portfolio loaded, nothing to do
    if portfolio is None:
        return {}
    
    # If portfolio has unsaved changes, save them
    if portfolio.dirty:
        portfolio.save()
    
    return {}

