"""
State definition for the TrailMarker portfolio agent.
"""

from typing import TypedDict, Optional
from trailmarker.portfolio import Portfolio


class PortfolioAgentState(TypedDict):
    """State that flows through the LangGraph workflow."""
    user_input: str                    # What the user typed
    portfolio: Portfolio               # The portfolio object we're working with
    intent: Optional[str]              # Classified intent: "add", "remove", "query", "save"
    response: str                      # Final response to user