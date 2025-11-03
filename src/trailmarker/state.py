"""
State definition for the TrailMarker portfolio agent.
"""

from pathlib import Path
from typing import TypedDict, Optional
from trailmarker.portfolio import Portfolio


class PortfolioAgentState(TypedDict):
    """State that flows through the LangGraph workflow."""
    user_input: str                    # What the user typed
    portfolio: Optional[Portfolio]     # "What portfolio IS loaded?", or None if not or doesn't exist
    portfolio_name: str                # Serves as chat session context, "What portfolio SHOULD I work with?" CLI uses 'default' if not specified
    data_dir: Optional[Path]           # If not in default 
    intent: Optional[str]              # Classified intent: "add", "remove", "query", "load"
    status: str                        # Status: "success", "error", "warning", "not_found", "empty", "no_changes"
    response: str                      # Final response to user