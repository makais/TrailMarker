"""
TrailMarker: AI-powered portfolio management with LangGraph

Trailing stop-loss tracking via natural language CLI.
"""

from trailmarker.portfolio import Portfolio
from trailmarker.state import PortfolioAgentState
from trailmarker.graph import create_graph

__version__ = "0.1.0"
__all__ = ["Portfolio", "PortfolioAgentState", "create_graph"]