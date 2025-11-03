"""
Command-line interface for TrailMarker portfolio agent.
"""

from pathlib import Path
from typing import Optional
import typer
from rich.console import Console 
from rich.table import Table
from rich.panel import Panel

from trailmarker.portfolio import Portfolio
from trailmarker.graph import create_graph
from trailmarker.state import PortfolioAgentState

app = typer.Typer(
    name="trailmarker",
    help="AI-powered portfolio management with trailing stop-loss tracking", 
    add_completion=False
)

console = Console()

@app.command() 
def main(
    message: str = typer.Argument(..., help="Natural language command to execute"),
    portfolio_name: str = typer.Option(
        "default", "--portfolio", "-p", help="Name of portfolio to use"
    ),
    data_dir: Optional[Path] = typer.Option(
        None, "--data-dir", "-d", help="Directory containing portfolio JSON file"
    ),
):
    """
    Send a natural language command to the portfolio agent.

    Examples:
        trailmarker "show my stocks"
        trailmarker "load portfolio tech stocks"
        trailmarker "add NVDA"
    """
    try:
        workflow = create_graph()

        # Don't load portfolio here - let nodes decide!
        state: PortfolioAgentState = {
            "user_input": message,
            "portfolio": None,           # ← Start with None
            "portfolio_name": portfolio_name,
            "data_dir": data_dir,
            "intent": None,
            "status": "",                # ← NEW: Will be set by nodes
            "response": "",
        }

        result = workflow.invoke(state)
        
        # Display response
        console.print(Panel(result["response"], title="TrailMarker", border_style="green"))
        
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(code=1)
    
if __name__ == "__main__":
    app()