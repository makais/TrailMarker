# TrailMarker

AI-powered portfolio management with LangGraph - trailing stop-loss tracking via natural language CLI.

## Features

- 🤖 **Natural Language Interface** - Manage portfolios using conversational commands
- 📊 **Trailing Stop-Loss** - Automatic tracking of stop-loss triggers
- 🔄 **LangGraph Workflow** - Stateful agent with conditional routing
- 💾 **Persistent Storage** - JSON-based portfolio data
- 🎯 **Modern Python** - Type hints, pandas DataFrames, pathlib

## Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/TrailMarker.git
cd TrailMarker

# Install with uv (recommended)
uv pip install -e .

# Or with pip
pip install -e .
```

## Quick Start

```bash
# Query portfolio
trailmarker query "show my stocks"

# Add stock (coming soon)
trailmarker add "NVDA with 15% stop"

# Save changes
trailmarker save
```

## Development

```bash
# Install with dev dependencies
uv pip install -e ".[dev]"

# Run tests
pytest

# Format code
black src/

# Lint
ruff src/
```

## Project Structure

```
TrailMarker/
├── src/trailmarker/      # Main package
│   ├── portfolio.py      # Portfolio management
│   ├── state.py          # LangGraph state definition
│   ├── nodes.py          # Graph node functions
│   ├── graph.py          # Workflow construction
│   └── cli.py            # Command-line interface
├── tests/                # Unit tests
├── data/portfolios/      # Portfolio JSON files
└── pyproject.toml        # Package configuration
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Learning Journey

This project demonstrates:
- LangGraph state management and routing
- Modern Python packaging (src layout)
- CLI development with Typer
- Financial data handling with pandas/yfinance

Built as a learning project for mastering LangGraph workflows.