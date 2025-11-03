"""
Modern Portfolio Manager with Trailing Stops

A clean, Pythonic implementation for managing stock portfolios with trailing stop-loss
strategies. Uses modern Python 3.10+ features, yfinance, and pandas DataFrames.

Copyright (c) 2013, 2024 Tom McDonald, Makai Smith
Licence: The MIT License (MIT)
"""

from __future__ import annotations
import json
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import warnings

import pandas as pd
import yfinance as yf


class Portfolio:
    """
    Manages a portfolio of stocks with trailing stop-loss tracking.

    Uses lazy persistence: modifications are made in-memory and must be
    explicitly saved using the save() method.

    Usage:
        # Create new portfolio
        p = Portfolio.create('my_portfolio', form_data)
        p.save()

        # Load and modify existing
        p = Portfolio.load('my_portfolio')
        p.add_stocks(stock_data)
        p.update_portfolio()  # Calculate trailing stops
        p.save()
    """

    def __init__(
        self,
        name: str,
        portfolio_data: Dict[str, Any],
        json_dir: Optional[Path] = None
    ):
        """
        Private constructor - use Portfolio.load() or Portfolio.create() instead.

        Args:
            name: Name of this portfolio
            portfolio_data: Dictionary containing portfolio configuration and stocks
            json_dir: Directory for JSON storage (defaults to ./json/)
        """
        self.name = name
        self.json_dir = Path(json_dir) if json_dir else Path.cwd() / 'json'
        self.json_dir.mkdir(parents=True, exist_ok=True)

        self.portfolio: Dict[str, Dict[str, Any]] = portfolio_data.copy()
        self.email: Dict[str, Any] = self.portfolio.pop('email', {})
        self._dirty: bool = False

    @classmethod
    def exists(cls, name: str, json_dir: Optional[Path] = None) -> bool:
        """
        Check if a portfolio exists and is valid JSON.

        Args:
            name: Name of the portfolio (without .json extension)
            json_dir: Optional directory containing JSON files

        Returns:
            True if portfolio file exists and contains valid JSON, False otherwise

        Example:
            >>> if Portfolio.exists('my_portfolio'):
            ...     p = Portfolio.load('my_portfolio')
            ... else:
            ...     print("Portfolio not found")
        """
        json_path = Path(json_dir) if json_dir else Path.cwd() / 'json'
        filename = json_path / f'{name}.json'

        # Check if file exists
        if not filename.exists():
            return False

        # Check if file is valid JSON
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                json.load(f)
            return True
        except (json.JSONDecodeError, IOError):
            return False

    @classmethod
    def load(cls, name: str, json_dir: Optional[Path] = None) -> Portfolio:
        """
        Load an existing portfolio from JSON file.

        Args:
            name: Name of the portfolio (without .json extension)
            json_dir: Optional directory containing JSON files

        Returns:
            Portfolio instance

        Raises:
            FileNotFoundError: If portfolio JSON doesn't exist
            json.JSONDecodeError: If JSON is malformed
        """
        json_path = Path(json_dir) if json_dir else Path.cwd() / 'json'
        filename = json_path / f'{name}.json'

        with open(filename, 'r', encoding='utf-8') as f:
            portfolio_data = json.load(f)

        return cls(name, portfolio_data, json_dir)

    @classmethod
    def list_portfolios(
        cls,
        json_dir: Optional[Path] = None,
        as_dataframe: bool = False
    ) -> Dict[str, Dict[str, Any]] | pd.DataFrame:
        """
        List all portfolios in the JSON directory.

        Args:
            json_dir: Optional directory containing JSON files (defaults to ./json/)
            as_dataframe: If True, return as DataFrame instead of dict (default False)

        Returns:
            Dict mapping portfolio names to summary info:
                {name: {'stock_count': int, 'tickers': tuple, 'email_address': str, ...}}
            OR DataFrame with same info if as_dataframe=True

        Example:
            # Dict format (default)
            >>> Portfolio.list_portfolios()
            {
                'my_portfolio': {
                    'stock_count': 3,
                    'tickers': ('AAPL', 'MSFT', 'GOOGL'),
                    'email_address': 'user@example.com',
                    'email_frequency': 3
                },
                'tech_stocks': {...}
            }

            # DataFrame format
            >>> Portfolio.list_portfolios(as_dataframe=True)
               name          stock_count  tickers              email_address
            0  my_portfolio  3            (AAPL, MSFT, GOOGL) user@example.com
        """
        json_path = Path(json_dir) if json_dir else Path.cwd() / 'json'

        if not json_path.exists():
            return {} if not as_dataframe else pd.DataFrame()

        # Find all .json files
        portfolio_files = sorted(json_path.glob('*.json'))

        portfolios = {}
        for pf in portfolio_files:
            name = pf.stem
            try:
                p = cls.load(name, json_dir)
                tickers = tuple(k for k in p.portfolio.keys())

                portfolios[name] = {
                    'stock_count': len(tickers),
                    'tickers': tickers,
                    'email_address': p.email.get('address', ''),
                    'email_frequency': p.email.get('frequency', None)
                }
            except Exception as e:
                warnings.warn(f"Could not load portfolio {name}: {e}")
                portfolios[name] = {
                    'stock_count': None,
                    'tickers': None,
                    'email_address': None,
                    'email_frequency': None,
                    'error': str(e)
                }

        if as_dataframe:
            # Convert dict to DataFrame with name as a column
            rows = [{'name': k, **v} for k, v in portfolios.items()]
            return pd.DataFrame(rows)

        return portfolios

    @classmethod
    def load_or_create(
        cls,
        name: str,
        form_data: Dict[str, Any],
        json_dir: Optional[Path] = None,
        max_rows: int = 5,
        merge_stocks: bool = True
    ) -> Portfolio:
        """
        Load existing portfolio or create new one if it doesn't exist.

        This is the recommended method for typical workflows where you want to
        either load an existing portfolio or create a new one with the same name.

        Args:
            name: Name of the portfolio
            form_data: Dictionary with keys like 'ticker1', 'purchase_date1', etc.
            json_dir: Optional directory for JSON storage
            max_rows: Maximum number of stock rows to process (default 5)
            merge_stocks: If True and portfolio exists, add new stocks to it.
                         If False, warn and return existing portfolio unchanged.

        Returns:
            Portfolio instance (existing or newly created)

        Example:
            >>> # First time - creates new portfolio
            >>> p = Portfolio.load_or_create('tech_stocks', form_data)
            >>> p.save()
            >>>
            >>> # Second time - loads existing and adds new stocks
            >>> p = Portfolio.load_or_create('tech_stocks', more_stocks)
            >>> p.save()
        """
        if cls.exists(name, json_dir):
            # Portfolio already exists
            portfolio = cls.load(name, json_dir)

            if merge_stocks:
                # Add new stocks from form_data to existing portfolio
                portfolio.add_stocks(form_data, max_rows)
            else:
                warnings.warn(
                    f"Portfolio '{name}' already exists. "
                    f"Set merge_stocks=True to add new stocks, or use load() directly."
                )

            return portfolio
        else:
            # Create new portfolio
            return cls.create(name, form_data, json_dir, max_rows)

    @classmethod
    def create_empty(
        cls,
        name: str,
        json_dir: Optional[Path] = None,
        overwrite: bool = False
    ) -> Portfolio:
        """
        Create an empty portfolio (no stocks).

        Useful for agent workflows where stocks will 
        be added from commands rather than form data.

        Args:
            name: Name for this portfolio
            json_dir: Optional directory for JSON storage
            overwrite: If True, allow overwriting existing portfolio.
                       If False (default), raise error if portfolio exists.
        
        Returns:
            New empty Portfolio instance (not yet saved to disk)

        Raises:
            FileExistsError: If portfolio exists and overwrite=False
        
        Example:
            >>> p = Portfolio.create_empty("tech_stocks")
            >>> p.add_stocks(...) # Add via agent workflow presumably
            >>> p.save()

        """
        # Check if portfolio already exists
        if cls.exists(name, json_dir) and not overwrite:
            raise FileExistsError(
                f"Portfolio '{name}' already exists. "
                f"Use overwrite=True to replace it, or use load() instead."
            )

        # Create empty portfolio data
        empty_data = {"email": {"address": "", "frequency": None}}
        portfolio = cls(name, empty_data, json_dir)
        portfolio._dirty = True  # ← ADD THIS LINE
        return portfolio
        
    @classmethod
    def create(
        cls,
        name: str,
        form_data: Dict[str, Any],
        json_dir: Optional[Path] = None,
        max_rows: int = 5,
        overwrite: bool = False
    ) -> Portfolio:
        """
        Create a new portfolio from form data.

        Args:
            name: Name for this portfolio
            form_data: Dictionary with keys like 'ticker1', 'purchase_date1', etc.
            json_dir: Optional directory for JSON storage
            max_rows: Maximum number of stock rows to process (default 5)
            overwrite: If True, allow overwriting existing portfolio.
                      If False (default), raise error if portfolio exists.

        Returns:
            New Portfolio instance (not yet saved to disk)

        Raises:
            FileExistsError: If portfolio with this name already exists and overwrite=False

        Note:
            Consider using load_or_create() for typical workflows instead.
        """
        # Check if portfolio already exists
        if cls.exists(name, json_dir) and not overwrite:
            raise FileExistsError(
                f"Portfolio '{name}' already exists. "
                f"Use overwrite=True to replace it, or use load_or_create() to merge stocks."
            )

        portfolio_data: Dict[str, Any] = {}
        existing_tickers = set()

        # Process each row of stock data
        for n in range(1, max_rows + 1):
            ticker = form_data.get(f'ticker{n}', '').strip().upper()

            if ticker and ticker not in existing_tickers:
                row_ticker, portfolio_row = cls._scrub_row_static(form_data, n)
                if row_ticker:
                    portfolio_data[row_ticker] = portfolio_row
                    existing_tickers.add(row_ticker)

        # Extract email configuration
        portfolio_data['email'] = {
            'address': form_data.get('email', ''),
            'frequency': form_data.get('frequency', 3)
        }

        return cls(name, portfolio_data, json_dir)

    def save(self) -> None:
        """
        Persist portfolio to JSON file.

        Writes the current in-memory state to disk and clears the dirty flag.
        """
        # Combine portfolio and email back into single dict
        save_data = self.portfolio.copy()
        save_data['email'] = self.email

        filename = self.json_dir / f'{self.name}.json'

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)

        self._dirty = False

    def reload(self) -> None:
        """
        Discard in-memory changes and reload from disk.

        Useful for abandoning uncommitted modifications.
        """
        reloaded = Portfolio.load(self.name, self.json_dir)
        self.portfolio = reloaded.portfolio
        self.email = reloaded.email
        self._dirty = False

    def add_stocks(
        self,
        form_data: Dict[str, Any],
        max_rows: int = 3,
        skip_duplicates: bool = True,
        overwrite_duplicates: bool = False
    ) -> Dict[str, List[str]]:
        """
        Add stocks to the portfolio from form data.

        Args:
            form_data: Dictionary with keys like 'ticker1', 'purchase_date1', etc.
            max_rows: Maximum number of rows to process (default 3)
            skip_duplicates: If True, silently skip stocks that already exist.
                           If False, warn about duplicates (default True for backward compat)
            overwrite_duplicates: If True, replace existing stock entries.
                                 If False, respect skip_duplicates setting.
                                 Note: Without quantity/lot tracking, this replaces the
                                 entire position (purchase date, price, stop %).

        Returns:
            Dictionary with 'added', 'skipped', and 'overwritten' ticker lists

        Raises:
            ValueError: If ticker already exists and both skip_duplicates=False
                       and overwrite_duplicates=False

        Note:
            Until quantity/lot tracking is implemented, each ticker can only
            have one position. Use overwrite_duplicates=True to update an
            existing position's purchase date, price, or stop percentage.
        """
        existing_tickers = set(self.portfolio.keys())
        added: List[str] = []
        skipped: List[str] = []
        overwritten: List[str] = []

        # Process each row
        for n in range(1, max_rows + 1):
            ticker = form_data.get(f'ticker{n}', '').strip().upper()

            if not ticker:
                continue

            # Check if ticker already exists
            if ticker in existing_tickers:
                if overwrite_duplicates:
                    # Replace existing position
                    row_ticker, portfolio_row = self._scrub_row(form_data, n)
                    if row_ticker:
                        self.portfolio[row_ticker] = portfolio_row
                        overwritten.append(row_ticker)
                        warnings.warn(
                            f"Overwriting existing position for {ticker}. "
                            f"Original purchase data has been replaced."
                        )
                elif skip_duplicates:
                    # Silently skip
                    skipped.append(ticker)
                else:
                    # Raise error
                    raise ValueError(
                        f"Stock {ticker} already exists in portfolio. "
                        f"Use overwrite_duplicates=True to replace it, or "
                        f"skip_duplicates=True to ignore duplicates."
                    )
            else:
                # Add new stock
                row_ticker, portfolio_row = self._scrub_row(form_data, n)
                if row_ticker:
                    self.portfolio[row_ticker] = portfolio_row
                    existing_tickers.add(row_ticker)
                    added.append(row_ticker)

        # Warn about skipped duplicates if any
        if skipped and not skip_duplicates:
            warnings.warn(
                f"Skipped existing stocks: {', '.join(skipped)}. "
                f"Use overwrite_duplicates=True to replace them."
            )

        # Update email if provided
        if form_data.get('email'):
            self.email['address'] = form_data['email']
            self.email['frequency'] = form_data.get('frequency', 3)
        elif self.email.get('address'):
            self.email['frequency'] = form_data.get('frequency', 3)

        if added or overwritten:
            self._dirty = True

        return {
            'added': added,
            'skipped': skipped,
            'overwritten': overwritten
        }

    def remove_stocks(self, tickers: List[str]) -> None:
        """
        Remove stocks from the portfolio.

        Args:
            tickers: List of ticker symbols to remove
        """
        for ticker in tickers:
            self.portfolio.pop(ticker.upper(), None)

        self._dirty = True

    def update_portfolio(self) -> None:
        """
        Calculate trailing stop status for all stocks in portfolio.

        This is the core algorithm that:
        1. Fetches historical price data for each stock
        2. Calculates rolling highs since purchase
        3. Determines if trailing stop was violated

        Side effects:
            Updates self.portfolio[symbol]['details'] with current status
            Marks portfolio as dirty (needs saving)

        Note:
            Use to_dataframe() to view the updated portfolio data
        """
        for symbol in self.portfolio.keys():
            try:
                result = self._calculate_stock_status(symbol)
                self.portfolio[symbol]['details'] = {
                    'last_price': result['last_price'],
                    'recent_high': result['recent_high'],
                    'date_high': result['date_high'],
                    'is_sell': result['is_sell']
                }
            except Exception as e:
                warnings.warn(f"Error processing {symbol}: {e}")
                self.portfolio[symbol]['details'] = {
                    'last_price': None,
                    'recent_high': None,
                    'date_high': None,
                    'is_sell': None,
                    'error': str(e)
                }

        self._dirty = True

    def to_dataframe(self) -> pd.DataFrame:
        """
        Return portfolio as a DataFrame.

        Returns:
            DataFrame with columns: ticker, date_purch, price_purch, stop_pct,
                                   last_price, recent_high, date_high, is_sell

        Note:
            If update_portfolio() hasn't been called, detail columns will be missing
        """
        rows = []
        for ticker, data in self.portfolio.items():
            row = {
                'ticker': ticker,
                'date_purch': data['date_purch'],
                'price_purch': data['price_purch'],
                'stop_pct': data['stop_pct']
            }
            # Add details if they exist (from update_portfolio())
            if 'details' in data:
                row.update(data['details'])
            rows.append(row)

        return pd.DataFrame(rows)

    def to_dict(self) -> Dict[str, Any]:
        """
        Return portfolio as a dictionary.

        Returns:
            Dictionary with portfolio name, stocks, email config, and dirty flag
        """
        return {
            'name': self.name,
            'stocks': self.portfolio.copy(),
            'email': self.email.copy(),
            'dirty': self._dirty
        }

    def _calculate_stock_status(self, symbol: str) -> Dict[str, Any]:
        """
        Calculate trailing stop status for a single stock.

        Args:
            symbol: Ticker symbol

        Returns:
            Dictionary with status information
        """
        stock_info = self.portfolio[symbol]
        purchase_date = pd.to_datetime(stock_info['date_purch']).date()
        purchase_price = float(stock_info['price_purch'])
        stop_pct = float(stock_info['stop_pct'])

        # Get historical prices
        prices_df = self._get_price_history(symbol, purchase_date)

        if prices_df.empty:
            raise ValueError(f"No price data available for {symbol}")

        # Calculate rolling high
        prices_df['rolling_high'] = prices_df['Close'].expanding().max()

        # Start with purchase price as the initial high
        current_high = purchase_price
        date_high = purchase_date

        # Check each day for new highs or stop violations
        for idx, row in prices_df.iterrows():
            close = float(row['Close'])

            # New high?
            if close > current_high:
                current_high = close
                date_high = idx.date()

            # Check stop violation
            stop_price = self._calculate_stop_price(stop_pct, current_high)
            if close <= stop_price:
                return {
                    'ticker': symbol,
                    'last_price': close,
                    'recent_high': current_high,
                    'date_high': str(date_high),
                    'is_sell': True,
                    'stop_price': stop_price
                }

        # No stop violation - currently holding
        last_price = float(prices_df['Close'].iloc[-1])
        return {
            'ticker': symbol,
            'last_price': last_price,
            'recent_high': current_high,
            'date_high': str(date_high),
            'is_sell': False,
            'stop_price': self._calculate_stop_price(stop_pct, current_high)
        }

    def _get_price_history(self, ticker: str, start_date: date) -> pd.DataFrame:
        """
        Fetch historical price data for a ticker.

        Args:
            ticker: Stock symbol
            start_date: Starting date for history

        Returns:
            DataFrame with DatetimeIndex and OHLCV columns
        """
        ticker_obj = yf.Ticker(ticker)
        end_date = date.today()

        # Use history() method for single ticker (cleaner than download())
        prices = ticker_obj.history(start=str(start_date), end=str(end_date))

        if prices.empty:
            raise ValueError(f"No price data for {ticker} from {start_date}")

        return prices

    def _calculate_stop_price(self, percent: float, high: float) -> float:
        """
        Calculate the stop-loss price based on percentage and high.

        Args:
            percent: Stop percentage (e.g., 15 for 15%)
            high: Recent high price

        Returns:
            Stop price as float
        """
        return high - (high * percent * 0.01)

    def _scrub_row(self, data: Dict[str, Any], row_num: int) -> Tuple[str, Dict[str, Any]]:
        """
        Validate and process a single row of stock data.

        Args:
            data: Form data dictionary
            row_num: Row number (1-indexed)

        Returns:
            Tuple of (ticker, portfolio_row_dict) or ('', {}) if invalid
        """
        ticker = data.get(f'ticker{row_num}', '').strip().upper()

        if not ticker:
            return ('', {})

        # Validate ticker using yfinance
        if not self._validate_ticker(ticker):
            warnings.warn(f"Invalid ticker: {ticker}")
            return ('', {})

        # Check for purchase date
        purchase_date_str = data.get(f'purchase_date{row_num}')
        if not purchase_date_str:
            return ('', {})

        # Parse purchase date (assume it's a date object or string)
        if isinstance(purchase_date_str, str):
            purchase_date = pd.to_datetime(purchase_date_str).date()
        else:
            purchase_date = purchase_date_str

        # Get actual purchase price from history
        try:
            purchase_date, price_paid = self._purch_date_price(ticker, purchase_date)
        except ValueError as e:
            warnings.warn(f"Could not get price for {ticker}: {e}")
            return ('', {})

        # Override with user-provided price if available
        if data.get(f'price_paid{row_num}'):
            price_paid = float(data[f'price_paid{row_num}'])

        # Check for stop percentage
        stop_percent = data.get(f'stop_percent{row_num}')
        if not stop_percent:
            return ('', {})

        portfolio_row = {
            'date_purch': purchase_date.strftime('%Y-%m-%d'),
            'price_purch': float(price_paid),
            'stop_pct': float(stop_percent)
        }

        return (ticker, portfolio_row)

    @staticmethod
    def _scrub_row_static(data: Dict[str, Any], row_num: int) -> Tuple[str, Dict[str, Any]]:
        """
        Static version of _scrub_row for use in create() classmethod.

        Args:
            data: Form data dictionary
            row_num: Row number (1-indexed)

        Returns:
            Tuple of (ticker, portfolio_row_dict) or ('', {}) if invalid
        """
        ticker = data.get(f'ticker{row_num}', '').strip().upper()

        if not ticker:
            return ('', {})

        # Validate ticker
        if not Portfolio._validate_ticker_static(ticker):
            warnings.warn(f"Invalid ticker: {ticker}")
            return ('', {})

        # Check for purchase date
        purchase_date_str = data.get(f'purchase_date{row_num}')
        if not purchase_date_str:
            return ('', {})

        # Parse purchase date
        if isinstance(purchase_date_str, str):
            purchase_date = pd.to_datetime(purchase_date_str).date()
        else:
            purchase_date = purchase_date_str

        # Get actual purchase price
        try:
            purchase_date, price_paid = Portfolio._purch_date_price_static(ticker, purchase_date)
        except ValueError as e:
            warnings.warn(f"Could not get price for {ticker}: {e}")
            return ('', {})

        # Override with user-provided price if available
        if data.get(f'price_paid{row_num}'):
            price_paid = float(data[f'price_paid{row_num}'])

        # Check for stop percentage
        stop_percent = data.get(f'stop_percent{row_num}')
        if not stop_percent:
            return ('', {})

        portfolio_row = {
            'date_purch': purchase_date.strftime('%Y-%m-%d'),
            'price_purch': float(price_paid),
            'stop_pct': float(stop_percent)
        }

        return (ticker, portfolio_row)

    def _validate_ticker(self, ticker: str) -> bool:
        """
        Validate that a ticker symbol exists and has data.

        Args:
            ticker: Stock symbol to validate

        Returns:
            True if valid, False otherwise
        """
        return self._validate_ticker_static(ticker)

    @staticmethod
    def _validate_ticker_static(ticker: str) -> bool:
        """
        Static method to validate ticker symbol.

        Args:
            ticker: Stock symbol to validate

        Returns:
            True if valid, False otherwise
        """
        try:
            ticker_obj = yf.Ticker(ticker)
            info = ticker_obj.info

            # Check if we got valid data
            if not info or len(info) < 5:
                return False

            # Try to get a recent price
            if 'currentPrice' in info or 'regularMarketPrice' in info:
                return True

            # Fallback: try to get recent history
            hist = ticker_obj.history(period='5d')
            return not hist.empty

        except Exception:
            return False

    def _purch_date_price(self, ticker: str, purchase_date: date) -> Tuple[date, float]:
        """
        Get the actual trading date and price for a purchase.

        Handles weekends and holidays by finding the next valid trading day.

        Args:
            ticker: Stock symbol
            purchase_date: Intended purchase date

        Returns:
            Tuple of (actual_date, closing_price)

        Raises:
            ValueError: If no valid trading data found
        """
        return self._purch_date_price_static(ticker, purchase_date)

    @staticmethod
    def _purch_date_price_static(ticker: str, purchase_date: date) -> Tuple[date, float]:
        """
        Static version for use in create() classmethod.

        Args:
            ticker: Stock symbol
            purchase_date: Intended purchase date

        Returns:
            Tuple of (actual_date, closing_price)

        Raises:
            ValueError: If no valid trading data found
        """
        # Skip weekends
        while purchase_date.weekday() >= 5:  # Saturday=5, Sunday=6
            purchase_date += timedelta(days=1)

        max_attempts = 10

        for _ in range(max_attempts):
            # Try to get data for a small range around this date
            start_date = purchase_date
            end_date = purchase_date + timedelta(days=5)

            try:
                ticker_obj = yf.Ticker(ticker)
                data = ticker_obj.history(start=str(start_date), end=str(end_date))

                if not data.empty:
                    # Get the first available date
                    actual_date = data.index[0].date()
                    close_price = float(data['Close'].iloc[0])
                    return (actual_date, close_price)

            except Exception as e:
                warnings.warn(f"Error fetching {ticker} on {start_date}: {e}")

            # Move to next day
            purchase_date += timedelta(days=1)

            # Skip weekends again
            while purchase_date.weekday() >= 5:
                purchase_date += timedelta(days=1)

        raise ValueError(f"Could not find valid trading data for {ticker} after {max_attempts} attempts")

    @property
    def dirty(self) -> bool:
        """Check if portfolio has unsaved changes."""
        return self._dirty

    def __repr__(self) -> str:
        """String representation of Portfolio."""
        stock_count = len([k for k in self.portfolio.keys() if k != 'email'])
        dirty_marker = '*' if self._dirty else ''
        return f"Portfolio('{self.name}', {stock_count} stocks){dirty_marker}"

    def __str__(self) -> str:
        """User-friendly string representation for print()."""
        tickers = list(self.portfolio.keys())
        stocks_str = ', '.join(tickers) if tickers else 'no stocks'
        dirty_marker = ' (unsaved)' if self._dirty else ''
        return f"Portfolio '{self.name}': {len(tickers)} stocks ({stocks_str}){dirty_marker}"

    def __del__(self):
        """Warn if portfolio is destroyed with unsaved changes."""
        if self._dirty:
            warnings.warn(
                f"Portfolio '{self.name}' has unsaved changes!",
                ResourceWarning,
                stacklevel=2
            )


def get_status(name: str, json_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    Convenience function to load a portfolio and get its status.

    Args:
        name: Portfolio name
        json_dir: Optional directory containing JSON files

    Returns:
        Dictionary of portfolio data with updated status information
    """
    p = Portfolio.load(name, json_dir)
    p.update_portfolio()
    return p.portfolio


if __name__ == "__main__":
    # Example usage
    print("Portfolio Manager - Example Usage")
    print("=" * 50)
    print("\nTo use this module, see portfolio_demo.ipynb")
    print("\nBasic usage:")
    print("  p = Portfolio.load('my_portfolio')")
    print("  results = p.update_portfolio()")
    print("  print(results)")
    print("  p.save()")
