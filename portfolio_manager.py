import sqlite3
import polars as pl
from datetime import datetime, date # Ensure date is imported
from config import PRICE_COLUMN # ALL_SYMBOLS not directly needed here anymore
from data_manager import load_data # To get historical prices

DATABASE_NAME = "portfolios.db"

def get_db_connection():
    conn = sqlite3.connect(DATABASE_NAME)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS portfolios (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT UNIQUE NOT NULL,
        creation_date TEXT NOT NULL
    )
    ''')
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS portfolio_assets (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        portfolio_id INTEGER NOT NULL,
        symbol_key TEXT NOT NULL,
        quantity REAL NOT NULL,
        purchase_price REAL NOT NULL,
        purchase_date TEXT NOT NULL, -- YYYY-MM-DD
        FOREIGN KEY (portfolio_id) REFERENCES portfolios (id)
    )
    ''')
    conn.commit()
    conn.close()

def add_portfolio(name: str) -> int | None:
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT INTO portfolios (name, creation_date) VALUES (?, ?)",
            (name, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        )
        portfolio_id = cursor.lastrowid
        conn.commit()
        return portfolio_id
    except sqlite3.IntegrityError:
        print(f"Portfolio with name '{name}' already exists.")
        return None
    finally:
        conn.close()

def add_asset_to_portfolio(portfolio_id: int, symbol_key: str, quantity: float, purchase_price: float, purchase_date_str: str): # Renamed purchase_date to purchase_date_str
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT INTO portfolio_assets (portfolio_id, symbol_key, quantity, purchase_price, purchase_date) VALUES (?, ?, ?, ?, ?)",
            (portfolio_id, symbol_key, quantity, purchase_price, purchase_date_str)
        )
        conn.commit()
    except Exception as e:
        print(f"Error adding asset: {e}")
    finally:
        conn.close()

def get_all_portfolio_names() -> list[str]:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM portfolios ORDER BY name")
    portfolios = [row['name'] for row in cursor.fetchall()]
    conn.close()
    return portfolios

def get_portfolio_id_by_name(name: str) -> int | None:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM portfolios WHERE name = ?", (name,))
    row = cursor.fetchone()
    conn.close()
    return row['id'] if row else None

def get_portfolio_assets(portfolio_name: str) -> pl.DataFrame | None:
    portfolio_id = get_portfolio_id_by_name(portfolio_name)
    if not portfolio_id:
        return None
    conn = get_db_connection()
    try:
        # Use Polars direct SQL reading if possible and your Polars version supports it well for SQLite
        # For wider compatibility and simplicity with SQLite's dynamic typing, pandas bridge is often easier.
        import pandas as pd
        df_pd = pd.read_sql_query(
            "SELECT symbol_key, quantity, purchase_price, purchase_date FROM portfolio_assets WHERE portfolio_id = ? ORDER BY purchase_date, symbol_key",
            conn,
            params=(portfolio_id,)
        )
        if df_pd.empty:
            return pl.DataFrame(schema={ # Define schema for empty DF
                "symbol_key": pl.Utf8,
                "quantity": pl.Float64,
                "purchase_price": pl.Float64,
                "purchase_date": pl.Utf8 # Stored as text, can be cast later
            })
        df_pl = pl.from_pandas(df_pd)
        # Ensure purchase_date is string, as it's stored
        df_pl = df_pl.with_columns(pl.col("purchase_date").cast(pl.Utf8))
        return df_pl
    except Exception as e:
        print(f"Error fetching portfolio assets: {e}")
        return None
    finally:
        conn.close()

def get_purchase_price_on_date(symbol_key: str, date_str: str, price_column: str) -> float | None:
    """
    Fetches the closing price for a symbol on a specific date or closest prior.
    date_str should be 'YYYY-MM-DD'.
    """
    df_symbol = load_data(symbol_key)
    if df_symbol is None or df_symbol.is_empty():
        print(f"No data loaded for {symbol_key} to get purchase price.")
        return None

    try:
        # Ensure df_symbol['Date'] is Polars Date type
        if df_symbol["Date"].dtype != pl.Date:
            # Attempt conversion if it's datetime or string
            if df_symbol["Date"].dtype == pl.Datetime:
                df_symbol = df_symbol.with_columns(pl.col("Date").dt.date())
            else: # try casting from string or other compatible types
                df_symbol = df_symbol.with_columns(pl.col("Date").str.to_date("%Y-%m-%d", strict=False).cast(pl.Date))
        
        # Convert input date_str to Polars Date object for comparison
        target_date_obj = datetime.strptime(date_str, "%Y-%m-%d").date() # Python date object
        
        # Filter for the exact date
        price_data = df_symbol.filter(pl.col("Date") == target_date_obj)
        
        if price_data.is_empty():
            # If no exact match, find the closest prior date
            price_data = df_symbol.filter(pl.col("Date") <= target_date_obj).sort("Date", descending=True).head(1)

        if not price_data.is_empty() and price_column in price_data.columns:
            price_value = price_data[price_column][0]
            if price_value is not None:
                return float(price_value) # Ensure it's a float
            else:
                print(f"Price is null for {symbol_key} on or before {date_str}.")
                return None
        else:
            print(f"No price data found for {symbol_key} on or before {date_str}.")
            return None
    except Exception as e:
        print(f"Error in get_purchase_price_on_date for {symbol_key} on {date_str}: {e}")
        return None

# Initialize DB
init_db()