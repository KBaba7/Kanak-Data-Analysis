import yfinance as yf
import polars as pl
import pandas as pd
from datetime import datetime, timedelta
import os
import logging
from curl_cffi import requests

from config import START_DATE, DATA_DIR, get_parquet_path, ALL_SYMBOLS

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def fetch_data(ticker: str, start_date: str, end_date: str) -> pl.DataFrame | None:
    """Fetches historical stock data from Yahoo Finance."""
    try:
        logging.info(f"Fetching data for {ticker} from {start_date} to {end_date}")
        session = requests.Session(impersonate="chrome")
        stock = yf.Ticker(ticker, session=session)
        # Fetch data - use pandas first as yfinance returns pandas
        pd_df = stock.history(start=start_date, end=end_date, auto_adjust=False) # auto_adjust=False gives 'Adj Close'

        if pd_df.empty:
            logging.warning(f"No data returned for {ticker} in the specified range.")
            return None

        # Convert to Polars DataFrame
        pl_df = pl.from_pandas(pd_df.reset_index()) # Reset index to get Date as column

        # Select and rename columns for consistency
        pl_df = pl_df.select(
            pl.col("Date").cast(pl.Date),
            pl.col("Open"),
            pl.col("High"),
            pl.col("Low"),
            pl.col("Close"),
            pl.col("Adj Close"),
            pl.col("Volume")
        ).sort("Date") # Ensure data is sorted by date

        logging.info(f"Successfully fetched {len(pl_df)} rows for {ticker}")
        return pl_df
    except Exception as e:
        logging.error(f"Error fetching data for {ticker}: {e}")
        return None

def save_data(df: pl.DataFrame, symbol_key: str):
    """Saves a Polars DataFrame to a Parquet file."""
    path = get_parquet_path(symbol_key)
    try:
        df.write_parquet(path)
        logging.info(f"Data for {symbol_key} saved to {path}")
    except Exception as e:
        logging.error(f"Error saving data for {symbol_key} to {path}: {e}")

def load_data(symbol_key: str) -> pl.DataFrame | None:
    """Loads data from a Parquet file."""
    path = get_parquet_path(symbol_key)
    if not os.path.exists(path):
        logging.warning(f"No local data file found for {symbol_key} at {path}")
        return None
    try:
        df = pl.read_parquet(path)
        logging.info(f"Data for {symbol_key} loaded from {path}")
        # Ensure Date is correct type after loading
        if "Date" in df.columns and df["Date"].dtype != pl.Date:
             df = df.with_columns(pl.col("Date").cast(pl.Date))
        return df.sort("Date")
    except Exception as e:
        logging.error(f"Error loading data for {symbol_key} from {path}: {e}")
        return None

def update_data_for_symbol(symbol_key: str, ticker: str):
    """Loads existing data, fetches new data, and saves the combined data."""
    logging.info(f"--- Updating data for {symbol_key} ({ticker}) ---")
    existing_df = load_data(symbol_key)
    today_str = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d') # Fetch up to tomorrow to include today's close

    if existing_df is not None and not existing_df.is_empty():
        last_date = existing_df['Date'].max()
        start_fetch_date = (last_date + timedelta(days=1)).strftime('%Y-%m-%d')
        logging.info(f"Found existing data for {symbol_key}. Last date: {last_date}. Fetching new data from {start_fetch_date}.")

        if start_fetch_date >= today_str:
             logging.info(f"Data for {symbol_key} is already up-to-date.")
             return existing_df # Return existing data if already up-to-date

        new_data_df = fetch_data(ticker, start_date=start_fetch_date, end_date=today_str)

        if new_data_df is not None and not new_data_df.is_empty():
            # Combine old and new data
            combined_df = pl.concat([existing_df, new_data_df], how="vertical_relaxed").unique(subset=["Date"], keep="last").sort("Date")
            save_data(combined_df, symbol_key)
            logging.info(f"Successfully updated data for {symbol_key}.")
            return combined_df
        else:
            logging.warning(f"No new data fetched for {symbol_key}. Using existing data.")
            return existing_df # Return the old data if fetch failed or returned empty
    else:
        # No existing data, fetch all data from start date
        logging.info(f"No existing data found for {symbol_key}. Fetching all data from {START_DATE}.")
        full_data_df = fetch_data(ticker, start_date=START_DATE, end_date=today_str)
        if full_data_df is not None and not full_data_df.is_empty():
            save_data(full_data_df, symbol_key)
            return full_data_df
        else:
            logging.error(f"Failed to fetch initial data for {symbol_key}.")
            return None

def update_all_data():
    """Updates data for all symbols defined in config.ALL_SYMBOLS."""
    all_data = {}
    for symbol_key, ticker in ALL_SYMBOLS.items():
        df = update_data_for_symbol(symbol_key, ticker)
        if df is not None:
            all_data[symbol_key] = df
    logging.info("--- Finished updating all symbol data ---")
    return all_data