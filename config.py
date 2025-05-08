import os
import re # Import regex for cleaning filenames

# --- Core Settings ---
START_DATE = "2020-04-01"

# Initial symbol (NSE 50 Index)
INITIAL_SYMBOLS = {"NIFTY_50": "^NSEI"}

# Nifty 50 Constituent Stocks (Yahoo Finance Tickers)
# !! IMPORTANT: This list needs occasional verification as index constituents change !!
# List as of early 2024 - Please verify with a current source if needed.
NIFTY_50_STOCKS = {
    "Adani Enterprises": "ADANIENT.NS",
    "Adani Ports": "ADANIPORTS.NS",
    "Apollo Hospitals": "APOLLOHOSP.NS",
    "Asian Paints": "ASIANPAINT.NS",
    "Axis Bank": "AXISBANK.NS",
    "Bajaj Auto": "BAJAJ-AUTO.NS",
    "Bajaj Finance": "BAJFINANCE.NS",
    "Bajaj Finserv": "BAJAJFINSV.NS",
    "Bharat Electronics": "BEL.NS",
    "Bharti Airtel": "BHARTIARTL.NS",
    #"Britannia Industries": "BRITANNIA.NS",
    "Cipla": "CIPLA.NS",
    "Coal India": "COALINDIA.NS",
    #"Divi's Laboratories": "DIVISLAB.NS",
    "Dr. Reddy's Labs": "DRREDDY.NS",
    "Eicher Motors": "EICHERMOT.NS",
    "Eternal Ltd": "ETERNAL.NS",
    "Grasim Industries": "GRASIM.NS",
    "HCL Technologies": "HCLTECH.NS",
    "HDFC Bank": "HDFCBANK.NS", # Merged entity
    "HDFC Life Insurance": "HDFCLIFE.NS",
    "Hero MotoCorp": "HEROMOTOCO.NS",
    "Hindalco Industries": "HINDALCO.NS",
    "Hindustan Unilever": "HINDUNILVR.NS",
    "ICICI Bank": "ICICIBANK.NS",
    "ITC": "ITC.NS",
    "IndusInd Bank": "INDUSINDBK.NS",
    "Infosys": "INFY.NS",
    "Jio Financial Services Ltd": "JIOFIN.NS",
    "JSW Steel": "JSWSTEEL.NS",
    "Kotak Mahindra Bank": "KOTAKBANK.NS",
    "Larsen & Toubro": "LT.NS",
    "Mahindra & Mahindra": "M&M.NS",
    "Maruti Suzuki": "MARUTI.NS",
    "NTPC": "NTPC.NS",
    "Nestle India": "NESTLEIND.NS",
    "ONGC": "ONGC.NS",
    "Power Grid Corp": "POWERGRID.NS",
    "Reliance Industries": "RELIANCE.NS",
    "SBI Life Insurance": "SBILIFE.NS",
    "Shriram Finance": "SHRIRAMFIN.NS",
    "State Bank of India": "SBIN.NS",
    "Sun Pharmaceutical": "SUNPHARMA.NS",
    "Tata Consultancy Services": "TCS.NS",
    "Tata Consumer Products": "TATACONSUM.NS",
    "Tata Motors": "TATAMOTORS.NS",
    "Tata Steel": "TATASTEEL.NS",
    "Tech Mahindra": "TECHM.NS",
    "Titan Company": "TITAN.NS",
    "Trent Ltd": "TRENT.NS",
    "UltraTech Cement": "ULTRACEMCO.NS",
    "Wipro": "WIPRO.NS",   
}


# Combine all symbols for the dropdown (Index first, then stocks sorted alphabetically)
ALL_SYMBOLS = {**INITIAL_SYMBOLS, **dict(sorted(NIFTY_50_STOCKS.items()))}


# --- Data Storage ---
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True) # Ensure data directory exists

# Parquet file naming convention
def get_parquet_path(symbol_key: str) -> str:
    """Generates the path for a symbol's parquet file using a cleaned key."""
    # Replace problematic characters for filenames (spaces, &, -, .)
    clean_key = re.sub(r'[ &.\-]+', '_', symbol_key)
    # Remove any remaining non-alphanumeric characters (except underscore)
    clean_key = re.sub(r'[^\w_]', '', clean_key)
    return os.path.join(DATA_DIR, f"{clean_key}.parquet")

# --- Analysis ---
PRICE_COLUMN = "Adj Close" # Or 'Close'

# --- Financial Year Definition ---
FY_START_MONTH = 4 # April