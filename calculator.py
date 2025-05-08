import polars as pl
from config import PRICE_COLUMN, FY_START_MONTH
import numpy as np # For pl.log -> np.log if polars.log is not direct

def calculate_daily_returns(df: pl.DataFrame) -> pl.DataFrame:
    """Calculates daily arithmetic returns based on the specified price column."""
    if PRICE_COLUMN not in df.columns:
        raise ValueError(f"Price column '{PRICE_COLUMN}' not found in DataFrame.")
    if df.height <= 1:
        return df.with_columns(pl.lit(None).cast(pl.Float64).alias("Daily Return"))

    # Calculate percentage return: (Price_t / Price_{t-1}) - 1
    df = df.with_columns(
        pl.col(PRICE_COLUMN).pct_change().alias("Daily Return")
    )
    # The first row will have a null return, which is expected
    return df

# --- NEW FUNCTION ---
def calculate_log_returns(df: pl.DataFrame) -> pl.DataFrame:
    """Calculates daily log returns based on the specified price column."""
    if PRICE_COLUMN not in df.columns:
        raise ValueError(f"Price column '{PRICE_COLUMN}' not found in DataFrame.")
    if df.height <= 1:
        return df.with_columns(pl.lit(None).cast(pl.Float64).alias("Log Return"))

    # Calculate log return: ln(Price_t) - ln(Price_{t-1})
    # Ensure price is positive before taking log
    df = df.with_columns(
        pl.when(pl.col(PRICE_COLUMN) > 0)
        .then(pl.col(PRICE_COLUMN).log()) # Polars' log is natural log (ln)
        .otherwise(None) # or handle as error/specific value
        .diff() # Computes difference between current and previous row
        .alias("Log Return")
    )
    # The first row will have a null return, which is expected
    return df
# --- END NEW FUNCTION ---

def add_time_periods(df: pl.DataFrame) -> pl.DataFrame:
    """Adds Financial Year, Quarter, and Month columns to the DataFrame."""
    df = df.with_columns([
        pl.col("Date").dt.year().alias("Year"),
        pl.col("Date").dt.month().alias("Month"),
        pl.col("Date").dt.quarter().alias("QuarterNum") # Quarter number (1-4)
    ])

    # Calculate Financial Year (e.g., FY21 covers Apr 2020 - Mar 2021)
    # If month is before April (1, 2, 3), FY is Year-1, otherwise FY is Year
    df = df.with_columns(
        pl.when(pl.col("Month") < FY_START_MONTH)
        .then(pl.col("Year") - 1)
        .otherwise(pl.col("Year"))
        .alias("FY_Start_Year") # The year the FY starts in (e.g., 2020 for FY21)
    )
    df = df.with_columns(
        pl.concat_str([
            pl.lit("FY"),
            (pl.col("FY_Start_Year") + 1).cast(pl.Utf8).str.slice(-2) # Get last 2 digits (e.g., 21 for 2020 start year)
        ]).alias("Financial Year")
    )

    # Create a Quarter label like 'Q1_FY21'
    # Map calendar quarter to financial quarter
    # Q1: Apr-Jun (Cal Q2), Q2: Jul-Sep (Cal Q3), Q3: Oct-Dec (Cal Q4), Q4: Jan-Mar (Cal Q1)
    df = df.with_columns(
        pl.when(pl.col("Month").is_in([4, 5, 6])).then(1)
        .when(pl.col("Month").is_in([7, 8, 9])).then(2)
        .when(pl.col("Month").is_in([10, 11, 12])).then(3)
        .otherwise(4) # Jan, Feb, Mar
        .alias("FY_Quarter")
    )
    df = df.with_columns(
        pl.concat_str([
            pl.lit("Q"),
            pl.col("FY_Quarter").cast(pl.Utf8),
            pl.lit("_"),
            pl.col("Financial Year")
        ]).alias("Quarter")
    )

    # Create Month label like 'Apr_2020'
    df = df.with_columns(
        pl.col("Date").dt.strftime("%b_%Y").alias("Month Year") # e.g., Apr_2020
    )

    return df.drop(["Year", "FY_Start_Year", "QuarterNum", "FY_Quarter"])

def assign_volatility_state(df: pl.DataFrame, mean_ret: float, std_ret: float) -> pl.DataFrame:
    """
    Assigns a state based on daily ARITHMETIC return volatility.
    The states are defined based on the mean and std of arithmetic returns.
    """
    if "Daily Return" not in df.columns: # Still based on arithmetic Daily Return
        raise ValueError("Daily Return column not calculated yet for state assignment.")
    if mean_ret is None or std_ret is None or std_ret == 0:
         return df.with_columns(pl.lit(None).cast(pl.Utf8).alias("State"))

    df = df.with_columns(
        pl.when(pl.col("Daily Return").is_null()).then(pl.lit(None))
        .when(pl.col("Daily Return") < mean_ret - 2 * std_ret).then(pl.lit("Large Loss: < -2σ"))
        .when(pl.col("Daily Return") < mean_ret - 1 * std_ret).then(pl.lit("Moderate Loss: -2σ to -1σ"))
        .when(pl.col("Daily Return") < mean_ret + 1 * std_ret).then(pl.lit("Normal/Flat: -1σ to +1σ"))
        .when(pl.col("Daily Return") < mean_ret + 2 * std_ret).then(pl.lit("Moderate Gain: +1σ to +2σ"))
        .otherwise(pl.lit("Large Gain: > +2σ"))
        .alias("State")
    )
    return df

def calculate_period_stats(df: pl.DataFrame, group_by_col: str) -> pl.DataFrame:
    """Calculates mean return and standard deviation for specified periods for both arithmetic and log returns."""
    if "Daily Return" not in df.columns:
        raise ValueError("Daily Return column not calculated yet.")
    if "Log Return" not in df.columns: # --- NEW CHECK ---
        raise ValueError("Log Return column not calculated yet.")
    if group_by_col not in df.columns:
        raise ValueError(f"Grouping column '{group_by_col}' not found.")

    aggregations = [
        pl.mean("Daily Return").alias("Mean Arithmetic Return"),
        pl.std("Daily Return").alias("Std Dev Arithmetic Return"),
        pl.mean("Log Return").alias("Mean Log Return"), # --- NEW ---
        pl.std("Log Return").alias("Std Dev Log Return"),   # --- NEW ---
        pl.count().alias("Trading Days")
    ]
    
    # Only add state-related stats if "State" column exists and is not all null
    if "State" in df.columns and not df["State"].is_null().all():
        # Example: Count occurrences of each state. Could also do % of days in each state.
        # This part might need more thought depending on what state stats are desired per period.
        # For now, let's keep it simple or omit it from here and do it separately if complex.
        # For simplicity, we'll omit per-period state distribution from this function for now.
        pass

    stats = df.group_by(group_by_col).agg(aggregations).sort(group_by_col)

    # Add std dev ranges for Arithmetic Return context
    stats = stats.with_columns([
        (pl.col("Mean Arithmetic Return") - pl.col("Std Dev Arithmetic Return")).alias("Arith -1 Std Dev"),
        (pl.col("Mean Arithmetic Return") + pl.col("Std Dev Arithmetic Return")).alias("Arith +1 Std Dev"),
        (pl.col("Mean Arithmetic Return") - 2 * pl.col("Std Dev Arithmetic Return")).alias("Arith -2 Std Dev"),
        (pl.col("Mean Arithmetic Return") + 2 * pl.col("Std Dev Arithmetic Return")).alias("Arith +2 Std Dev"),
        # (pl.col("Mean Arithmetic Return") - 3 * pl.col("Std Dev Arithmetic Return")).alias("Arith -3 Std Dev"), # Can be verbose
        # (pl.col("Mean Arithmetic Return") + 3 * pl.col("Std Dev Arithmetic Return")).alias("Arith +3 Std Dev"),
    ])

    # Format percentages/small numbers nicely (multiplying by 100)
    # Log returns are also often small, so scaling them can be useful for display.
    percent_cols = [
        "Mean Arithmetic Return", "Std Dev Arithmetic Return",
        "Arith -1 Std Dev", "Arith +1 Std Dev", "Arith -2 Std Dev", "Arith +2 Std Dev",
        # "Arith -3 Std Dev", "Arith +3 Std Dev",
        "Mean Log Return", "Std Dev Log Return" # --- NEW ---
    ]
    for col_name in percent_cols:
        if col_name in stats.columns: # Check if column exists (e.g. std dev ranges)
            stats = stats.with_columns(
                (pl.col(col_name) * 100).round(3).alias(f"{col_name} (%)")
            )
            # Optional: drop original non-percentage column if desired
            # stats = stats.drop(col_name)
    
    # Rename original columns if scaled versions were created, or select specific columns
    # For now, the new scaled columns have "(%)" appended, so originals are still there if needed.

    return stats


def calculate_transition_matrix(df: pl.DataFrame) -> tuple[pl.DataFrame | None, list[str] | None]:
    """
    Calculates the Markov Chain transition matrix based on the 'State' column.
    (No changes needed here as states are based on arithmetic returns)
    """
    if "State" not in df.columns or df['State'].drop_nulls().is_empty():
        return None, None

    state_order = sorted(df['State'].drop_nulls().unique().to_list())

    df_shifted = df.with_columns(
        pl.col("State").shift(-1).alias("Next State")
    ).drop_nulls(["State", "Next State"])

    if df_shifted.is_empty():
        return None, state_order

    transitions = df_shifted.group_by(["State", "Next State"]).agg(
        pl.count().alias("Count")
    )

    transition_counts_matrix = transitions.pivot(
        index="State",
        columns="Next State",
        values="Count"
    ).fill_null(0)

    current_cols = transition_counts_matrix.columns[1:]
    for state in state_order:
        if state not in transition_counts_matrix['State'].to_list():
            zero_row_data = {'State': [state]}
            for col_name in current_cols:
                zero_row_data[col_name] = 0
            zero_row = pl.DataFrame(zero_row_data, schema=transition_counts_matrix.schema)
            transition_counts_matrix = pl.concat([transition_counts_matrix, zero_row], how="vertical_relaxed") # Use vertical_relaxed
        if state not in current_cols:
             transition_counts_matrix = transition_counts_matrix.with_columns(pl.lit(0).cast(pl.Int64).alias(state)) # Ensure type consistency

    transition_counts_matrix = transition_counts_matrix.select(['State'] + state_order)
    transition_counts_matrix = transition_counts_matrix.sort('State')

    row_sums = transition_counts_matrix.select(pl.sum_horizontal(pl.exclude('State'))).to_series()

    prob_cols = []
    state_cols = transition_counts_matrix.columns[1:]
    for col in state_cols:
         prob_cols.append(
             pl.when(row_sums == 0).then(0.0)
             .otherwise(pl.col(col) / row_sums)
             .alias(f"{col}_Prob")
         )

    transition_prob_matrix = transition_counts_matrix.with_columns(prob_cols)

    final_matrix_cols = ['State'] + [f"{s}_Prob" for s in state_order]
    transition_prob_matrix = transition_prob_matrix.select(final_matrix_cols)
    transition_prob_matrix = transition_prob_matrix.rename({f"{s}_Prob": s for s in state_order})

    return transition_prob_matrix, state_order

# --- MODIFIED FUNCTION ---
def calculate_avg_returns_per_state(df: pl.DataFrame) -> pl.DataFrame | None:
    """Calculates the average daily arithmetic and log returns for each state."""
    if "State" not in df.columns or \
       "Daily Return" not in df.columns or \
       "Log Return" not in df.columns: # --- NEW CHECK ---
        return None

    avg_returns = df.drop_nulls(["State", "Daily Return", "Log Return"]).group_by("State").agg([
        pl.mean("Daily Return").alias("Avg Arithmetic Return"),
        pl.mean("Log Return").alias("Avg Log Return") # --- NEW ---
    ]).sort("State")

    return avg_returns