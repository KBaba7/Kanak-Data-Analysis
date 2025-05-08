import polars as pl
import numpy as np

def define_states(returns_series: pl.Series, n_states: int = 5) -> tuple[pl.Series, list[float], list[str]]:
    """
    Discretizes returns into states based on std deviations using Polars.
    Returns the state series (indices 0 to n_states-1), bin edges, and state labels.
    """
    returns_series_non_null = returns_series.drop_nulls()
    if returns_series_non_null.is_empty():
        # Return a series of nulls with the original length
        return pl.Series(values=[None]*len(returns_series), dtype=pl.Int8), [], []

    # Method 2: Standard Deviation based states
    mean = returns_series_non_null.mean()
    std = returns_series_non_null.std()

    if mean is None or std is None or std == 0: # Handle cases with no variance
        # Assign a single state (e.g., 0) to non-null values
        # We need to map this back to the original series' length including nulls
        # For simplicity in this case, let's return a single state label and the mean bin
        state_labels = ["Neutral"]
        bin_edges = [-np.inf, np.inf] # Only one bin
        n_states = 1
        # Create a series of 0s for non-nulls, null otherwise
        states = pl.when(returns_series.is_not_null()).then(pl.lit(0, dtype=pl.Int8)).otherwise(pl.lit(None, dtype=pl.Int8))
        return states, bin_edges, state_labels

    # Define bin edges based on std dev (Example: 5 states)
    bin_edges = [
        -np.inf,
        mean - 1.5 * std,
        mean - 0.5 * std,
        mean + 0.5 * std,
        mean + 1.5 * std,
        np.inf
    ]
    state_labels = ["Strong Down", "Mod Down", "Neutral", "Mod Up", "Strong Up"]
    n_states = len(state_labels) # Ensure n_states matches labels

    # Use Polars expressions to assign states based on bins
    # Start with null state
    state_expr = pl.lit(None, dtype=pl.Int8)
    # Iterate backwards through bins to assign states
    for i in range(n_states - 1, -1, -1):
        state_expr = pl.when(pl.col("Daily Return") >= bin_edges[i]) \
                       .then(pl.lit(i, dtype=pl.Int8)) \
                       .otherwise(state_expr)

    # Apply the expression, requires 'Daily Return' column to be present
    # It's better to pass the full dataframe or apply this outside
    # Let's modify to work just on the series input temporarily
    # We'll apply it properly in the main app logic

    # --- Temporary approach directly on series (less efficient) ---
    # This is complex. It's easier to do this within the DataFrame context in app.py.
    # Let's return the bins and labels, and do the assignment in app.py
    # --- Modification: Return bins/labels, assign state in app.py ---

    return None, bin_edges, state_labels # Return None for state_series, will compute in app


def calculate_transition_matrix(state_series: pl.Series, n_states: int) -> np.ndarray:
    """Calculates the transition probability matrix from a Polars series of states."""
    # Drop nulls before calculating transitions
    state_series_non_null = state_series.drop_nulls().cast(pl.Int64) # Cast to int for indexing

    if state_series_non_null.len() <= 1:
         return np.zeros((n_states, n_states))

    # Get pairs of consecutive states (state_t-1, state_t)
    # Ensure correct alignment by creating shifted series first
    previous_states = state_series_non_null.shift(1)
    current_states = state_series_non_null

    # Combine into a DataFrame, drop the first row where previous_state is null
    transitions = pl.DataFrame({
        'previous_state': previous_states,
        'current_state': current_states
    }).slice(1) # Remove first row with null shift

    if transitions.is_empty():
        return np.zeros((n_states, n_states))

    # Count transitions
    counts = transitions.group_by(['previous_state', 'current_state']).agg(
        pl.count().alias('count')
    )

    # Create the transition matrix (rows = previous state, cols = current state)
    matrix = np.zeros((n_states, n_states))
    for row in counts.iter_rows(named=True):
        # Get state indices safely
        prev_s = row['previous_state']
        curr_s = row['current_state']
        # Ensure state indices are within bounds (already cast to Int64, should be fine)
        if 0 <= prev_s < n_states and 0 <= curr_s < n_states:
            matrix[prev_s, curr_s] = row['count']

    # Normalize rows to get probabilities
    row_sums = matrix.sum(axis=1)
    # Avoid division by zero for states that were never entered or never transitioned from
    # Create a mask for rows where sum > 0
    mask = row_sums > 0
    prob_matrix = np.zeros_like(matrix, dtype=float)
    # Calculate probabilities only for rows where the sum is positive
    prob_matrix[mask] = matrix[mask] / row_sums[mask, np.newaxis]

    return prob_matrix


def predict_next_state(current_state: int, transition_matrix: np.ndarray) -> tuple[int, float]:
    """Predicts the most likely next state and its probability."""
    if not (0 <= current_state < transition_matrix.shape[0]):
        raise ValueError(f"Invalid current state index: {current_state}")

    probabilities = transition_matrix[current_state, :]
    if np.all(probabilities == 0): # Handle case where the current state has no observed transitions out
         # Predict staying in the same state with 0 probability as an indicator
         return int(current_state), 0.0

    next_state = np.argmax(probabilities)
    probability = probabilities[next_state]
    return int(next_state), float(probability)