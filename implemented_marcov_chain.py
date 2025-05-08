import streamlit as st
import polars as pl
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from uuid import uuid4

from data_manager import load_data, update_data_for_symbol, update_all_data
from calculator import calculate_daily_returns, add_time_periods, calculate_period_stats
from config import INITIAL_SYMBOLS, ALL_SYMBOLS

# --- Page Config ---
st.set_page_config(
    page_title="kaNak Data Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Helper Functions ---
@st.cache_data(ttl=3600)
def load_symbol_data(symbol_key):
    """Loads data for a specific symbol, updates if necessary."""
    ticker = ALL_SYMBOLS.get(symbol_key)
    if not ticker:
        st.error(f"Ticker for symbol key '{symbol_key}' not found in config.")
        return None
    df = load_data(symbol_key)
    if df is None:
        st.info(f"Local data for {symbol_key} not found. Fetching...")
        df = update_data_for_symbol(symbol_key, ticker)
    return df

@st.cache_data(ttl=3600)
def get_processed_data(df: pl.DataFrame):
    """Calculates returns and adds time periods."""
    if df is None or df.is_empty():
        return None, None, None, None
    df_returns = calculate_daily_returns(df)
    df_processed = add_time_periods(df_returns)
    fy_stats = calculate_period_stats(df_processed.drop_nulls("Daily Return"), "Financial Year")
    q_stats = calculate_period_stats(df_processed.drop_nulls("Daily Return"), "Quarter")
    m_stats = calculate_period_stats(df_processed.drop_nulls("Daily Return"), "Month Year")
    return df_processed, fy_stats, q_stats, m_stats

def create_markov_chain(df: pl.DataFrame):
    """Creates a Markov Chain model for daily returns and predicts next day's Adj Close."""
    if df is None or df.is_empty():
        return None, None, None, None, None, None

    returns = df["Daily Return"].drop_nulls() * 100
    if returns.is_empty():
        return None, None, None, None, None, None
    
    boundaries = [-1.0, 1.0]
    state_names = ["Negative", "Near-Zero", "Positive"]
    
    # Assign states to each return using a list
    states = []
    for ret in returns:
        if ret < boundaries[0]:
            states.append(state_names[0])
        elif ret <= boundaries[1]:
            states.append(state_names[1])
        else:
            states.append(state_names[2])
    
    # Create Series from the list
    states_series = pl.Series("State", states)
    
    # Add states to DataFrame, aligning with non-null returns
    df_with_returns = df.filter(pl.col("Daily Return").is_not_null())
    df_with_states = df_with_returns.with_columns(states_series)
    
    # Calculate average return per state (in decimal form for price calculation)
    avg_returns = {}
    for state in state_names:
        state_returns = df_with_states.filter(pl.col("State") == state)["Daily Return"]
        avg_returns[state] = state_returns.mean() if not state_returns.is_empty() else 0.0
    
    # Calculate transition matrix
    n_states = len(state_names)
    transition_matrix = np.zeros((n_states, n_states))
    state_indices = {name: idx for idx, name in enumerate(state_names)}
    
    # Iterate through consecutive days to count transitions
    prev_state = None
    for state in df_with_states["State"]:
        if prev_state is not None and state is not None:
            transition_matrix[state_indices[prev_state], state_indices[state]] += 1
        prev_state = state
    
    # Normalize to get probabilities
    row_sums = transition_matrix.sum(axis=1)
    transition_matrix = np.divide(
        transition_matrix,
        row_sums[:, np.newaxis],
        where=row_sums[:, np.newaxis] != 0,
        out=np.zeros_like(transition_matrix)
    )
    
    # Predict next day's state and price
    latest_state = df_with_states["State"][-1] if not df_with_states.is_empty() else None
    latest_price = df_with_states["Adj Close"][-1] if not df_with_states.is_empty() else None
    if latest_state is None or latest_price is None:
        return state_names, transition_matrix, None, None, df_with_states, avg_returns
    
    latest_state_idx = state_indices[latest_state]
    next_state_probs = transition_matrix[latest_state_idx]
    predicted_state = state_names[np.argmax(next_state_probs)]
    
    # Calculate predicted price using the average return of the predicted state
    predicted_return = avg_returns[predicted_state]  # In decimal (e.g., 0.01 for 1%)
    predicted_price = latest_price * (1 + predicted_return)
    
    return state_names, transition_matrix, predicted_state, predicted_price, df_with_states, avg_returns

# --- Sidebar ---
st.sidebar.title("📈 kaNak Analysis Tool")
st.sidebar.markdown("Select analysis parameters.")

available_symbols = list(ALL_SYMBOLS.keys())
selected_symbol_key = st.sidebar.selectbox(
    "Select Symbol:",
    options=available_symbols,
    index=available_symbols.index("NIFTY_50")
)

if st.sidebar.button(f"🔄 Update Data for {selected_symbol_key}"):
    ticker = ALL_SYMBOLS.get(selected_symbol_key)
    if ticker:
        with st.spinner(f"Updating data for {selected_symbol_key}..."):
            update_data_for_symbol(selected_symbol_key, ticker)
        st.sidebar.success(f"Data updated for {selected_symbol_key}!")
        st.cache_data.clear()
    else:
        st.sidebar.error("Symbol ticker not found.")

# --- Main App Area ---
st.title(f"📊 Daily Return Analysis: {selected_symbol_key}")

# Load and process data
df_raw = load_symbol_data(selected_symbol_key)

if df_raw is None or df_raw.is_empty():
    st.error(f"Could not load or fetch data for {selected_symbol_key}. Please check logs or try updating.")
else:
    df_processed, fy_stats, q_stats, m_stats = get_processed_data(df_raw)

    if df_processed is None or df_processed.is_empty():
        st.warning("Data loaded, but processing failed or resulted in empty data.")
    else:
        # --- Section 1: Return Distribution ---
        st.header("Return Distribution")
        returns_series = df_processed['Daily Return'].drop_nulls() * 100

        if not returns_series.is_empty():
            fig_hist = px.histogram(
                returns_series.to_pandas(),
                nbins=100,
                title=f"Distribution of Daily Returns (%) for {selected_symbol_key}",
                labels={'value': 'Daily Return (%)'}
            )
            fig_hist.update_layout(showlegend=False)

            mean_ret = returns_series.mean()
            std_dev_ret = returns_series.std()
            if mean_ret is not None and std_dev_ret is not None:
                fig_hist.add_vline(x=mean_ret, line_dash="dash", line_color="red", annotation_text="Mean")
                for i in range(1, 4):
                    fig_hist.add_vline(x=mean_ret + i * std_dev_ret, line_dash="dot", line_color="green", annotation_text=f"+{i}σ")
                    fig_hist.add_vline(x=mean_ret - i * std_dev_ret, line_dash="dot", line_color="orange", annotation_text=f"-{i}σ")

            st.plotly_chart(fig_hist, use_container_width=True)

            st.subheader("Overall Statistics (Full Period)")
            overall_mean = returns_series.mean()
            overall_std = returns_series.std()
            overall_count = len(returns_series)
            if overall_mean is not None and overall_std is not None:
                st.metric(label="Mean Daily Return (%)", value=f"{overall_mean:.3f}")
                st.metric(label="Standard Deviation (%)", value=f"{overall_std:.3f}")
                st.metric(label="Trading Days", value=f"{overall_count}")

        # --- Section 2: Period Analysis ---
        st.header("Analysis by Period")

        all_fy = ["All"] + sorted(df_processed['Financial Year'].unique().to_list(), reverse=True)
        all_q = ["All"] + sorted(df_processed['Quarter'].unique().to_list(), reverse=True)
        all_m = ["All"] + sorted(df_processed['Month Year'].unique().to_list(), reverse=True)

        col1, col2, col3 = st.columns(3)
        selected_fy = col1.selectbox("Select Financial Year:", all_fy)
        selected_q = col2.selectbox("Select Quarter:", all_q)
        selected_m = col3.selectbox("Select Month:", all_m)

        filtered_df = df_processed.clone()
        if selected_fy != "All":
            filtered_df = filtered_df.filter(pl.col("Financial Year") == selected_fy)
        if selected_q != "All":
            filtered_df = filtered_df.filter(pl.col("Quarter") == selected_q)
        if selected_m != "All":
            filtered_df = filtered_df.filter(pl.col("Month Year") == selected_m)

        st.subheader("Statistics for Selected Period")
        if filtered_df.is_empty():
            st.warning("No data available for the selected period.")
        else:
            period_mean = filtered_df['Daily Return'].mean()
            period_std = filtered_df['Daily Return'].std()
            period_count = filtered_df['Daily Return'].drop_nulls().count()

            if period_mean is not None and period_std is not None:
                col_m, col_s, col_c = st.columns(3)
                col_m.metric("Mean Return (%)", f"{(period_mean * 100):.3f}")
                col_s.metric("Std Dev (%)", f"{(period_std * 100):.3f}")
                col_c.metric("Trading Days", f"{period_count}")

                st.write("Standard Deviation Ranges:")
                st.text(f" Mean ± 1σ: [{(period_mean - period_std)*100:.3f}%, {(period_mean + period_std)*100:.3f}%]")
                st.text(f" Mean ± 2σ: [{(period_mean - 2*period_std)*100:.3f}%, {(period_mean + 2*period_std)*100:.3f}%]")
                st.text(f" Mean ± 3σ: [{(period_mean - 3*period_std)*100:.3f}%, {(period_mean + 3*period_std)*100:.3f}%]")

            st.dataframe(
                filtered_df.select([
                    "Date", "Adj Close", "Daily Return", "Financial Year", "Quarter", "Month Year"
                ]).sort("Date", descending=True),
                use_container_width=True
            )

        # --- Section 3: Ready Reckoners ---
        st.header("Ready Reckoners")
        tab1, tab2, tab3 = st.tabs(["Financial Years", "Quarters", "Months"])

        with tab1:
            st.subheader("Stats per Financial Year")
            if fy_stats is not None:
                st.dataframe(fy_stats, use_container_width=True)
            else:
                st.warning("Could not calculate Financial Year stats.")

        with tab2:
            st.subheader("Stats per Quarter")
            if q_stats is not None:
                st.dataframe(q_stats, use_container_width=True)
            else:
                st.warning("Could not calculate Quarterly stats.")

        with tab3:
            st.subheader("Stats per Month")
            if m_stats is not None:
                st.dataframe(m_stats, use_container_width=True)
            else:
                st.warning("Could not calculate Monthly stats.")

        # --- Section 4: Markov Chain Analysis ---
        st.header("Markov Chain Analysis")
        state_names, transition_matrix, predicted_state, predicted_price, df_with_states, avg_returns = create_markov_chain(df_processed)

        if state_names is None or transition_matrix is None:
            st.warning("Could not create Markov Chain model due to insufficient or invalid data.")
        else:
            st.subheader("Transition Probabilities")
            transition_df = pl.DataFrame(
                transition_matrix,
                schema=[f"To {name}" for name in state_names]
            ).with_columns(
                pl.Series("From", state_names)
            ).select(["From"] + [f"To {name}" for name in state_names])
            
            st.dataframe(
                transition_df,
                use_container_width=True,
                column_config={
                    col: st.column_config.NumberColumn(format="%.3f") for col in transition_df.columns if col != "From"
                }
            )

            st.subheader("Next Day Prediction")
            latest_return = df_processed["Daily Return"][-1] * 100 if df_processed["Daily Return"][-1] is not None else None
            latest_price = df_processed["Adj Close"][-1] if df_processed["Adj Close"][-1] is not None else None
            latest_state = df_with_states["State"][-1] if not df_with_states.is_empty() else "Unknown"
            
            if latest_price is not None:
                st.write(f"Latest Adj Close: {latest_price:.2f}")
            if latest_return is not None:
                st.write(f"Latest Daily Return: {latest_return:.3f}% (State: {latest_state})")
            else:
                st.write(f"Latest Daily Return: Unknown (State: {latest_state})")
            
            if predicted_state and predicted_price is not None:
                st.write(f"Predicted State for Next Day: **{predicted_state}**")
                st.write(f"Predicted Adj Close for Next Day: **{predicted_price:.2f}**")
                # Display average returns per state and transition probabilities
                st.write("Average Returns by State:")
                for state, avg_ret in avg_returns.items():
                    st.write(f"{state}: {(avg_ret * 100):.3f}%")
                st.write("Transition Probabilities from Current State:")
                latest_state_idx = state_names.index(latest_state) if latest_state in state_names else None
                if latest_state_idx is not None:
                    probs = transition_matrix[latest_state_idx]
                    for state, prob in zip(state_names, probs):
                        st.write(f"Probability of {state}: {prob:.3f}")
            else:
                st.warning("Could not predict next day's state or price due to missing data.")

        # --- Section 5: Raw Data ---
        with st.expander("Show Raw Data"):
            st.dataframe(df_raw, use_container_width=True)