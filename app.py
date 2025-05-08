# import streamlit as st
# import polars as pl
# import plotly.express as px
# import plotly.graph_objects as go

# from data_manager import load_data, update_data_for_symbol # update_all_data (not used directly here)
# from calculator import (
#     calculate_daily_returns,
#     calculate_log_returns, 
#     add_time_periods,
#     calculate_period_stats,
#     assign_volatility_state,
#     calculate_transition_matrix,
#     calculate_avg_returns_per_state 
# )
# from config import INITIAL_SYMBOLS, ALL_SYMBOLS, PRICE_COLUMN

# # --- Page Config ---
# st.set_page_config(
#     page_title="kaNak Data Analysis",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # --- Helper Functions ---
# @st.cache_data(ttl=3600)
# def load_symbol_data(symbol_key):
#     ticker = ALL_SYMBOLS.get(symbol_key)
#     if not ticker:
#         st.error(f"Ticker for symbol key '{symbol_key}' not found in config.")
#         return None
#     df = load_data(symbol_key)
#     if df is None:
#          st.info(f"Local data for {symbol_key} not found. Fetching...")
#          df = update_data_for_symbol(symbol_key, ticker)
#     return df

# @st.cache_data(ttl=3600)
# def get_processed_data(df: pl.DataFrame):
#     """Calculates returns (arithmetic & log), assigns states, adds time periods."""
#     if df is None or df.is_empty() or df.height <= 1:
#         return None, None, None, None, None, None, None, None

#     df_arith_returns = calculate_daily_returns(df)
#     df_log_returns = calculate_log_returns(df_arith_returns) 

#     returns_series_arith = df_log_returns['Daily Return'].drop_nulls()
#     returns_series_log = df_log_returns['Log Return'].drop_nulls() 

#     if returns_series_arith.is_empty():
#          st.warning("No valid daily arithmetic returns could be calculated.")
#          return None, None, None, None, None, None, None, None

#     overall_mean_arith = returns_series_arith.mean()
#     overall_std_arith = returns_series_arith.std()
#     overall_mean_log = returns_series_log.mean() if not returns_series_log.is_empty() else None
#     overall_std_log = returns_series_log.std() if not returns_series_log.is_empty() else None

#     if overall_mean_arith is None or overall_std_arith is None or overall_std_arith == 0:
#          st.warning("Could not calculate overall mean/std for state assignment. Skipping state analysis.")
#          df_states = df_log_returns.with_columns(pl.lit(None).cast(pl.Utf8).alias("State"))
#     else:
#         df_states = assign_volatility_state(df_log_returns, overall_mean_arith, overall_std_arith)

#     df_processed = add_time_periods(df_states)

#     # Ensure 'Daily Return' and 'Log Return' are present for period stats
#     # --- CORRECTED LINE ---
#     base_df_for_stats = df_processed.drop_nulls(subset=["Daily Return", "Log Return"])

#     fy_stats = calculate_period_stats(base_df_for_stats, "Financial Year")
#     q_stats = calculate_period_stats(base_df_for_stats, "Quarter")
#     m_stats = calculate_period_stats(base_df_for_stats, "Month Year")

#     return df_processed, fy_stats, q_stats, m_stats, \
#            overall_mean_arith, overall_std_arith, \
#            overall_mean_log, overall_std_log


# @st.cache_data(ttl=3600)
# def get_markov_model(df_processed: pl.DataFrame):
#     if df_processed is None or "State" not in df_processed.columns or \
#        df_processed['State'].drop_nulls().is_empty() or \
#        "Daily Return" not in df_processed.columns or \
#        "Log Return" not in df_processed.columns: 
#         return None, None, None

#     trans_matrix, state_names = calculate_transition_matrix(df_processed)
#     avg_returns_df_state = calculate_avg_returns_per_state(df_processed)

#     return trans_matrix, avg_returns_df_state, state_names


# st.sidebar.title("📈 kaNak Analysis Tool")
# st.sidebar.markdown("Select analysis parameters.")

# available_symbols = list(ALL_SYMBOLS.keys())
# selected_symbol_key = st.sidebar.selectbox(
#     "Select Symbol:",
#     options=available_symbols,
#     index=available_symbols.index("NIFTY_50")
# )

# if st.sidebar.button(f"🔄 Update Data for {selected_symbol_key}"):
#     ticker = ALL_SYMBOLS.get(selected_symbol_key)
#     if ticker:
#         with st.spinner(f"Updating data for {selected_symbol_key}..."):
#             update_data_for_symbol(selected_symbol_key, ticker)
#         st.sidebar.success(f"Data updated for {selected_symbol_key}!")
#         st.cache_data.clear()
#     else:
#         st.sidebar.error("Symbol ticker not found.")

# st.title(f"📊 Return Analysis: {selected_symbol_key}")

# df_raw_loaded = load_symbol_data(selected_symbol_key) 

# if df_raw_loaded is None or df_raw_loaded.is_empty():
#     st.error(f"Could not load or fetch data for {selected_symbol_key}. Please check logs or try updating.")
# else:
#     df_processed, fy_stats, q_stats, m_stats, \
#     overall_mean_arith, overall_std_arith, \
#     overall_mean_log, overall_std_log = get_processed_data(df_raw_loaded)

#     if df_processed is None or df_processed.is_empty():
#         st.warning("Data loaded, but processing failed or resulted in empty data.")
#     else:
#         st.header("Return Distributions")
#         dist_col1, dist_col2 = st.columns(2)

#         with dist_col1:
#             st.subheader("Arithmetic Returns")
#             returns_series_arith_pct = df_processed['Daily Return'].drop_nulls() * 100

#             if not returns_series_arith_pct.is_empty() and overall_mean_arith is not None and overall_std_arith is not None:
#                 fig_hist_arith = px.histogram(
#                     returns_series_arith_pct.to_pandas(),
#                     nbins=100,
#                     title=f"Daily Arithmetic Returns (%)",
#                     labels={'value': 'Arithmetic Return (%)'}
#                 )
#                 fig_hist_arith.update_layout(showlegend=False)
#                 mean_arith_ret_pct = overall_mean_arith * 100
#                 std_dev_arith_ret_pct = overall_std_arith * 100
#                 fig_hist_arith.add_vline(x=mean_arith_ret_pct, line_dash="dash", line_color="red", annotation_text="Mean")
#                 for i in range(1, 4):
#                     fig_hist_arith.add_vline(x=mean_arith_ret_pct + i * std_dev_arith_ret_pct, line_dash="dot", line_color="green", annotation_text=f"+{i}σ")
#                     fig_hist_arith.add_vline(x=mean_arith_ret_pct - i * std_dev_arith_ret_pct, line_dash="dot", line_color="orange", annotation_text=f"-{i}σ")
#                 st.plotly_chart(fig_hist_arith, use_container_width=True)

#                 st.markdown("**Overall Arithmetic Stats (Full Period)**")
#                 overall_count_arith = len(returns_series_arith_pct)
#                 col_om_a, col_os_a, col_oc_a = st.columns(3)
#                 col_om_a.metric(label="Mean Arith. Return (%)", value=f"{mean_arith_ret_pct:.3f}")
#                 col_os_a.metric(label="Std Dev Arith. (%)", value=f"{std_dev_arith_ret_pct:.3f}")
#                 col_oc_a.metric(label="Trading Days", value=f"{overall_count_arith}")

#         with dist_col2:
#             st.subheader("Logarithmic Returns")
#             returns_series_log_pct = df_processed['Log Return'].drop_nulls() * 100 

#             if not returns_series_log_pct.is_empty() and overall_mean_log is not None and overall_std_log is not None:
#                 fig_hist_log = px.histogram(
#                     returns_series_log_pct.to_pandas(),
#                     nbins=100,
#                     title=f"Daily Log Returns (%)",
#                     labels={'value': 'Log Return (%)'}
#                 )
#                 fig_hist_log.update_layout(showlegend=False)
#                 mean_log_ret_pct = overall_mean_log * 100
#                 std_dev_log_ret_pct = overall_std_log * 100
#                 fig_hist_log.add_vline(x=mean_log_ret_pct, line_dash="dash", line_color="blue", annotation_text="Mean")
#                 for i in range(1, 4):
#                     fig_hist_log.add_vline(x=mean_log_ret_pct + i * std_dev_log_ret_pct, line_dash="dot", line_color="purple", annotation_text=f"+{i}σ")
#                     fig_hist_log.add_vline(x=mean_log_ret_pct - i * std_dev_log_ret_pct, line_dash="dot", line_color="brown", annotation_text=f"-{i}σ")
#                 st.plotly_chart(fig_hist_log, use_container_width=True)

#                 st.markdown("**Overall Log Stats (Full Period)**")
#                 overall_count_log = len(returns_series_log_pct)
#                 col_om_l, col_os_l, col_oc_l = st.columns(3)
#                 col_om_l.metric(label="Mean Log Return (%)", value=f"{mean_log_ret_pct:.3f}")
#                 col_os_l.metric(label="Std Dev Log (%)", value=f"{std_dev_log_ret_pct:.3f}")
#                 col_oc_l.metric(label="Trading Days", value=f"{overall_count_log}")
#             elif returns_series_log_pct.is_empty():
#                 st.warning("No log returns could be calculated to display distribution.")


#         st.header("Analysis by Period")
#         all_fy = ["All"] + sorted(df_processed['Financial Year'].drop_nulls().unique().to_list(), reverse=True)
#         all_q = ["All"] + sorted(df_processed['Quarter'].drop_nulls().unique().to_list(), reverse=True)
#         all_m = ["All"] + sorted(df_processed['Month Year'].drop_nulls().unique().to_list(), reverse=True)

#         col1, col2, col3 = st.columns(3)
#         selected_fy = col1.selectbox("Select Financial Year:", all_fy)
#         selected_q = col2.selectbox("Select Quarter:", all_q)
#         selected_m = col3.selectbox("Select Month:", all_m)

#         filtered_df = df_processed.clone()
#         if selected_fy != "All":
#             filtered_df = filtered_df.filter(pl.col("Financial Year") == selected_fy)
#         if selected_q != "All":
#             filtered_df = filtered_df.filter(pl.col("Quarter") == selected_q)
#         if selected_m != "All":
#             filtered_df = filtered_df.filter(pl.col("Month Year") == selected_m)

#         st.subheader("Statistics for Selected Period")
#         # --- CORRECTED LINE ---
#         if filtered_df.drop_nulls(subset=["Daily Return", "Log Return"]).is_empty(): 
#             st.warning("No data with returns available for the selected period.")
#         else:
#             period_mean_arith = filtered_df['Daily Return'].mean()
#             period_std_arith = filtered_df['Daily Return'].std()
#             period_mean_log = filtered_df['Log Return'].mean() 
#             period_std_log = filtered_df['Log Return'].std()   
#             period_count = filtered_df['Daily Return'].drop_nulls().count()

#             if period_mean_arith is not None and period_std_arith is not None:
#                 st.markdown("##### Arithmetic Returns")
#                 col_m_a, col_s_a, col_c_a = st.columns(3)
#                 col_m_a.metric("Mean Arith. Return (%)", f"{(period_mean_arith * 100):.3f}")
#                 col_s_a.metric("Std Dev Arith. (%)", f"{(period_std_arith * 100):.3f}")
#                 col_c_a.metric("Trading Days", f"{period_count}")
#                 st.markdown("**Standard Deviation Ranges (Arithmetic):**")
#                 st.text(f" Mean ± 1σ: [{(period_mean_arith - 1 * period_std_arith)*100:.3f}%, {(period_mean_arith + 1 * period_std_arith)*100:.3f}%]")
#                 st.text(f" Mean ± 2σ: [{(period_mean_arith - 2 * period_std_arith)*100:.3f}%, {(period_mean_arith + 2 * period_std_arith)*100:.3f}%]")
#                 st.text(f" Mean ± 3σ: [{(period_mean_arith - 3 * period_std_arith)*100:.3f}%, {(period_mean_arith + 3 * period_std_arith)*100:.3f}%]")

#             if period_mean_log is not None and period_std_log is not None: 
#                 st.markdown("##### Logarithmic Returns")
#                 col_m_l, col_s_l, col_c_l = st.columns(3)
#                 col_m_l.metric("Mean Log Return (%)", f"{(period_mean_log * 100):.3f}")
#                 col_s_l.metric("Std Dev Log (%)", f"{(period_std_log * 100):.3f}")
#                 col_c_l.metric("Trading Days", f"{filtered_df['Log Return'].drop_nulls().count()}") 
#                 st.markdown("**Standard Deviation Ranges (Logarithmic):**")
#                 st.text(f" Mean ± 1σ: [{(period_mean_log - 1 * period_std_log)*100:.3f}%, {(period_mean_log + 1 * period_std_log)*100:.3f}%]")
#                 st.text(f" Mean ± 2σ: [{(period_mean_log - 2 * period_std_log)*100:.3f}%, {(period_mean_log + 2 * period_std_log)*100:.3f}%]")
#                 st.text(f" Mean ± 3σ: [{(period_mean_log - 3 * period_std_log)*100:.3f}%, {(period_mean_log + 3 * period_std_log)*100:.3f}%]")
            
#             if (period_mean_arith is None or period_std_arith is None) and \
#                (period_mean_log is None or period_std_log is None):
#                  st.warning("Could not calculate statistics for the selected period (maybe too few data points).")

#             st.dataframe(filtered_df.select([
#                  "Date", PRICE_COLUMN, "Daily Return", "Log Return", "State", 
#                  "Financial Year", "Quarter", "Month Year"
#                  ]).sort("Date", descending=True), use_container_width=True, height=300)


#         st.header("📈 Markov Chain Prediction (Volatility States from Arithmetic Returns)")
#         trans_matrix, avg_returns_df_state, state_names = get_markov_model(df_processed)

#         if trans_matrix is None or avg_returns_df_state is None or state_names is None:
#             st.warning("Could not compute Markov model components. Check data and state assignments.")
#         else:
#             st.subheader("Transition Probability Matrix")
#             st.markdown("Rows: Current State, Columns: Next State Probability. Based on volatility states from ARITHMETIC returns.")
#             st.dataframe(
#                 trans_matrix.with_columns(
#                      pl.col(s).round(4) for s in state_names
#                 ).rename({'State': ''}),
#                 use_container_width=True
#             )

#             st.subheader("Average Returns per State")
#             avg_returns_display_df = avg_returns_df_state.clone() 
#             if "Avg Arithmetic Return" in avg_returns_display_df.columns:
#                 avg_returns_display_df = avg_returns_display_df.with_columns(
#                     (pl.col("Avg Arithmetic Return") * 100).round(3).alias("Avg Arith Return (%)")
#                 )
#             if "Avg Log Return" in avg_returns_display_df.columns:
#                  avg_returns_display_df = avg_returns_display_df.with_columns(
#                     (pl.col("Avg Log Return") * 100).round(3).alias("Avg Log Return (%)")
#                 )
#             st.dataframe(avg_returns_display_df, use_container_width=True)


#             st.subheader("Next Day Prediction")
#             latest_data = df_processed.drop_nulls("State").sort("Date", descending=True).head(1)

#             if latest_data.is_empty():
#                 st.error("Could not find the latest day's state for prediction.")
#             else:
#                 current_date = latest_data["Date"][0]
#                 current_state = latest_data["State"][0]
#                 current_price = latest_data[PRICE_COLUMN][0]

#                 st.write(f"Latest Data Point:")
#                 st.markdown(f"- **Date:** {current_date}")
#                 st.markdown(f"- **{PRICE_COLUMN}:** {current_price:.2f}")
#                 st.markdown(f"- **Current State (from Arith. Ret):** {current_state}")

#                 current_state_probs = trans_matrix.filter(pl.col("State") == current_state)

#                 if current_state_probs.is_empty():
#                     st.error(f"Could not find transition probabilities for the current state '{current_state}'.")
#                 else:
#                     expected_next_arith_return = 0.0
#                     next_state_prob_list = []
#                     avg_ret_col_for_pred = "Avg Arithmetic Return"

#                     if avg_ret_col_for_pred not in avg_returns_df_state.columns:
#                         st.error(f"'{avg_ret_col_for_pred}' not found in average returns per state. Cannot predict.")
#                     else:
#                         value_columns_to_unpivot = [col for col in current_state_probs.columns if col != 'State']
#                         if not value_columns_to_unpivot:
#                              st.error("No state probability columns found to unpivot.")
#                              prob_df = pl.DataFrame()
#                         else:
#                             prob_df = current_state_probs.unpivot(
#                                 index="State",
#                                 on=value_columns_to_unpivot,
#                                 variable_name="Next State",
#                                 value_name="Probability"
#                             )
                        
#                         expected_df = prob_df.join(
#                             avg_returns_df_state.rename({"State": "Next State"}), 
#                             on="Next State", 
#                             how="inner"
#                         )

#                         if not expected_df.is_empty():
#                              expected_next_arith_return = expected_df.select(
#                                   (pl.col("Probability") * pl.col(avg_ret_col_for_pred)).sum()
#                              ).item()
#                              next_state_prob_list = expected_df.select(["Next State", "Probability"]).sort("Next State").rows(named=True)

#                         predicted_price = current_price * (1 + expected_next_arith_return)

#                         st.metric(label="Predicted Next Trading Day State", value="Based on probabilities below")
#                         pred_cols = st.columns(2)
#                         with pred_cols[0]:
#                             st.metric(label="Expected Arith. Return (%)", value=f"{(expected_next_arith_return * 100):.3f}")
#                         with pred_cols[1]:
#                             st.metric(label=f"Predicted Next {PRICE_COLUMN}", value=f"{predicted_price:.2f}")

#                         st.write("Predicted Probabilities for Next State:")
#                         prob_display_df = pl.DataFrame(next_state_prob_list).with_columns(pl.col("Probability").round(4))
#                         st.dataframe(prob_display_df, use_container_width=True)

#                         fig_prob = px.bar(prob_display_df, x="Next State", y="Probability", title="Next State Probabilities")
#                         fig_prob.update_layout(yaxis_title="Probability")
#                         st.plotly_chart(fig_prob, use_container_width=True)

#         st.header("Ready Reckoners")
#         tab1, tab2, tab3 = st.tabs(["Financial Years", "Quarters", "Months"])
#         with tab1:
#             st.subheader("Stats per Financial Year")
#             if fy_stats is not None and not fy_stats.is_empty():
#                 st.dataframe(fy_stats, use_container_width=True)
#             else:
#                 st.warning("Could not calculate Financial Year stats or no data.")
#         with tab2:
#             st.subheader("Stats per Quarter")
#             if q_stats is not None and not q_stats.is_empty():
#                 st.dataframe(q_stats, use_container_width=True)
#             else:
#                 st.warning("Could not calculate Quarterly stats or no data.")
#         with tab3:
#              st.subheader("Stats per Month")
#              if m_stats is not None and not m_stats.is_empty():
#                   st.dataframe(m_stats, use_container_width=True)
#              else:
#                   st.warning("Could not calculate Monthly stats or no data.")

#         with st.expander("Show Processed Data with All Calculations"):
#             st.dataframe(df_processed.select([ 
#                 "Date", PRICE_COLUMN, "Daily Return", "Log Return", "State",
#                 "Financial Year", "Quarter", "Month Year"
#             ]).sort("Date", descending=True), use_container_width=True, height=500)

import streamlit as st
import polars as pl
import plotly.express as px
# import plotly.graph_objects as go # Not explicitly used in portfolio part yet
from datetime import datetime, date

# Make sure data_manager and calculator are correct and in the same directory orPYTHONPATH
from data_manager import load_data as dm_load_data, update_data_for_symbol as dm_update_data_for_symbol
from calculator import (
    calculate_daily_returns, calculate_log_returns, add_time_periods,
    calculate_period_stats, assign_volatility_state,
    calculate_transition_matrix, calculate_avg_returns_per_state
)
from config import ALL_SYMBOLS, PRICE_COLUMN # INITIAL_SYMBOLS not directly used if ALL_SYMBOLS covers it

import portfolio_manager as pm

# --- Page Config ---
st.set_page_config(
    page_title="kaNak Data Analysis & Portfolio",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Helper Functions ---
@st.cache_data(ttl=3600)
def load_symbol_data_app(symbol_key): # Renamed to avoid conflict if you have another load_data
    ticker = ALL_SYMBOLS.get(symbol_key)
    if not ticker:
        st.error(f"Ticker for symbol key '{symbol_key}' not found in config.")
        return None
    # Use the functions from data_manager.py
    df = dm_load_data(symbol_key)
    if df is None:
         st.info(f"Local data for {symbol_key} not found. Fetching from source...")
         df = dm_update_data_for_symbol(symbol_key, ticker)
    return df

@st.cache_data(ttl=3600)
def get_processed_data(df: pl.DataFrame):
    if df is None or df.is_empty() or df.height <= 1:
        return None, None, None, None, None, None, None, None
    df_arith_returns = calculate_daily_returns(df)
    df_log_returns = calculate_log_returns(df_arith_returns)
    returns_series_arith = df_log_returns['Daily Return'].drop_nulls()
    returns_series_log = df_log_returns['Log Return'].drop_nulls()
    if returns_series_arith.is_empty():
         return None, None, None, None, None, None, None, None
    overall_mean_arith = returns_series_arith.mean()
    overall_std_arith = returns_series_arith.std()
    overall_mean_log = returns_series_log.mean() if not returns_series_log.is_empty() else None
    overall_std_log = returns_series_log.std() if not returns_series_log.is_empty() else None
    if overall_mean_arith is None or overall_std_arith is None or overall_std_arith == 0:
         df_states = df_log_returns.with_columns(pl.lit(None).cast(pl.Utf8).alias("State"))
    else:
        df_states = assign_volatility_state(df_log_returns, overall_mean_arith, overall_std_arith)
    df_processed = add_time_periods(df_states)
    base_df_for_stats = df_processed.drop_nulls(subset=["Daily Return", "Log Return"])
    fy_stats = calculate_period_stats(base_df_for_stats, "Financial Year")
    q_stats = calculate_period_stats(base_df_for_stats, "Quarter")
    m_stats = calculate_period_stats(base_df_for_stats, "Month Year")
    return df_processed, fy_stats, q_stats, m_stats, \
           overall_mean_arith, overall_std_arith, \
           overall_mean_log, overall_std_log

@st.cache_data(ttl=3600)
def get_markov_model(df_processed: pl.DataFrame):
    if df_processed is None or "State" not in df_processed.columns or \
       df_processed['State'].drop_nulls().is_empty() or \
       "Daily Return" not in df_processed.columns or \
       "Log Return" not in df_processed.columns:
        return None, None, None
    trans_matrix, state_names = calculate_transition_matrix(df_processed)
    avg_returns_df_state = calculate_avg_returns_per_state(df_processed)
    return trans_matrix, avg_returns_df_state, state_names

# --- APP NAVIGATION ---
st.sidebar.title("📈 kaNak Analysis & Portfolio")
analysis_mode = st.sidebar.radio(
    "Select Mode:",
    ("Individual Symbol Analysis", "Portfolio Management"),
    key="main_mode_radio" # Add key for robust state
)

# ================= INDIVIDUAL SYMBOL ANALYSIS =================
if analysis_mode == "Individual Symbol Analysis":
    st.sidebar.markdown("---")
    st.sidebar.header("Symbol Analysis Settings")
    # Use a filtered list if INITIAL_SYMBOLS is different from ALL_SYMBOLS for this section
    available_symbols_analysis = list(ALL_SYMBOLS.keys()) # Or list(INITIAL_SYMBOLS.keys())
    default_symbol = "NIFTY_50" if "NIFTY_50" in available_symbols_analysis else available_symbols_analysis[0]
    
    selected_symbol_key_analysis = st.sidebar.selectbox(
        "Select Symbol for Analysis:",
        options=available_symbols_analysis,
        index=available_symbols_analysis.index(default_symbol) if default_symbol in available_symbols_analysis else 0,
        key="symbol_analysis_selectbox"
    )

    if st.sidebar.button(f"🔄 Update Data for {selected_symbol_key_analysis}", key="update_symbol_analysis"):
        ticker = ALL_SYMBOLS.get(selected_symbol_key_analysis)
        if ticker:
            with st.spinner(f"Updating data for {selected_symbol_key_analysis}..."):
                dm_update_data_for_symbol(selected_symbol_key_analysis, ticker)
            st.sidebar.success(f"Data updated for {selected_symbol_key_analysis}!")
            st.cache_data.clear() # Clear all Streamlit cache
        else:
            st.sidebar.error("Symbol ticker not found.")

    st.title(f"📊 Return Analysis: {selected_symbol_key_analysis}")
    df_raw_loaded = load_symbol_data_app(selected_symbol_key_analysis)

    if df_raw_loaded is None or df_raw_loaded.is_empty():
        st.error(f"Could not load or fetch data for {selected_symbol_key_analysis}.")
    else:
        df_processed, fy_stats, q_stats, m_stats, \
        overall_mean_arith, overall_std_arith, \
        overall_mean_log, overall_std_log = get_processed_data(df_raw_loaded)

        if df_processed is None or df_processed.is_empty():
            st.warning("Data loaded, but processing failed or resulted in empty data.")
        else:
            st.header("Return Distributions")
            dist_col1, dist_col2 = st.columns(2)
            with dist_col1:
                st.subheader("Arithmetic Returns")
                returns_series_arith_pct = df_processed['Daily Return'].drop_nulls() * 100
                if not returns_series_arith_pct.is_empty() and overall_mean_arith is not None and overall_std_arith is not None:
                    fig_hist_arith = px.histogram(returns_series_arith_pct.to_pandas(), nbins=100, title=f"Daily Arithmetic Returns (%)", labels={'value': 'Arithmetic Return (%)'})
                    fig_hist_arith.update_layout(showlegend=False)
                    mean_arith_ret_pct = overall_mean_arith * 100
                    std_dev_arith_ret_pct = overall_std_arith * 100
                    fig_hist_arith.add_vline(x=mean_arith_ret_pct, line_dash="dash", line_color="red", annotation_text="Mean")
                    for i in range(1, 4):
                        fig_hist_arith.add_vline(x=mean_arith_ret_pct + i * std_dev_arith_ret_pct, line_dash="dot", line_color="green", annotation_text=f"+{i}σ")
                        fig_hist_arith.add_vline(x=mean_arith_ret_pct - i * std_dev_arith_ret_pct, line_dash="dot", line_color="orange", annotation_text=f"-{i}σ")
                    st.plotly_chart(fig_hist_arith, use_container_width=True)
                    st.markdown("**Overall Arithmetic Stats (Full Period)**")
                    col_om_a, col_os_a, col_oc_a = st.columns(3)
                    col_om_a.metric(label="Mean Arith. Return (%)", value=f"{mean_arith_ret_pct:.3f}")
                    col_os_a.metric(label="Std Dev Arith. (%)", value=f"{std_dev_arith_ret_pct:.3f}")
                    col_oc_a.metric(label="Trading Days", value=f"{len(returns_series_arith_pct)}")
            with dist_col2:
                st.subheader("Logarithmic Returns")
                returns_series_log_pct = df_processed['Log Return'].drop_nulls() * 100
                if not returns_series_log_pct.is_empty() and overall_mean_log is not None and overall_std_log is not None:
                    fig_hist_log = px.histogram(returns_series_log_pct.to_pandas(), nbins=100, title=f"Daily Log Returns (%)", labels={'value': 'Log Return (%)'})
                    fig_hist_log.update_layout(showlegend=False)
                    mean_log_ret_pct = overall_mean_log * 100
                    std_dev_log_ret_pct = overall_std_log * 100
                    fig_hist_log.add_vline(x=mean_log_ret_pct, line_dash="dash", line_color="blue", annotation_text="Mean")
                    for i in range(1, 4):
                        fig_hist_log.add_vline(x=mean_log_ret_pct + i * std_dev_log_ret_pct, line_dash="dot", line_color="purple", annotation_text=f"+{i}σ")
                        fig_hist_log.add_vline(x=mean_log_ret_pct - i * std_dev_log_ret_pct, line_dash="dot", line_color="brown", annotation_text=f"-{i}σ")
                    st.plotly_chart(fig_hist_log, use_container_width=True)
                    st.markdown("**Overall Log Stats (Full Period)**")
                    col_om_l, col_os_l, col_oc_l = st.columns(3)
                    col_om_l.metric(label="Mean Log Return (%)", value=f"{mean_log_ret_pct:.3f}")
                    col_os_l.metric(label="Std Dev Log (%)", value=f"{std_dev_log_ret_pct:.3f}")
                    col_oc_l.metric(label="Trading Days", value=f"{len(returns_series_log_pct)}")
                elif returns_series_log_pct.is_empty():
                    st.warning("No log returns for distribution.")
            
            st.header("Analysis by Period")
            all_fy = ["All"] + sorted(df_processed['Financial Year'].drop_nulls().unique().to_list(), reverse=True)
            all_q = ["All"] + sorted(df_processed['Quarter'].drop_nulls().unique().to_list(), reverse=True)
            all_m = ["All"] + sorted(df_processed['Month Year'].drop_nulls().unique().to_list(), reverse=True)
            col1, col2, col3 = st.columns(3)
            selected_fy = col1.selectbox("Select Financial Year:", all_fy, key="fy_select_analysis")
            selected_q = col2.selectbox("Select Quarter:", all_q, key="q_select_analysis")
            selected_m = col3.selectbox("Select Month:", all_m, key="m_select_analysis")
            filtered_df = df_processed.clone()
            if selected_fy != "All": filtered_df = filtered_df.filter(pl.col("Financial Year") == selected_fy)
            if selected_q != "All": filtered_df = filtered_df.filter(pl.col("Quarter") == selected_q)
            if selected_m != "All": filtered_df = filtered_df.filter(pl.col("Month Year") == selected_m)
            st.subheader("Statistics for Selected Period")
            if filtered_df.drop_nulls(subset=["Daily Return", "Log Return"]).is_empty():
                st.warning("No data with returns for selected period.")
            else:
                period_mean_arith = filtered_df['Daily Return'].mean()
                period_std_arith = filtered_df['Daily Return'].std()
                period_mean_log = filtered_df['Log Return'].mean()
                period_std_log = filtered_df['Log Return'].std()
                period_count_arith = filtered_df['Daily Return'].drop_nulls().count()
                period_count_log = filtered_df['Log Return'].drop_nulls().count()
                if period_mean_arith is not None and period_std_arith is not None:
                    st.markdown("##### Arithmetic Returns")
                    col_m_a, col_s_a, col_c_a = st.columns(3)
                    col_m_a.metric("Mean Arith. Return (%)", f"{(period_mean_arith * 100):.3f}")
                    col_s_a.metric("Std Dev Arith. (%)", f"{(period_std_arith * 100):.3f}")
                    col_c_a.metric("Trading Days", f"{period_count_arith}")
                    if period_std_arith > 0:
                        st.markdown("**Standard Deviation Ranges (Arithmetic):**")
                        for i in range(1, 4): st.text(f" Mean ± {i}σ: [{(period_mean_arith - i * period_std_arith)*100:.3f}%, {(period_mean_arith + i * period_std_arith)*100:.3f}%]")
                    else: st.warning("Std dev for arithmetic returns is zero for this period.")
                if period_mean_log is not None and period_std_log is not None:
                    st.markdown("##### Logarithmic Returns")
                    col_m_l, col_s_l, col_c_l = st.columns(3)
                    col_m_l.metric("Mean Log Return (%)", f"{(period_mean_log * 100):.3f}")
                    col_s_l.metric("Std Dev Log (%)", f"{(period_std_log * 100):.3f}")
                    col_c_l.metric("Trading Days", f"{period_count_log}")
                    if period_std_log > 0:
                        st.markdown("**Standard Deviation Ranges (Logarithmic):**")
                        for i in range(1, 4): st.text(f" Mean ± {i}σ: [{(period_mean_log - i * period_std_log)*100:.3f}%, {(period_mean_log + i * period_std_log)*100:.3f}%]")
                    else: st.warning("Std dev for log returns is zero for this period.")
                if (period_mean_arith is None or period_std_arith is None) and (period_mean_log is None or period_std_log is None):
                     st.warning("Could not calculate stats for the selected period.")
                st.dataframe(filtered_df.select(["Date", PRICE_COLUMN, "Daily Return", "Log Return", "State", "Financial Year", "Quarter", "Month Year"]).sort("Date", descending=True), height=300, use_container_width=True)

            st.header("📈 Markov Chain Prediction (Volatility States from Arithmetic Returns)")
            trans_matrix, avg_returns_df_state, state_names = get_markov_model(df_processed)
            if trans_matrix is None or avg_returns_df_state is None or state_names is None:
                st.warning("Could not compute Markov model components.")
            else:
                st.subheader("Transition Probability Matrix")
                st.dataframe(trans_matrix.with_columns(pl.col(s).round(4) for s in state_names).rename({'State': ''}), use_container_width=True)
                st.subheader("Average Returns per State")
                avg_returns_display_df = avg_returns_df_state.clone()
                if "Avg Arithmetic Return" in avg_returns_display_df.columns: avg_returns_display_df = avg_returns_display_df.with_columns((pl.col("Avg Arithmetic Return") * 100).round(3).alias("Avg Arith Return (%)"))
                if "Avg Log Return" in avg_returns_display_df.columns: avg_returns_display_df = avg_returns_display_df.with_columns((pl.col("Avg Log Return") * 100).round(3).alias("Avg Log Return (%)"))
                st.dataframe(avg_returns_display_df, use_container_width=True)
                st.subheader("Next Day Prediction")
                latest_data = df_processed.drop_nulls("State").sort("Date", descending=True).head(1)
                if latest_data.is_empty(): st.error("No latest day's state for prediction.")
                else:
                    current_date_pred, current_state_pred, current_price_pred = latest_data["Date"][0], latest_data["State"][0], latest_data[PRICE_COLUMN][0]
                    st.write(f"Latest Data Point: Date: {current_date_pred}, {PRICE_COLUMN}: {current_price_pred:.2f}, Current State: {current_state_pred}")
                    current_state_probs = trans_matrix.filter(pl.col("State") == current_state_pred)
                    if current_state_probs.is_empty(): st.error(f"No transition probabilities for state '{current_state_pred}'.")
                    else:
                        avg_ret_col_for_pred = "Avg Arithmetic Return"
                        if avg_ret_col_for_pred not in avg_returns_df_state.columns: st.error(f"'{avg_ret_col_for_pred}' missing for prediction.")
                        else:
                            value_cols_unpivot = [col for col in current_state_probs.columns if col != 'State']
                            if not value_cols_unpivot: prob_df = pl.DataFrame()
                            else: prob_df = current_state_probs.unpivot(index="State", on=value_cols_unpivot, variable_name="Next State", value_name="Probability")
                            expected_df = prob_df.join(avg_returns_df_state.rename({"State": "Next State"}), on="Next State", how="inner")
                            if not expected_df.is_empty():
                                expected_next_arith_return = expected_df.select((pl.col("Probability") * pl.col(avg_ret_col_for_pred)).sum()).item()
                                next_state_prob_list = expected_df.select(["Next State", "Probability"]).sort("Next State").rows(named=True)
                                predicted_price = current_price_pred * (1 + expected_next_arith_return)
                                st.metric(label="Predicted Next Trading Day State", value="Based on probabilities below")
                                pred_cols = st.columns(2)
                                pred_cols[0].metric(label="Expected Arith. Return (%)", value=f"{(expected_next_arith_return * 100):.3f}")
                                pred_cols[1].metric(label=f"Predicted Next {PRICE_COLUMN}", value=f"{predicted_price:.2f}")
                                prob_display_df = pl.DataFrame(next_state_prob_list).with_columns(pl.col("Probability").round(4))
                                st.dataframe(prob_display_df, use_container_width=True)
                                fig_prob = px.bar(prob_display_df, x="Next State", y="Probability", title="Next State Probabilities")
                                st.plotly_chart(fig_prob, use_container_width=True)
                            else: st.warning("Could not compute expected return for prediction.")
            st.header("Ready Reckoners")
            tab1, tab2, tab3 = st.tabs(["Financial Years", "Quarters", "Months"])
            with tab1: st.subheader("Stats per Financial Year"); st.dataframe(fy_stats, use_container_width=True) if fy_stats is not None and not fy_stats.is_empty() else st.warning("No FY stats.")
            with tab2: st.subheader("Stats per Quarter"); st.dataframe(q_stats, use_container_width=True) if q_stats is not None and not q_stats.is_empty() else st.warning("No Q stats.")
            with tab3: st.subheader("Stats per Month"); st.dataframe(m_stats, use_container_width=True) if m_stats is not None and not m_stats.is_empty() else st.warning("No M stats.")
            with st.expander("Show Processed Data with All Calculations"):
                st.dataframe(df_processed.select(["Date", PRICE_COLUMN, "Daily Return", "Log Return", "State", "Financial Year", "Quarter", "Month Year"]).sort("Date", descending=True), height=500, use_container_width=True)

# ================= PORTFOLIO MANAGEMENT =================
# ... (previous parts of app.py) ...

# ================= PORTFOLIO MANAGEMENT =================
elif analysis_mode == "Portfolio Management":
    st.sidebar.markdown("---")
    st.sidebar.header("Portfolio Actions")
    portfolio_action = st.sidebar.radio(
        "Action:",
        ("View Portfolio", "Create New Portfolio"),
        key="portfolio_action_radio"
    )
    st.title("💰 Portfolio Management")

    if portfolio_action == "Create New Portfolio":
        st.subheader("📝 Create a New Portfolio")

        # Initialize session state for the portfolio name input if not present
        if 'pf_current_portfolio_name' not in st.session_state:
            st.session_state.pf_current_portfolio_name = ""

        # Portfolio name input is now controlled by 'pf_current_portfolio_name'
        st.session_state.pf_current_portfolio_name = st.text_input(
            "Portfolio Name:",
            value=st.session_state.pf_current_portfolio_name, # Use the session state variable as the value
            placeholder="e.g., My Tech Stocks",
            key="pf_name_controlled_input" # Can use a different key if needed, or reuse if logic is simple
        )

        st.markdown("**Add Assets to List:**")
        if 'assets_to_add' not in st.session_state:
            st.session_state.assets_to_add = []

        cols_asset_input = st.columns([3, 2, 2, 1])
        # Give unique keys to these inputs as well if they aren't already unique across the app
        asset_symbol_key = cols_asset_input[0].selectbox("Symbol", options=[""] + list(ALL_SYMBOLS.keys()), key="pf_asset_symbol_select_create")
        asset_quantity = cols_asset_input[1].number_input("Quantity", min_value=0.00001, step=0.01, format="%.5f", key="pf_asset_qty_input_create")
        asset_purchase_date = cols_asset_input[2].date_input("Purchase Date", value=date.today(), max_value=date.today(), key="pf_asset_date_input_create")

        if cols_asset_input[3].button("➕ Add to List", key="pf_add_asset_to_list_btn_create"):
            # Access widget values through their keys stored in st.session_state
            s_key = st.session_state.pf_asset_symbol_select_create
            s_qty = st.session_state.pf_asset_qty_input_create
            s_date = st.session_state.pf_asset_date_input_create
            if s_key and s_qty > 0:
                st.session_state.assets_to_add.append({
                    "symbol_key": s_key,
                    "quantity": s_qty,
                    "purchase_date": s_date.strftime("%Y-%m-%d")
                })
            else:
                st.warning("Please select a symbol and enter a valid quantity to add to the list.")

        if st.session_state.assets_to_add:
            st.markdown("**Assets currently in list for the new portfolio:**")
            temp_asset_df = pl.DataFrame(st.session_state.assets_to_add)
            st.dataframe(temp_asset_df, use_container_width=True)
            if st.button("🗑️ Clear Asset List", key="pf_clear_assets_list_btn_create"):
                st.session_state.assets_to_add = []
                st.rerun()

        with st.form("create_portfolio_submit_form"):
            st.markdown("---")
            submitted = st.form_submit_button("💾 Create Portfolio with Listed Assets")

            if submitted:
                # Get the portfolio name from the session state variable that controls the input
                final_portfolio_name = st.session_state.pf_current_portfolio_name.strip()

                if not final_portfolio_name:
                    st.error("Portfolio name cannot be empty.")
                elif not st.session_state.assets_to_add:
                    st.error("Please add at least one asset to the list above before creating the portfolio.")
                else:
                    existing_names = pm.get_all_portfolio_names()
                    if final_portfolio_name in existing_names:
                        st.error(f"Portfolio name '{final_portfolio_name}' already exists. Please choose a different name.")
                    else:
                        with st.spinner(f"Creating portfolio '{final_portfolio_name}'..."):
                            portfolio_id = pm.add_portfolio(final_portfolio_name)
                            if portfolio_id:
                                success_count = 0
                                error_assets = []
                                for asset in st.session_state.assets_to_add:
                                    price = pm.get_purchase_price_on_date(asset["symbol_key"], asset["purchase_date"], PRICE_COLUMN)
                                    if price is not None:
                                        pm.add_asset_to_portfolio(
                                            portfolio_id,
                                            asset["symbol_key"],
                                            asset["quantity"],
                                            price,
                                            asset["purchase_date"]
                                        )
                                        success_count += 1
                                    else:
                                        st.warning(f"Could not fetch purchase price for {asset['symbol_key']} on {asset['purchase_date']}. Asset not added.")
                                        error_assets.append(asset['symbol_key'])
                                
                                if success_count > 0 :
                                    st.success(f"Portfolio '{final_portfolio_name}' created with {success_count} asset(s)!")
                                    if error_assets:
                                        st.warning(f"Failed to add: {', '.join(error_assets)} due to missing price data.")
                                    
                                    # --- CORRECT WAY TO CLEAR ---
                                    st.session_state.assets_to_add = [] 
                                    st.session_state.pf_current_portfolio_name = "" # Clear the controlling session state variable
                                    # The st.text_input will now use this empty string as its value on the next rerun
                                    
                                    st.rerun() 
                                elif not error_assets :
                                     st.info("No assets were added. The list might have been empty or all price fetches failed.")
                                else: 
                                    st.error(f"Failed to add any assets to '{final_portfolio_name}'. Check price data for {', '.join(error_assets)}.")
                            else:
                                st.error(f"Failed to create portfolio entry for '{final_portfolio_name}'. It might already exist or another DB error occurred.")
            
    elif portfolio_action == "View Portfolio":
        st.subheader("📂 View Portfolio Performance")
        all_portfolios = pm.get_all_portfolio_names()
        if not all_portfolios:
            st.info("No portfolios created yet. Go to 'Create New Portfolio'.")
        else:
            selected_portfolio_name = st.selectbox("Select Portfolio:", options=all_portfolios, key="select_portfolio_view_dd")
            if selected_portfolio_name:
                portfolio_assets_df = pm.get_portfolio_assets(selected_portfolio_name)
                if portfolio_assets_df is None or portfolio_assets_df.is_empty():
                    st.warning(f"No assets in '{selected_portfolio_name}' or portfolio error.")
                else:
                    st.markdown(f"### Assets in {selected_portfolio_name}")
                    portfolio_assets_df = portfolio_assets_df.with_columns((pl.col("quantity") * pl.col("purchase_price")).alias("Initial Investment"))
                    
                    current_prices_dict = {}
                    with st.spinner("Fetching current market prices..."):
                        unique_symbols = portfolio_assets_df["symbol_key"].unique().to_list()
                        for symbol_key_pf in unique_symbols: 
                            symbol_df_latest = load_symbol_data_app(symbol_key_pf) 
                            if symbol_df_latest is not None and not symbol_df_latest.is_empty():
                                latest_price = symbol_df_latest.sort("Date", descending=True).head(1)[PRICE_COLUMN][0]
                                current_prices_dict[symbol_key_pf] = latest_price if latest_price is not None else 0.0 
                            else:
                                current_prices_dict[symbol_key_pf] = 0.0 
                                st.warning(f"Could not fetch current price for {symbol_key_pf}, using 0.")
                    
                    # --- CORRECTED LINE USING map_elements ---
                    portfolio_assets_df = portfolio_assets_df.with_columns(
                        pl.col("symbol_key").map_elements(
                            lambda sk: current_prices_dict.get(sk, 0.0), # Use .get for safety
                            return_dtype=pl.Float64
                        ).alias("Current Price")
                    )
                    # --- END CORRECTION ---
                                        
                    portfolio_assets_df = portfolio_assets_df.with_columns(
                        (pl.col("quantity") * pl.col("Current Price")).alias("Current Value")
                    )
                    portfolio_assets_df = portfolio_assets_df.with_columns(
                        (pl.col("Current Value") - pl.col("Initial Investment")).alias("P&L")
                    )
                    portfolio_assets_df = portfolio_assets_df.with_columns(
                        pl.when(pl.col("Initial Investment") != 0)
                        .then((pl.col("P&L") / pl.col("Initial Investment") * 100))
                        .otherwise(None) 
                        .round(2).alias("P&L (%)")
                    )
                    
                    st.dataframe(portfolio_assets_df.select([
                        "symbol_key", "quantity", "purchase_date", 
                        pl.col("purchase_price").round(2), 
                        pl.col("Initial Investment").round(2), 
                        pl.col("Current Price").round(2), 
                        pl.col("Current Value").round(2),
                        pl.col("P&L").round(2), 
                        "P&L (%)"
                    ]), use_container_width=True)

                    total_initial_investment = portfolio_assets_df["Initial Investment"].sum()
                    total_current_value = portfolio_assets_df["Current Value"].sum() # Assumes Current Price is 0.0 if not found
                    
                    st.markdown("### Portfolio Summary")
                    summary_cols = st.columns(3)
                    summary_cols[0].metric("Total Initial Investment", f"₹{total_initial_investment:,.2f}")
                    summary_cols[1].metric("Total Current Value", f"₹{total_current_value:,.2f}")
                    if total_initial_investment != 0:
                        overall_pl = total_current_value - total_initial_investment
                        overall_pl_percent = (overall_pl / total_initial_investment) * 100
                        summary_cols[2].metric("Overall P&L", f"₹{overall_pl:,.2f} ({overall_pl_percent:.2f}%)")
                    else:
                        summary_cols[2].metric("Overall P&L", "₹0.00 (N/A)")