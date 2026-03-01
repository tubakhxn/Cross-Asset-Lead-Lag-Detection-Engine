import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.express as px
import yfinance as yf
from scipy.stats import zscore
from statsmodels.tsa.stattools import grangercausalitytests
from scipy.spatial.distance import cdist
from scipy.signal import correlate
import itertools
import io

# =====================
# THEME & STYLES
# =====================
QUANT_DARK_BG = '#181A20'
QUANT_NEON = '#00FFC6'
QUANT_ACCENT = '#FF00A6'
QUANT_FONT = 'Montserrat, sans-serif'

st.set_page_config(
    page_title="Cross-Asset Lead-Lag Detection Engine",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(f"""
    <style>
    body {{ background-color: {QUANT_DARK_BG}; }}
    .reportview-container {{ background: {QUANT_DARK_BG}; }}
    .sidebar .sidebar-content {{ background: {QUANT_DARK_BG}; }}
    h1, h2, h3, h4, h5, h6 {{ font-family: {QUANT_FONT}; color: white; }}
    .stApp {{ background: {QUANT_DARK_BG}; }}
    .stMarkdown, .stTextInput, .stSelectbox, .stSlider, .stButton {{ color: white; }}
    .stMetric {{ background: {QUANT_DARK_BG}; color: {QUANT_NEON}; border-radius: 10px; }}
    </style>
""", unsafe_allow_html=True)

# =====================
# SIDEBAR
# =====================
st.sidebar.title("Asset & Analysis Controls")
mode = st.sidebar.selectbox("Data Source", ["Upload CSV", "Download from Yahoo Finance"])

if mode == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload CSV (Date, Asset1, Asset2, ...)", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file, index_col=0, parse_dates=True)
        assets = list(df.columns)
else:
    tickers = st.sidebar.text_input("Tickers (comma separated)", value="AAPL,MSFT,GOOG,AMZN")
    start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2022-01-01"))
    end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("2023-01-01"))
    if st.sidebar.button("Download Data"):
        tickers_list = [t.strip() for t in tickers.split(",")]
        raw_df = yf.download(tickers_list, start=start_date, end=end_date)
        # Handle both single and multi-index columns
        if isinstance(raw_df.columns, pd.MultiIndex):
            if "Adj Close" in raw_df.columns.levels[0]:
                df = raw_df["Adj Close"].dropna()
            else:
                # Fallback: use 'Close' if 'Adj Close' not present
                df = raw_df["Close"].dropna()
        else:
            # Single-index: fallback to 'Close' if 'Adj Close' not present
            if "Adj Close" in raw_df.columns:
                df = raw_df["Adj Close"].dropna()
            elif "Close" in raw_df.columns:
                df = raw_df["Close"].dropna()
            else:
                st.error("No 'Adj Close' or 'Close' columns found in downloaded data.")
                df = pd.DataFrame()
        assets = list(df.columns) if not df.empty else []

window = st.sidebar.slider("Rolling Window Length (days)", min_value=10, max_value=120, value=30, step=5)
max_lag = st.sidebar.slider("Max Lag (days)", min_value=1, max_value=20, value=5, step=1)
sig_thresh = st.sidebar.slider("Granger Significance Threshold", min_value=0.001, max_value=0.1, value=0.05, step=0.001)
dark_mode = st.sidebar.selectbox("Theme", ["Dark", "Extra-Dark"])

# =====================
# DATA PREP
# =====================
def compute_log_returns(df):
    return np.log(df / df.shift(1)).dropna()

def rolling_standardized_returns(returns, window):
    return returns.rolling(window).apply(lambda x: zscore(x) if len(x) > 1 else np.zeros_like(x), raw=False)

def rolling_cross_corr(returns, window, max_lag):
    asset_pairs = list(itertools.combinations(returns.columns, 2))
    time_idx = returns.index[window:]
    corr_matrices = []
    for t in range(window, len(returns)):
        mat = np.zeros((len(asset_pairs), max_lag*2+1))
        for i, (a1, a2) in enumerate(asset_pairs):
            s1 = returns[a1].iloc[t-window:t].values
            s2 = returns[a2].iloc[t-window:t].values
            for lag in range(-max_lag, max_lag+1):
                if lag < 0:
                    mat[i, lag+max_lag] = np.corrcoef(s1[-lag:], s2[:lag])[0,1]
                elif lag > 0:
                    mat[i, lag+max_lag] = np.corrcoef(s1[:-lag], s2[lag:])[0,1]
                else:
                    mat[i, lag+max_lag] = np.corrcoef(s1, s2)[0,1]
        corr_matrices.append(mat)
    return np.array(corr_matrices), asset_pairs, time_idx

def rolling_granger(returns, window, max_lag, sig_thresh):
    asset_pairs = list(itertools.combinations(returns.columns, 2))
    time_idx = returns.index[window:]
    pval_matrices = []
    for t in range(window, len(returns)):
        mat = np.zeros((len(asset_pairs), max_lag))
        for i, (a1, a2) in enumerate(asset_pairs):
            s1 = returns[a1].iloc[t-window:t].values
            s2 = returns[a2].iloc[t-window:t].values
            for lag in range(1, max_lag+1):
                try:
                    res = grangercausalitytests(np.column_stack([s1, s2]), maxlag=lag, verbose=False)
                    pval = res[lag][0]['ssr_ftest'][1]
                except Exception:
                    pval = 1.0
                mat[i, lag-1] = pval
        pval_matrices.append(mat)
    return np.array(pval_matrices), asset_pairs, time_idx

def rolling_dtw(returns, window):
    asset_pairs = list(itertools.combinations(returns.columns, 2))
    time_idx = returns.index[window:]
    dtw_matrices = []
    for t in range(window, len(returns)):
        mat = np.zeros(len(asset_pairs))
        for i, (a1, a2) in enumerate(asset_pairs):
            s1 = returns[a1].iloc[t-window:t].values
            s2 = returns[a2].iloc[t-window:t].values
            dist = np.linalg.norm(s1 - s2)
            mat[i] = dist
        dtw_matrices.append(mat)
    dtw_arr = np.array(dtw_matrices)
    # Normalize distances to similarity scores
    sim_arr = 1 - (dtw_arr - dtw_arr.min()) / (dtw_arr.max() - dtw_arr.min() + 1e-8)
    return sim_arr, asset_pairs, time_idx

def lead_lag_strength_index(corrs, grangers, dtws):
    # Composite metric: mean of normalized correlation, inverse p-value, and dtw similarity
    corrs_norm = (np.abs(corrs).mean(axis=2) - np.abs(corrs).min()) / (np.abs(corrs).max() - np.abs(corrs).min() + 1e-8)
    granger_inv = 1 - grangers.mean(axis=2)
    dtw_sim = dtws
    composite = (corrs_norm + granger_inv + dtw_sim) / 3
    return composite

def dominant_lag_histogram(corrs, asset_pairs, max_lag):
    dom_lags = np.argmax(np.abs(corrs), axis=2) - max_lag
    hist = {}
    for i, pair in enumerate(asset_pairs):
        vals = dom_lags[:,i]
        hist[pair] = np.histogram(vals, bins=np.arange(-max_lag-1, max_lag+2))[0]
    return hist

# =====================
# MAIN LOGIC
# =====================
st.title("Cross-Asset Lead-Lag Detection Engine")
st.markdown("""
    <h2 style='color:#00FFC6;font-size:2.5em;font-family:Montserrat,sans-serif;'>Dynamic Lead-Lag & Causality Analysis</h2>
    <hr style='border:1px solid #FF00A6;'>
""", unsafe_allow_html=True)


# --- DEMO DATA FOR 3D GRAPH IF NO DATA LOADED ---
if 'df' in locals() and not df.empty:
    returns = compute_log_returns(df)
    std_returns = rolling_standardized_returns(returns, window)
    corrs, asset_pairs, time_idx = rolling_cross_corr(std_returns, window, max_lag)
    grangers, _, _ = rolling_granger(std_returns, window, max_lag, sig_thresh)
    dtws, _, _ = rolling_dtw(std_returns, window)
    strength = lead_lag_strength_index(corrs, grangers, dtws)
    dom_lag_hist = dominant_lag_histogram(corrs, asset_pairs, max_lag)

    # ...existing code for metrics and plots...

    st.subheader("3D Lead-Lag Strength Surface")
    fig3d = go.Figure(data=[go.Surface(
        z=strength.T,
        x=np.arange(len(time_idx)),
        y=np.arange(len(asset_pairs)),
        colorscale=[[0, QUANT_DARK_BG], [0.5, QUANT_NEON], [1, QUANT_ACCENT]],
        showscale=True
    )])
    fig3d.update_layout(
        template="plotly_dark",
        width=1200,
        height=600,
        scene=dict(
            xaxis_title="Time",
            yaxis_title="Asset Pair",
            zaxis_title="Strength",
            xaxis=dict(showgrid=False, color=QUANT_NEON),
            yaxis=dict(showgrid=False, color=QUANT_NEON),
            zaxis=dict(showgrid=False, color=QUANT_ACCENT),
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
        ),
        font=dict(family=QUANT_FONT, color=QUANT_NEON),
        margin=dict(l=0, r=0, t=40, b=0),
        paper_bgcolor=QUANT_DARK_BG,
        plot_bgcolor=QUANT_DARK_BG
    )
    st.plotly_chart(fig3d, use_container_width=True)
else:
    demo_time = np.linspace(0, 49, 50)
    demo_pairs = np.array([0, 1, 2])
    demo_z = np.array([
        1.5 + np.sin(demo_time/8) * 0.8 + np.cos(demo_time/4 + i) * 0.5 + 0.2*i for i in demo_pairs
    ])
    # Add a little noise for realism
    demo_z += np.random.normal(0, 0.08, demo_z.shape)
    # Make surface smoother
    fig3d = go.Figure(data=[go.Surface(
        z=demo_z,
        x=demo_time,
        y=demo_pairs,
        colorscale=[[0, QUANT_DARK_BG], [0.2, '#00FFC6'], [0.8, '#FF00A6'], [1, '#fff']],
        showscale=True,
        contours = {
            "z": {"show": True, "color": QUANT_NEON, "width": 4}
        },
        lighting=dict(ambient=0.7, diffuse=0.8, specular=0.5, roughness=0.5, fresnel=0.2),
        opacity=0.98
    )])
    fig3d.update_layout(
        template="plotly_dark",
        width=1200,
        height=600,
        scene=dict(
            xaxis_title="Time",
            yaxis_title="Asset Pair",
            zaxis_title="Strength",
            xaxis=dict(showgrid=True, gridcolor=QUANT_NEON, color=QUANT_NEON, zerolinecolor=QUANT_ACCENT, tickfont=dict(color=QUANT_NEON)),
            yaxis=dict(showgrid=True, gridcolor=QUANT_NEON, color=QUANT_NEON, zerolinecolor=QUANT_ACCENT, tickfont=dict(color=QUANT_NEON)),
            zaxis=dict(showgrid=True, gridcolor=QUANT_ACCENT, color=QUANT_ACCENT, zerolinecolor=QUANT_NEON, tickfont=dict(color=QUANT_ACCENT)),
            camera=dict(eye=dict(x=2.2, y=1.8, z=1.6), up=dict(x=0, y=0, z=1)),
            aspectmode='manual', aspectratio=dict(x=2, y=1, z=0.7)
        ),
        font=dict(family=QUANT_FONT, color=QUANT_NEON),
        margin=dict(l=0, r=0, t=40, b=0),
        paper_bgcolor=QUANT_DARK_BG,
        plot_bgcolor=QUANT_DARK_BG
    )
    st.plotly_chart(fig3d, use_container_width=True)
