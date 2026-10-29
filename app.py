"""
Gold & Bitcoin Quantitative Backtesting Platform — Streamlit Dashboard

Design:
    - Dark theme with amber/gold accents (inspired by BacktestingMax)
    - Sidebar for all configuration inputs
    - Main area: candlestick chart with indicators, equity curve, tearsheet, trade log
    - Professional, institutional-grade look

Entry point:
    streamlit run app.py
"""

import sys
import hashlib
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from datetime import datetime, timedelta

# --- Ensure project root is in path ---
sys.path.insert(0, ".")

from data.yfinance_source import YFinanceSource
from data.cache import CachedDataSource
from data.interval_limits import (
    INTERVAL_LIMITS, VALID_INTERVALS, INTERVAL_LABELS,
    validate_date_range, get_max_start_date,
)
from indicators.trend import ema, supertrend, adx
from indicators.momentum import rsi, macd
from indicators.volatility import bollinger_bands, atr
from indicators.volume import vwap
from indicators.registry import INDICATOR_REGISTRY, INDICATOR_FUNCTIONS, INDICATOR_PARAMS
from strategy.ema_crossover import EMACrossover
from risk.controls import apply_risk_controls, RiskParams
from analytics.tearsheet import compute_tearsheet
from logging_.recorder import record_run, get_all_runs


# ======================================================================
#  PAGE CONFIG & CUSTOM CSS
# ======================================================================

st.set_page_config(
    page_title="Gold & BTC Backtester",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom dark theme CSS with amber/gold accents
st.markdown("""
<style>
    /* === Import Google Font === */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* === Global Dark Theme === */
    .stApp {
        background-color: #0f1117;
        color: #e0e0e0;
        font-family: 'Inter', sans-serif;
    }

    /* === Sidebar === */
    section[data-testid="stSidebar"] {
        background-color: #1a1d27;
        border-right: 1px solid #2a2d3a;
    }
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #f0a030;
    }

    /* === Headers === */
    .main-title {
        font-family: 'Inter', sans-serif;
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #f0a030, #e8873a, #d4a44a);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0;
        letter-spacing: -0.5px;
    }

    .subtitle {
        font-family: 'Inter', sans-serif;
        font-size: 0.95rem;
        color: #8a8d9a;
        margin-top: 0;
        margin-bottom: 2rem;
    }

    /* === Metric Cards === */
    .metric-card {
        background: linear-gradient(135deg, #1a1d27, #22263a);
        border: 1px solid #2a2d3a;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        text-align: center;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    .metric-card:hover {
        border-color: #f0a030;
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(240,160,48,0.15);
    }
    .metric-label {
        font-size: 0.75rem;
        color: #8a8d9a;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.3rem;
        font-weight: 500;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #f0f0f0;
    }
    .metric-value.positive { color: #00d4aa; }
    .metric-value.negative { color: #ff6b6b; }

    /* === Section Headers === */
    .section-header {
        font-family: 'Inter', sans-serif;
        font-size: 1.15rem;
        font-weight: 600;
        color: #f0a030;
        margin-top: 2rem;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #2a2d3a;
        letter-spacing: 0.5px;
    }

    /* === Status Badge === */
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    .badge-info {
        background: rgba(240,160,48,0.15);
        color: #f0a030;
        border: 1px solid rgba(240,160,48,0.3);
    }
    .badge-success {
        background: rgba(0,212,170,0.15);
        color: #00d4aa;
        border: 1px solid rgba(0,212,170,0.3);
    }
    .badge-danger {
        background: rgba(255,107,107,0.15);
        color: #ff6b6b;
        border: 1px solid rgba(255,107,107,0.3);
    }

    /* === Interval Limit Badge === */
    .interval-badge {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 6px;
        font-size: 0.7rem;
        font-weight: 500;
        background: rgba(240,160,48,0.1);
        color: #f0a030;
        border: 1px solid rgba(240,160,48,0.2);
        margin-left: 0.3rem;
    }

    /* === Button Styling === */
    .stButton > button {
        background: linear-gradient(135deg, #f0a030, #e8873a);
        color: #0f1117;
        border: none;
        border-radius: 8px;
        font-weight: 700;
        font-size: 1rem;
        padding: 0.6rem 2rem;
        letter-spacing: 0.5px;
        transition: all 0.3s ease;
        width: 100%;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #ffc04a, #f0a030);
        box-shadow: 0 4px 15px rgba(240,160,48,0.4);
        transform: translateY(-1px);
    }

    /* === Divider === */
    .gold-divider {
        height: 2px;
        background: linear-gradient(90deg, transparent, #f0a030, transparent);
        margin: 1.5rem 0;
        border: none;
    }

    /* === Trade Table === */
    .trade-win { color: #00d4aa; font-weight: 600; }
    .trade-loss { color: #ff6b6b; font-weight: 600; }

    /* === Hide Streamlit branding === */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* === Selectbox & Input Styling === */
    .stSelectbox label, .stDateInput label, .stSlider label, .stNumberInput label {
        color: #c0c0c0 !important;
        font-weight: 500 !important;
    }

    /* === Expander Styling === */
    .streamlit-expanderHeader {
        color: #f0a030 !important;
        font-weight: 600 !important;
    }
</style>
""", unsafe_allow_html=True)


# ======================================================================
#  HELPER FUNCTIONS
# ======================================================================

def create_metric_card(label: str, value: str, is_positive: bool = None) -> str:
    """Generate HTML for a styled metric card."""
    value_class = "metric-value"
    if is_positive is True:
        value_class += " positive"
    elif is_positive is False:
        value_class += " negative"

    return f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="{value_class}">{value}</div>
    </div>
    """


def create_candlestick_chart(
    df: pd.DataFrame,
    indicators_to_show: list,
    indicator_params: dict,
    trades: list = None,
) -> go.Figure:
    """Create a Plotly candlestick chart with optional indicators and trade markers."""

    # Determine if we need subplots (RSI, MACD, ADX go in separate panes)
    overlay_indicators = []
    subplot_indicators = []

    for ind_name in indicators_to_show:
        if ind_name in ["RSI", "MACD", "ADX"]:
            subplot_indicators.append(ind_name)
        else:
            overlay_indicators.append(ind_name)

    num_subplots = 1 + len(subplot_indicators)
    row_heights = [0.6] + [0.2] * len(subplot_indicators) if subplot_indicators else [1.0]

    subplot_titles = [""] + [ind for ind in subplot_indicators]

    fig = make_subplots(
        rows=num_subplots, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=row_heights,
        subplot_titles=subplot_titles,
    )

    # --- Candlestick ---
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="OHLC",
            increasing_line_color="#00d4aa",
            decreasing_line_color="#ff6b6b",
            increasing_fillcolor="#00d4aa",
            decreasing_fillcolor="#ff6b6b",
        ),
        row=1, col=1,
    )

    # --- Overlay indicators (on main chart) ---
    overlay_colors = ["#f0a030", "#5b9bd5", "#ed7d31", "#70ad47", "#9b59b6"]
    color_idx = 0

    for ind_name in overlay_indicators:
        params = indicator_params.get(ind_name, {})
        color = overlay_colors[color_idx % len(overlay_colors)]
        color_idx += 1

        if ind_name == "EMA":
            result = ema(df, **params)
            fig.add_trace(go.Scatter(
                x=df.index, y=result, name=f"EMA({params.get('period', 20)})",
                line=dict(color=color, width=1.5),
            ), row=1, col=1)

        elif ind_name == "Supertrend":
            result = supertrend(df, **params)
            # Color based on direction
            up_mask = result["direction"] == 1
            down_mask = result["direction"] == -1
            st_series = result["supertrend"]
            fig.add_trace(go.Scatter(
                x=df.index, y=st_series.where(up_mask),
                name="Supertrend (Bull)", line=dict(color="#00d4aa", width=1.5),
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=df.index, y=st_series.where(down_mask),
                name="Supertrend (Bear)", line=dict(color="#ff6b6b", width=1.5),
            ), row=1, col=1)

        elif ind_name == "Bollinger Bands":
            result = bollinger_bands(df, **params)
            fig.add_trace(go.Scatter(
                x=df.index, y=result["upper"], name="BB Upper",
                line=dict(color="#5b9bd5", width=1, dash="dash"),
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=df.index, y=result["lower"], name="BB Lower",
                line=dict(color="#5b9bd5", width=1, dash="dash"),
                fill="tonexty", fillcolor="rgba(91,155,213,0.08)",
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=df.index, y=result["middle"], name="BB Middle",
                line=dict(color="#5b9bd5", width=1),
            ), row=1, col=1)

        elif ind_name == "ATR":
            result = atr(df, **params)
            fig.add_trace(go.Scatter(
                x=df.index, y=result, name=f"ATR({params.get('period', 14)})",
                line=dict(color=color, width=1.5),
            ), row=1, col=1)

        elif ind_name == "VWAP":
            result = vwap(df)
            fig.add_trace(go.Scatter(
                x=df.index, y=result, name="VWAP",
                line=dict(color="#9b59b6", width=1.5, dash="dot"),
            ), row=1, col=1)

    # --- Subplot indicators ---
    for i, ind_name in enumerate(subplot_indicators):
        subplot_row = 2 + i
        params = indicator_params.get(ind_name, {})

        if ind_name == "RSI":
            result = rsi(df, **params)
            fig.add_trace(go.Scatter(
                x=df.index, y=result, name=f"RSI({params.get('period', 14)})",
                line=dict(color="#f0a030", width=1.5),
            ), row=subplot_row, col=1)
            # Overbought/oversold lines
            fig.add_hline(y=70, line_dash="dash", line_color="rgba(255,107,107,0.5)",
                          row=subplot_row, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="rgba(0,212,170,0.5)",
                          row=subplot_row, col=1)

        elif ind_name == "MACD":
            result = macd(df, **params)
            fig.add_trace(go.Scatter(
                x=df.index, y=result["macd"], name="MACD",
                line=dict(color="#5b9bd5", width=1.5),
            ), row=subplot_row, col=1)
            fig.add_trace(go.Scatter(
                x=df.index, y=result["signal"], name="Signal",
                line=dict(color="#f0a030", width=1.5),
            ), row=subplot_row, col=1)
            colors = ["#00d4aa" if v >= 0 else "#ff6b6b" for v in result["histogram"]]
            fig.add_trace(go.Bar(
                x=df.index, y=result["histogram"], name="Histogram",
                marker_color=colors, opacity=0.6,
            ), row=subplot_row, col=1)

        elif ind_name == "ADX":
            result = adx(df, **params)
            fig.add_trace(go.Scatter(
                x=df.index, y=result, name=f"ADX({params.get('period', 14)})",
                line=dict(color="#ed7d31", width=1.5),
            ), row=subplot_row, col=1)
            fig.add_hline(y=25, line_dash="dash", line_color="rgba(240,160,48,0.4)",
                          row=subplot_row, col=1)

    # --- Trade markers on main chart ---
    if trades:
        buy_times = [t.entry_time for t in trades if t.side == "LONG"]
        buy_prices = [t.entry_price for t in trades if t.side == "LONG"]
        sell_times = [t.exit_time for t in trades]
        sell_prices = [t.exit_price for t in trades]

        short_times = [t.entry_time for t in trades if t.side == "SHORT"]
        short_prices = [t.entry_price for t in trades if t.side == "SHORT"]

        if buy_times:
            fig.add_trace(go.Scatter(
                x=buy_times, y=buy_prices, mode="markers",
                name="Long Entry", marker=dict(
                    symbol="triangle-up", size=12, color="#00d4aa",
                    line=dict(width=1, color="#ffffff"),
                ),
            ), row=1, col=1)

        if short_times:
            fig.add_trace(go.Scatter(
                x=short_times, y=short_prices, mode="markers",
                name="Short Entry", marker=dict(
                    symbol="triangle-down", size=12, color="#ff6b6b",
                    line=dict(width=1, color="#ffffff"),
                ),
            ), row=1, col=1)

        if sell_times:
            fig.add_trace(go.Scatter(
                x=sell_times, y=sell_prices, mode="markers",
                name="Exit", marker=dict(
                    symbol="x", size=10, color="#f0a030",
                    line=dict(width=2, color="#f0a030"),
                ),
            ), row=1, col=1)

    # --- Layout ---
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0f1117",
        plot_bgcolor="#0f1117",
        font=dict(family="Inter", color="#e0e0e0"),
        xaxis_rangeslider_visible=False,
        height=250 + 200 * num_subplots,
        margin=dict(l=60, r=20, t=30, b=40),
        legend=dict(
            bgcolor="rgba(26,29,39,0.9)",
            bordercolor="#2a2d3a",
            borderwidth=1,
            font=dict(size=11),
        ),
        hovermode="x unified",
    )

    # Style all axes
    for i in range(1, num_subplots + 1):
        fig.update_xaxes(
            gridcolor="#1e2130", showgrid=True,
            zeroline=False, row=i, col=1,
        )
        fig.update_yaxes(
            gridcolor="#1e2130", showgrid=True,
            zeroline=False, row=i, col=1,
        )

    return fig


def create_equity_chart(equity_curve: pd.Series, initial_capital: float) -> go.Figure:
    """Create equity curve chart with drawdown shading."""
    eq = equity_curve.dropna()

    fig = go.Figure()

    # Equity line
    fig.add_trace(go.Scatter(
        x=eq.index, y=eq.values, name="Equity",
        line=dict(color="#f0a030", width=2.5),
        fill="tozeroy",
        fillcolor="rgba(240,160,48,0.08)",
    ))

    # Initial capital reference line
    fig.add_hline(
        y=initial_capital, line_dash="dash",
        line_color="rgba(255,255,255,0.2)",
        annotation_text=f"Initial: ${initial_capital:,.0f}",
        annotation_font_color="#8a8d9a",
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0f1117",
        plot_bgcolor="#0f1117",
        font=dict(family="Inter", color="#e0e0e0"),
        height=300,
        margin=dict(l=60, r=20, t=30, b=40),
        xaxis=dict(gridcolor="#1e2130", showgrid=True, zeroline=False),
        yaxis=dict(
            gridcolor="#1e2130", showgrid=True, zeroline=False,
            title="Portfolio Value ($)",
        ),
        showlegend=False,
        hovermode="x unified",
    )

    return fig


# ======================================================================
#  SIDEBAR — CONFIGURATION
# ======================================================================

with st.sidebar:
    st.markdown('<h1 style="color: #f0a030; margin-bottom: 0;">Gold & BTC</h1>', unsafe_allow_html=True)
    st.markdown('<p style="color: #8a8d9a; margin-top: 0;">Quantitative Backtester</p>', unsafe_allow_html=True)

    st.markdown('<div class="gold-divider"></div>', unsafe_allow_html=True)

    # --- Asset Selection ---
    st.markdown("### 1. Select Asset")
    asset = st.selectbox(
        "Asset",
        options=["BTC-USD", "GC=F"],
        format_func=lambda x: "Bitcoin (BTC-USD)" if x == "BTC-USD" else "Gold Futures (GC=F)",
        label_visibility="collapsed",
    )

    # --- Interval Selection ---
    st.markdown("### 2. Timeframe")
    interval = st.selectbox(
        "Interval",
        options=sorted(VALID_INTERVALS, key=lambda x: ["1m","5m","15m","30m","1h","1d"].index(x)),
        format_func=lambda x: INTERVAL_LABELS[x],
        index=5,  # Default to 1d
        label_visibility="collapsed",
    )

    max_days = INTERVAL_LIMITS[interval]
    if max_days:
        st.markdown(
            f'<span class="interval-badge">Max lookback: {max_days} days</span>',
            unsafe_allow_html=True,
        )

    # --- Date Range ---
    st.markdown("### 3. Date Range")
    end_date = datetime.now()
    max_start = get_max_start_date(interval, end_date)

    # Default start: half of max range or 180 days, whichever is smaller
    if max_days:
        default_start = end_date - timedelta(days=min(max_days // 2, 180))
    else:
        default_start = end_date - timedelta(days=365)

    col_s, col_e = st.columns(2)
    with col_s:
        start_date = st.date_input("Start", value=default_start.date(), min_value=max_start.date())
    with col_e:
        end_date_input = st.date_input("End", value=end_date.date())

    # --- Strategy ---
    st.markdown("### 4. Strategy")
    strategy_name = st.selectbox(
        "Strategy",
        options=["EMA Crossover"],
        label_visibility="collapsed",
    )

    col_f, col_sl = st.columns(2)
    with col_f:
        fast_period = st.number_input("Fast EMA", min_value=2, max_value=50, value=9, step=1)
    with col_sl:
        slow_period = st.number_input("Slow EMA", min_value=5, max_value=200, value=21, step=1)

    if fast_period >= slow_period:
        st.error("Fast period must be less than slow period.")

    # --- Risk Management ---
    st.markdown("### 5. Risk Controls")
    initial_capital = st.number_input(
        "Initial Capital ($)", min_value=1000, max_value=10_000_000,
        value=100_000, step=10_000,
    )
    risk_per_trade = st.slider("Risk per Trade (%)", 0.5, 5.0, 1.0, 0.5) / 100
    stop_loss_atr = st.slider("Stop-Loss (ATR x)", 0.5, 5.0, 2.0, 0.5)
    take_profit_atr = st.slider("Take-Profit (ATR x)", 0.5, 10.0, 3.0, 0.5)
    max_dd_pct = st.slider("Max Drawdown Kill (%)", 5, 50, 20, 5) / 100
    commission_bps = st.slider("Commission (bps/side)", 0.0, 50.0, 10.0, 1.0)
    slippage_bps = st.slider("Slippage (bps/side)", 0.0, 50.0, 5.0, 1.0)
    max_leverage = st.slider("Max Leverage (x)", 1.0, 10.0, 1.0, 0.5)
    kill_switch_mode = st.selectbox("Kill-Switch Mode", ["latch", "reset"], index=0,
        help="latch = halt trading permanently once max drawdown is hit; reset = resume after a cooldown.")

    # --- Indicators ---
    st.markdown("### 6. Chart Indicators")
    selected_indicators = []
    indicator_params_input = {}

    for category, ind_names in INDICATOR_REGISTRY.items():
        with st.expander(f"{category}", expanded=False):
            for ind_name in ind_names:
                if st.checkbox(ind_name, key=f"ind_{ind_name}"):
                    selected_indicators.append(ind_name)
                    params = INDICATOR_PARAMS.get(ind_name, {})
                    ind_params = {}
                    for param_key, param_info in params.items():
                        val = st.slider(
                            param_info["label"],
                            min_value=param_info["min"],
                            max_value=param_info["max"],
                            value=param_info["default"],
                            step=param_info["step"],
                            key=f"param_{ind_name}_{param_key}",
                        )
                        ind_params[param_key] = val
                    indicator_params_input[ind_name] = ind_params

    st.markdown('<div class="gold-divider"></div>', unsafe_allow_html=True)

    # --- Run Button ---
    run_backtest = st.button("Run Backtest", use_container_width=True)


# ======================================================================
#  MAIN AREA
# ======================================================================

# --- Header ---
st.markdown('<div class="main-title">Gold & Bitcoin Backtesting Platform</div>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Multi-timeframe quantitative backtesting with institutional-grade analytics</p>', unsafe_allow_html=True)


# --- Run Backtest ---
if run_backtest:
    if fast_period >= slow_period:
        st.error("Fast EMA period must be less than Slow EMA period. Adjust in the sidebar.")
        st.stop()

    # Convert dates to datetime
    start_dt = datetime.combine(start_date, datetime.min.time())
    end_dt = datetime.combine(end_date_input, datetime.max.time())

    # Validate date range
    try:
        validate_date_range(interval, start_dt, end_dt)
    except ValueError as e:
        st.error(f"Invalid date range: {e}")
        st.stop()

    # --- Fetch Data ---
    with st.spinner("Fetching market data..."):
        source = CachedDataSource(YFinanceSource())
        try:
            df = source.fetch_ohlcv(asset, interval, start_dt, end_dt)
        except Exception as e:
            st.error(f"Data fetch failed: {e}")
            st.stop()

    # Store in session state
    st.session_state["df"] = df
    st.session_state["asset"] = asset
    st.session_state["interval"] = interval

    # --- Generate Signals ---
    with st.spinner("Running strategy..."):
        strategy = EMACrossover(fast_period=fast_period, slow_period=slow_period)
        signals = strategy.generate_signals(df)

    # --- Apply Risk Controls ---
    with st.spinner("Applying risk management..."):
        risk_params = RiskParams(
            initial_capital=initial_capital,
            risk_per_trade=risk_per_trade,
            atr_period=14,
            atr_multiplier=2.0,
            stop_loss_atr=stop_loss_atr,
            take_profit_atr=take_profit_atr,
            max_drawdown_pct=max_dd_pct,
            commission_bps=commission_bps,
            slippage_bps=slippage_bps,
            max_leverage=max_leverage,
            kill_switch_mode=kill_switch_mode,
        )
        trades, equity_curve = apply_risk_controls(signals, df, risk_params)

    # --- Compute Tearsheet ---
    tearsheet = compute_tearsheet(trades, equity_curve, initial_capital, asset, interval)

    # --- Record Run ---
    data_fingerprint = hashlib.sha256(
        pd.util.hash_pandas_object(df, index=True).values.tobytes()
    ).hexdigest()
    run_id = record_run(
        asset=asset,
        interval=interval,
        start_date=start_dt,
        end_date=end_dt,
        strategy_name=strategy.name,
        strategy_params=strategy.get_params(),
        tearsheet=tearsheet,
        risk_params=risk_params,
        data_fingerprint=data_fingerprint,
    )

    # Store results in session state
    st.session_state["trades"] = trades
    st.session_state["equity_curve"] = equity_curve
    st.session_state["tearsheet"] = tearsheet
    st.session_state["run_id"] = run_id
    st.session_state["strategy"] = strategy
    st.session_state["risk_params"] = risk_params
    st.session_state["selected_indicators"] = selected_indicators
    st.session_state["indicator_params_input"] = indicator_params_input


# --- Display Results (from session state) ---
if "tearsheet" in st.session_state:
    tearsheet = st.session_state["tearsheet"]
    trades = st.session_state["trades"]
    equity_curve = st.session_state["equity_curve"]
    df = st.session_state["df"]
    asset = st.session_state["asset"]
    run_id = st.session_state["run_id"]
    strategy = st.session_state["strategy"]
    risk_params = st.session_state["risk_params"]
    selected_inds = st.session_state.get("selected_indicators", [])
    ind_params = st.session_state.get("indicator_params_input", {})

    # --- Top Metrics Row ---
    st.markdown('<div class="section-header">PERFORMANCE OVERVIEW</div>', unsafe_allow_html=True)

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        is_pos = tearsheet.total_return > 0
        st.markdown(create_metric_card(
            "Total Return",
            f"{tearsheet.total_return * 100:+.2f}%",
            is_pos,
        ), unsafe_allow_html=True)

    with col2:
        st.markdown(create_metric_card(
            "Sharpe Ratio",
            f"{tearsheet.sharpe_ratio:.2f}",
            tearsheet.sharpe_ratio > 1,
        ), unsafe_allow_html=True)

    with col3:
        st.markdown(create_metric_card(
            "Max Drawdown",
            f"{tearsheet.max_drawdown * 100:.2f}%",
            False if tearsheet.max_drawdown < -0.1 else None,
        ), unsafe_allow_html=True)

    with col4:
        st.markdown(create_metric_card(
            "Win Rate",
            f"{tearsheet.win_rate * 100:.1f}%",
            tearsheet.win_rate > 0.5,
        ), unsafe_allow_html=True)

    with col5:
        st.markdown(create_metric_card(
            "Profit Factor",
            f"{tearsheet.profit_factor:.2f}",
            tearsheet.profit_factor > 1,
        ), unsafe_allow_html=True)

    # --- Second Metrics Row ---
    col6, col7, col8, col9, col10 = st.columns(5)

    with col6:
        st.markdown(create_metric_card(
            "Sortino Ratio", f"{tearsheet.sortino_ratio:.2f}",
            tearsheet.sortino_ratio > 1,
        ), unsafe_allow_html=True)

    with col7:
        st.markdown(create_metric_card(
            "Calmar Ratio", f"{tearsheet.calmar_ratio:.2f}",
            tearsheet.calmar_ratio > 1,
        ), unsafe_allow_html=True)

    with col8:
        st.markdown(create_metric_card(
            "Avg Trade Return", f"{tearsheet.avg_trade_return * 100:.2f}%",
            tearsheet.avg_trade_return > 0,
        ), unsafe_allow_html=True)

    with col9:
        st.markdown(create_metric_card(
            "Max DD Duration", f"{tearsheet.max_drawdown_duration} bars",
        ), unsafe_allow_html=True)

    with col10:
        st.markdown(create_metric_card(
            "Total Trades", f"{tearsheet.num_trades}",
        ), unsafe_allow_html=True)

    # --- Candlestick Chart ---
    st.markdown('<div class="section-header">PRICE CHART & INDICATORS</div>', unsafe_allow_html=True)

    candle_fig = create_candlestick_chart(df, selected_inds, ind_params, trades)
    st.plotly_chart(candle_fig, use_container_width=True)

    # --- Equity Curve ---
    st.markdown('<div class="section-header">EQUITY CURVE</div>', unsafe_allow_html=True)

    equity_fig = create_equity_chart(equity_curve, risk_params.initial_capital)
    st.plotly_chart(equity_fig, use_container_width=True)

    # --- Trade Log ---
    st.markdown('<div class="section-header">TRADE LOG</div>', unsafe_allow_html=True)

    if trades:
        trade_data = []
        for t in trades:
            trade_data.append({
                "Entry Time": t.entry_time.strftime("%Y-%m-%d %H:%M") if hasattr(t.entry_time, 'strftime') else str(t.entry_time),
                "Exit Time": t.exit_time.strftime("%Y-%m-%d %H:%M") if hasattr(t.exit_time, 'strftime') else str(t.exit_time),
                "Side": t.side,
                "Entry Price": f"${t.entry_price:,.2f}",
                "Exit Price": f"${t.exit_price:,.2f}",
                "Size": f"{t.position_size:.4f}",
                "P&L": f"${t.pnl:,.2f}",
                "Fee": f"${t.fee:,.2f}",
                "Return": f"{t.pnl_pct * 100:+.2f}%",
                "Exit Reason": t.exit_reason.replace("_", " ").title(),
            })

        trade_df = pd.DataFrame(trade_data)
        st.dataframe(
            trade_df,
            use_container_width=True,
            hide_index=True,
            height=min(400, 40 + 35 * len(trade_data)),
        )
    else:
        st.info("No trades were generated in this backtest period.")

    # --- Run Info ---
    st.markdown('<div class="gold-divider"></div>', unsafe_allow_html=True)
    info_col1, info_col2, info_col3 = st.columns(3)
    with info_col1:
        st.markdown(f'<span class="status-badge badge-success">Run ID: {run_id[:8]}...</span>', unsafe_allow_html=True)
    with info_col2:
        st.markdown(f'<span class="status-badge badge-info">{strategy.name} ({strategy.get_params()})</span>', unsafe_allow_html=True)
    with info_col3:
        st.markdown(f'<span class="status-badge badge-info">{asset} | {st.session_state.get("interval", "1d")}</span>', unsafe_allow_html=True)

else:
    # --- Welcome State ---
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 4rem 2rem;">
        <div style="font-size: 3rem; margin-bottom: 1rem;">&#9878;</div>
        <h2 style="color: #f0a030; font-family: Inter;">Configure & Run Your Backtest</h2>
        <p style="color: #8a8d9a; font-size: 1.1rem; max-width: 600px; margin: 0 auto;">
            Select an asset, timeframe, and strategy parameters in the sidebar,
            then hit <strong style="color: #f0a030;">Run Backtest</strong> to see
            the candlestick chart, equity curve, and full performance tearsheet.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # --- Run History ---
    st.markdown('<div class="section-header">RECENT BACKTEST RUNS</div>', unsafe_allow_html=True)
    try:
        history = get_all_runs()
        if not history.empty:
            display_cols = [
                "timestamp", "asset", "interval", "strategy_name",
                "total_return", "sharpe_ratio", "max_drawdown", "win_rate", "num_trades",
            ]
            available_cols = [c for c in display_cols if c in history.columns]
            hist_display = history[available_cols].head(10)

            # Format numeric columns
            if "total_return" in hist_display.columns:
                hist_display["total_return"] = hist_display["total_return"].apply(lambda x: f"{x*100:.2f}%")
            if "sharpe_ratio" in hist_display.columns:
                hist_display["sharpe_ratio"] = hist_display["sharpe_ratio"].apply(lambda x: f"{x:.2f}")
            if "max_drawdown" in hist_display.columns:
                hist_display["max_drawdown"] = hist_display["max_drawdown"].apply(lambda x: f"{x*100:.2f}%")
            if "win_rate" in hist_display.columns:
                hist_display["win_rate"] = hist_display["win_rate"].apply(lambda x: f"{x*100:.1f}%")

            st.dataframe(hist_display, use_container_width=True, hide_index=True)
        else:
            st.markdown(
                '<p style="color: #8a8d9a; text-align: center;">No backtest runs yet. Configure and run your first backtest!</p>',
                unsafe_allow_html=True,
            )
    except Exception:
        st.markdown(
            '<p style="color: #8a8d9a; text-align: center;">No backtest history available.</p>',
            unsafe_allow_html=True,
        )
