import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from finance_utils import get_stock_data, add_financial_features, create_data_summary
from agent import ai_finance_agent

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="FinSight AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# PROFESSIONAL MINIMAL CSS
# --------------------------------------------------
st.markdown("""
<style>
:root {
    --bg: #0b1220;
    --panel: #111827;
    --panel-2: #0f172a;
    --border: #1f2937;
    --text: #e5e7eb;
    --muted: #94a3b8;
    --accent: #22c55e;
    --accent-soft: rgba(34, 197, 94, 0.12);
    --danger: #ef4444;
    --warning: #f59e0b;
    --blue: #3b82f6;
}

html, body, [class*="css"] {
    font-family: Inter, ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;
}

.stApp {
    background: var(--bg);
    color: var(--text);
}

#MainMenu, footer, header {
    visibility: hidden;
}

section[data-testid="stSidebar"] {
    background: #0f172a;
    border-right: 1px solid var(--border);
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1500px;
}

/* Headings */
.app-title {
    font-size: 2rem;
    font-weight: 700;
    color: var(--text);
    margin-bottom: 0.2rem;
    letter-spacing: -0.02em;
}

.app-subtitle {
    color: var(--muted);
    font-size: 0.98rem;
    margin-bottom: 1.5rem;
}

.section-title {
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--muted);
    margin-bottom: 0.8rem;
    font-weight: 600;
}

/* Cards */
.panel-card {
    background: linear-gradient(180deg, #111827 0%, #0f172a 100%);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 18px 20px;
}

/* KPI metrics */
[data-testid="metric-container"] {
    background: linear-gradient(180deg, #111827 0%, #0f172a 100%);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 16px 18px !important;
    box-shadow: none;
}

[data-testid="stMetricLabel"] {
    color: var(--muted) !important;
    font-size: 0.78rem !important;
    font-weight: 600;
}

[data-testid="stMetricValue"] {
    color: var(--text) !important;
    font-size: 1.45rem !important;
    font-weight: 700;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: transparent;
    gap: 0.4rem;
    border-bottom: 1px solid var(--border);
    padding-bottom: 0.5rem;
}

.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: var(--muted);
    border-radius: 10px;
    padding: 0.55rem 0.9rem;
    font-weight: 600;
}

.stTabs [aria-selected="true"] {
    color: var(--text) !important;
    background: #111827 !important;
    border: 1px solid var(--border) !important;
}

/* Inputs */
.stTextInput input, .stSelectbox > div > div {
    background: #111827 !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
}

.stTextInput input:focus {
    border-color: #334155 !important;
    box-shadow: none !important;
}

/* Buttons */
.stButton > button {
    background: var(--text);
    color: #0b1220;
    border: none;
    border-radius: 10px;
    font-weight: 700;
    padding: 0.65rem 1rem;
    width: 100%;
}

.stButton > button:hover {
    background: #ffffff;
    color: #0b1220;
}

/* Sidebar branding */
.sidebar-brand {
    font-size: 1.1rem;
    font-weight: 700;
    color: var(--text);
    margin-bottom: 1rem;
}

.sidebar-muted {
    color: var(--muted);
    font-size: 0.85rem;
    margin-bottom: 1.25rem;
}

/* AI response */
.ai-box {
    background: linear-gradient(180deg, #111827 0%, #0f172a 100%);
    border: 1px solid var(--border);
    border-left: 4px solid var(--accent);
    border-radius: 14px;
    padding: 18px 20px;
    color: var(--text);
    line-height: 1.7;
}

.ai-label {
    color: var(--muted);
    font-size: 0.74rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.7rem;
    font-weight: 700;
}

/* Chat history */
.history-q {
    background: #0f172a;
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 12px 14px;
    margin-bottom: 8px;
    color: var(--text);
}

.history-a {
    background: #111827;
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 12px 14px;
    margin-bottom: 14px;
    color: var(--muted);
    line-height: 1.6;
}

.history-label {
    font-size: 0.72rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.35rem;
    font-weight: 700;
}

/* Tables */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border);
    border-radius: 12px;
    overflow: hidden;
}

/* Info box */
.empty-state {
    text-align: center;
    padding: 4rem 1rem;
    color: var(--muted);
    border: 1px dashed var(--border);
    border-radius: 16px;
    background: rgba(255,255,255,0.01);
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

if "df" not in st.session_state:
    st.session_state["df"] = None

if "ticker" not in st.session_state:
    st.session_state["ticker"] = None

# --------------------------------------------------
# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
with st.sidebar:
    st.markdown('<div class="sidebar-brand">FinSight AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-muted">Equity analysis dashboard with AI-assisted interpretation.</div>', unsafe_allow_html=True)

    st.markdown("### Controls")
    ticker_input = st.text_input("Ticker Symbol", value="NVDA").upper()
    period = st.selectbox("Time Period", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=3)

    if st.button("Load Stock Data"):
        try:
            with st.spinner(f"Loading {ticker_input}..."):
                df = get_stock_data(ticker_input, period)
                df = add_financial_features(df)

                st.session_state["df"] = df
                st.session_state["ticker"] = ticker_input
                st.session_state["chat_history"] = []

            st.success(f"{ticker_input} loaded")

            # -------- DEBUG CHECK --------
            st.write("Latest date in dataframe:", df["Date"].max())
            st.write(df.tail())

        except Exception as e:
            st.error(f"Could not load data: {e}")

    st.markdown("---")
    st.markdown("### Quick Tickers")
    q1, q2, q3 = st.columns(3)
    quick_tickers = ["AAPL", "MSFT", "NVDA", "AMZN", "META", "TSLA"]

    cols = [q1, q2, q3]
    for i, qt in enumerate(quick_tickers):
        with cols[i % 3]:
            if st.button(qt, key=f"quick_{qt}"):
                try:
                    with st.spinner(f"Loading {qt}..."):
                        df = get_stock_data(qt, period)
                        df = add_financial_features(df)

                        st.session_state["df"] = df
                        st.session_state["ticker"] = qt
                        st.session_state["chat_history"] = []

                    st.rerun()
                except Exception as e:
                    st.error(str(e))

# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.markdown('<div class="app-title">FinSight AI</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-subtitle">Analyze historical market data, inspect price behavior, and ask an AI analyst for concise interpretation.</div>',
    unsafe_allow_html=True
)

# --------------------------------------------------
# EMPTY STATE
# --------------------------------------------------
if st.session_state["df"] is None:
    st.markdown("""
    <div class="empty-state">
        <div style="font-size:1.15rem;font-weight:700;color:#e5e7eb;margin-bottom:0.5rem;">No stock loaded</div>
        <div>Use the left sidebar to load a ticker such as NVDA, AAPL, MSFT, or TSLA.</div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

df = st.session_state["df"]
ticker = st.session_state["ticker"]

# --------------------------------------------------
# SAFE NUMERIC EXTRACTION
# --------------------------------------------------
close_series = df["Close"].squeeze()
latest_close = float(close_series.iloc[-1])
first_close = float(close_series.iloc[0])
prev_close = float(close_series.iloc[-2]) if len(close_series) > 1 else latest_close

total_return = ((latest_close - first_close) / first_close) * 100 if first_close != 0 else 0.0
day_change = ((latest_close - prev_close) / prev_close) * 100 if prev_close != 0 else 0.0
high_period = float(close_series.max())
low_period = float(close_series.min())

avg_volume = None
if "Volume" in df.columns:
    avg_volume = float(df["Volume"].squeeze().mean())

volatility = None
if "Daily_Return" in df.columns:
    dr = df["Daily_Return"].squeeze().dropna()
    if len(dr) > 0:
        volatility = float(dr.std() * 100)

# --------------------------------------------------
# KPI ROW
# --------------------------------------------------
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric(f"{ticker} Close", f"${latest_close:.2f}", f"{day_change:+.2f}%")
k2.metric("Period Return", f"{total_return:+.2f}%")
k3.metric("Period High", f"${high_period:.2f}")
k4.metric("Period Low", f"${low_period:.2f}")
k5.metric("Volatility", f"{volatility:.2f}%" if volatility is not None else "N/A")

st.markdown("")

# --------------------------------------------------
# TABS
# --------------------------------------------------
tab_ai, tab_charts, tab_data, tab_summary = st.tabs(
    ["Ask AI", "Charts", "Data", "Summary"]
)

# --------------------------------------------------
# ASK AI TAB
# --------------------------------------------------
with tab_ai:
    left, right = st.columns([3, 2], gap="large")

    with left:
        st.markdown('<div class="section-title">AI Analyst</div>', unsafe_allow_html=True)

        sample_questions = [
            "What trend do you observe in this stock?",
            "Is this stock volatile over this period?",
            "What do the moving averages suggest?",
            "How would you summarize recent performance?",
            "What does trading volume indicate?",
            "What are the main risk signals in this data?"
        ]

        selected_question = st.selectbox("Sample question", sample_questions)
        custom_question = st.text_input("Custom question", placeholder="Ask a question about this stock...")

        final_question = custom_question.strip() if custom_question.strip() else selected_question

        if st.button("Analyze with AI"):
            try:
                with st.spinner("Generating analysis..."):
                    answer = ai_finance_agent(final_question, df, ticker)

                st.session_state["chat_history"].append(
                    {"question": final_question, "answer": answer}
                )

                st.markdown(
                    f"""
                    <div class="ai-box">
                        <div class="ai-label">AI Response</div>
                        {answer}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            except Exception as e:
                st.error(f"Agent error: {e}")

    with right:
        st.markdown('<div class="section-title">Recent History</div>', unsafe_allow_html=True)

        if not st.session_state["chat_history"]:
            st.info("No AI queries yet.")
        else:
            for chat in reversed(st.session_state["chat_history"][-5:]):
                st.markdown(
                    f"""
                    <div class="history-q">
                        <div class="history-label">Question</div>
                        {chat["question"]}
                    </div>
                    <div class="history-a">
                        <div class="history-label">Answer</div>
                        {chat["answer"]}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            if st.button("Clear History"):
                st.session_state["chat_history"] = []
                st.rerun()

# --------------------------------------------------
# CHARTS TAB
# --------------------------------------------------
with tab_charts:
    DARK_BG = "#0b1220"
    PANEL_BG = "#111827"
    BORDER = "#1f2937"
    TEXT = "#94a3b8"
    GREEN = "#22c55e"
    BLUE = "#3b82f6"
    AMBER = "#f59e0b"
    RED = "#ef4444"

    def style_ax(ax, fig, date_axis=True):
        fig.patch.set_facecolor(DARK_BG)
        ax.set_facecolor(PANEL_BG)
        for spine in ax.spines.values():
            spine.set_edgecolor(BORDER)
        ax.tick_params(colors=TEXT, labelsize=8)
        ax.xaxis.label.set_color(TEXT)
        ax.yaxis.label.set_color(TEXT)
        ax.title.set_color("#e5e7eb")
        ax.grid(True, color=BORDER, linewidth=0.6, alpha=0.5)

        if date_axis:
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")

    st.markdown('<div class="section-title">Price and Moving Averages</div>', unsafe_allow_html=True)
    fig1, ax1 = plt.subplots(figsize=(12, 4.5))
    ax1.plot(df["Date"], df["Close"].squeeze(), linewidth=1.8, color=GREEN, label="Close")

    if "MA_10" in df.columns:
        ax1.plot(df["Date"], df["MA_10"].squeeze(), linewidth=1.2, color=BLUE, linestyle="--", label="MA 10")
    if "MA_50" in df.columns:
        ax1.plot(df["Date"], df["MA_50"].squeeze(), linewidth=1.2, color=AMBER, linestyle="--", label="MA 50")

    ax1.set_title(f"{ticker} Price Trend", pad=12, fontsize=11, fontweight="bold")
    ax1.set_ylabel("Price")
    ax1.legend(facecolor=PANEL_BG, edgecolor=BORDER, labelcolor=TEXT, fontsize=8)
    style_ax(ax1, fig1)
    plt.tight_layout()
    st.pyplot(fig1)

    c1, c2 = st.columns(2, gap="large")

    with c1:
        if "Volume" in df.columns:
            st.markdown('<div class="section-title">Volume</div>', unsafe_allow_html=True)
            fig2, ax2 = plt.subplots(figsize=(6, 3.4))
            volume_series = df["Volume"].squeeze()
            ax2.bar(df["Date"], volume_series, color=BLUE, alpha=0.8)
            ax2.set_title("Trading Volume", pad=10, fontsize=10, fontweight="bold")
            ax2.set_ylabel("Volume")
            style_ax(ax2, fig2)
            plt.tight_layout()
            st.pyplot(fig2)

    with c2:
        if "Daily_Return" in df.columns:
            st.markdown('<div class="section-title">Daily Returns</div>', unsafe_allow_html=True)
            fig3, ax3 = plt.subplots(figsize=(6, 3.4))
            returns = df["Daily_Return"].squeeze().fillna(0)
            colors = [GREEN if r >= 0 else RED for r in returns]
            ax3.bar(df["Date"], returns, color=colors, alpha=0.85)
            ax3.axhline(0, color=TEXT, linewidth=0.8)
            ax3.set_title("Daily Returns", pad=10, fontsize=10, fontweight="bold")
            ax3.set_ylabel("Return")
            style_ax(ax3, fig3)
            plt.tight_layout()
            st.pyplot(fig3)

# --------------------------------------------------
# DATA TAB
# --------------------------------------------------
with tab_data:
    d1, d2 = st.columns([3, 2], gap="large")

    with d1:
        st.markdown('<div class="section-title">Latest Rows</div>', unsafe_allow_html=True)
        st.dataframe(df.tail(15), use_container_width=True, height=420)

    with d2:
        st.markdown('<div class="section-title">Descriptive Statistics</div>', unsafe_allow_html=True)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            st.dataframe(df[numeric_cols].describe().round(4), use_container_width=True, height=420)
        else:
            st.info("No numeric columns found.")

    st.markdown('<div class="section-title">Export</div>', unsafe_allow_html=True)
    st.download_button(
        label=f"Download {ticker} CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=f"{ticker}_data.csv",
        mime="text/csv"
    )

# --------------------------------------------------
# SUMMARY TAB
# --------------------------------------------------
with tab_summary:
    s1, s2 = st.columns([1.1, 1.4], gap="large")

    with s1:
        st.markdown('<div class="section-title">Overview</div>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="panel-card">
                <div style="line-height:2;color:#e5e7eb;font-size:0.95rem;">
                    <b>Ticker:</b> {ticker}<br>
                    <b>Rows:</b> {df.shape[0]:,}<br>
                    <b>Columns:</b> {df.shape[1]}<br>
                    <b>Start Date:</b> {str(df["Date"].iloc[0])[:10]}<br>
                    <b>End Date:</b> {str(df["Date"].iloc[-1])[:10]}<br>
                    <b>Period Return:</b> {total_return:+.2f}%<br>
                    <b>Average Volume:</b> {f"{avg_volume:,.0f}" if avg_volume is not None else "N/A"}<br>
                    <b>Volatility:</b> {f"{volatility:.2f}%" if volatility is not None else "N/A"}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with s2:
        st.markdown('<div class="section-title">Column Summary</div>', unsafe_allow_html=True)
        st.code(create_data_summary(df), language=None)

       