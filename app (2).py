import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import io

# 1. PAGE CONFIGURATION
st.set_page_config(page_title="VoltRisk Analytics", page_icon="⚡", layout="wide")

# 2. THEMATIC STYLING
st.markdown("""
    <style>
    .stApp { background-color: #0E1117; color: #FFFFFF; }
    [data-testid="stMetricValue"] { font-weight: 700 !important; }
    section[data-testid="stSidebar"] { background-color: #161B22 !important; border-right: 1px solid #30363D; }
    .signal-box { padding: 20px; border-radius: 10px; text-align: center; border: 1px solid #30363D; margin-bottom: 20px; }
    .beginner-card { background-color: #161B22; padding: 15px; border-radius: 10px; border-left: 5px solid #00FBFF; margin-bottom: 10px; }
    </style>
    """, unsafe_allow_html=True)

# 3. SIDEBAR
with st.sidebar:
    st.markdown("<h1 style='text-align: center; color: #00FFAA;'>⚡ VOLTRISK</h1>", unsafe_allow_html=True)
    st.markdown("---")
    ticker = st.text_input("Asset Ticker", value="NVDA").upper()
    investment = st.number_input("Capital Allocation ($)", min_value=10.0, value=1000.0)
    time_horizon = st.slider("Days to Forecast", 1, 730, 252)
    iterations = st.slider("Number of Simulations", 100, 10000, 2000, step=100)
    st.markdown("### 🛠️ Risk Scenarios")
    apply_crash = st.checkbox("Simulate '2020 Covid crash'")
    start_sim = st.button("RUN SIMULATION")
    st.markdown("---")
    st.markdown("**⚙️ Engine Notes**\nVoltRisk uses GBM math to simulate potential futures based on 3-year history.")

# 4. MAIN DASHBOARD
st.title("⚡ :blue[Volt]Risk Analytics")

if start_sim:
    with st.spinner('Calculating...'):
        data = yf.download(ticker, start=(datetime.now() - timedelta(days=1095)), auto_adjust=False)
        spy_data = yf.download("SPY", start=(datetime.now() - timedelta(days=1095)), auto_adjust=False)
        
        if data.empty:
            st.error("Ticker not found.")
        else:
            # ENGINE
            def run_mc(df, inv, n):
                if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
                rets = df['Adj Close'].pct_change().dropna()
                mu, sigma, last = rets.mean(), rets.std(), df['Adj Close'].iloc[-1]
                daily = np.random.normal(mu, sigma, (time_horizon, n))
                paths = np.zeros_like(daily)
                paths[0] = last * (1 + daily[0])
                for t in range(1, time_horizon): paths[t] = paths[t-1] * (1 + daily[t])
                return (paths / last) * inv

            asset_paths = run_mc(data, investment, iterations)
            spy_paths = run_mc(spy_data, investment, 1000)
            final_vals = asset_paths[-1]
            win_prob = (np.sum(final_vals > investment) / iterations) * 100
            tp_95, sl_5, mean_outcome = np.percentile(final_vals, 95), np.percentile(final_vals, 5), np.mean(final_vals)
            cum_max = np.maximum.accumulate(asset_paths, axis=0)
            avg_max_dd = np.mean(np.min((asset_paths - cum_max) / cum_max, axis=0)) * 100

            # 5. NEW: SPEEDOMETER & STRATEGY SECTION
            st.divider()
            c_gauge, c_signal = st.columns([1, 1])
            
            with c_gauge:
                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = win_prob,
                    number = {'suffix': "%", 'font': {'color': "#FFFFFF"}},
                    title = {'text': "WIN PROBABILITY", 'font': {'size': 24}},
                    gauge = {
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "#1b5e20"},
                        'steps': [
                            {'range': [0, 40], 'color': "#FF4B4B"},
                            {'range': [40, 70], 'color': "#FFD700"},
                            {'range': [70, 100], 'color': "#00FFAA"}]
                    }
                ))
                fig_gauge.update_layout(paper_bgcolor='rgba(0,0,0,0)', height=300, margin=dict(t=50, b=0))
                st.plotly_chart(fig_gauge, use_container_width=True)

            with c_signal:
                st.markdown("<br><br>", unsafe_allow_html=True)
                if win_prob > 60:
                    st.markdown(f"<div class='signal-box' style='background-color: rgba(0, 255, 170, 0.1); border-color: #00FFAA;'><h2 style='color:#00FFAA;'>BUY SIGNAL</h2><p>Math favors profit over loss.</p></div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class='signal-box'><h2 style='color:#8B949E;'>WAIT</h2><p>Risk is currently too high.</p></div>", unsafe_allow_html=True)

            # 6. METRICS
            st.divider()
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("CURRENT PRICE", f"${data['Adj Close'].iloc[-1]:,.2f}")
            m2.metric("MEAN OUTCOME", f"${mean_outcome:,.2f}")
            m3.metric("MAX DRAWDOWN", f"{avg_max_dd:.1f}%")
            m4.metric("STOP LOSS (5%)", f"${sl_5:,.2f}")

            # 7. CHART
            st.subheader("🔍 Market Benchmark & Volatility Bands")
            fig = go.Figure()
            days_axis = list(range(time_horizon))
            for i in range(min(50, iterations)):
                fig.add_trace(go.Scatter(x=days_axis, y=asset_paths[:, i], line=dict(color='rgba(0, 251, 255, 0.05)', width=1), hoverinfo='none', showlegend=False))
            fig.add_trace(go.Scatter(x=days_axis, y=np.mean(spy_paths, axis=1), name="S&P 500 (Market)", line=dict(color='#FFFFFF', width=2, dash='dot')))
            fig.add_trace(go.Scatter(x=days_axis, y=np.mean(asset_paths, axis=1), name="Asset Projection", line=dict(color='#FFD700', width=4)))
            
            p5, p95 = np.percentile(asset_paths, 5, axis=1), np.percentile(asset_paths, 95, axis=1)
            fig.add_trace(go.Scatter(x=days_axis+days_axis[::-1], y=list(p95)+list(p5)[::-1], fill='toself', fillcolor='rgba(0, 255, 170, 0.1)', line_color='rgba(0,0,0,0)', name='Risk Band'))
            
            fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=500)
            st.plotly_chart(fig, use_container_width=True)

            # 8. BEGINNER'S CHEAT SHEET
            st.divider()
            st.subheader("💡 How to Read This (For Beginners)")
            b1, b2, b3 = st.columns(3)
            with b1:
                st.markdown("<div class='beginner-card'><b>The Speedometer</b><br>Tells you the chance of making $1 or more. Above 70% is very strong.</div>", unsafe_allow_html=True)
            with b2:
                st.markdown("<div class='beginner-card'><b>Max Drawdown</b><br>The 'Stomach Test'. It shows how big the dips might feel during the year.</div>", unsafe_allow_html=True)
            with b3:
                st.markdown("<div class='beginner-card'><b>The Gold Line</b><br>If the Gold Line is ABOVE the White Dashed Line, this stock is beating the market.</div>", unsafe_allow_html=True)
else:
    st.info("👋 **Welcome to VoltRisk.** Enter a ticker like 'AAPL' and click 'Run Simulation' to start.")