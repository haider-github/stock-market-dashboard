import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time

st.set_page_config(
    page_title="Stock Market Dashboard",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Stock Market Dashboard")
st.markdown("Real-time stock data and technical analysis")
st.markdown("---")

with st.sidebar:
    st.header("Settings")
    stock_symbol = st.text_input("Stock Symbol:", "AAPL").upper()
    time_period = st.selectbox("Time Period:", ["1mo", "3mo", "6mo", "1y", "2y"], index=3)
    
    st.markdown("### Popular Stocks")
    if st.button("MSFT"):
        stock_symbol = "MSFT"
    if st.button("GOOGL"):
        stock_symbol = "GOOGL"
    if st.button("TSLA"):
        stock_symbol = "TSLA"
    if st.button("AMZN"):
        stock_symbol = "AMZN"

@st.cache_data(ttl=600)
def get_stock_data(symbol, period):
    try:
        time.sleep(0.5)
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period)
        info = ticker.info
        return df, info, None
    except Exception as e:
        return None, None, str(e)

with st.spinner(f'Loading {stock_symbol}...'):
    data, info, error = get_stock_data(stock_symbol, time_period)

if data is None or len(data) == 0:
    st.error(f"Cannot load data for {stock_symbol}")
    if error and "Rate" in str(error):
        st.warning("Yahoo Finance rate limit. Wait 5 minutes or try another stock.")
    st.stop()

current_price = data['Close'].iloc[-1]
prev_price = data['Close'].iloc[-2] if len(data) > 1 else current_price
price_change = current_price - prev_price
price_change_pct = (price_change / prev_price) * 100

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Price", f"${current_price:.2f}", f"{price_change:.2f} ({price_change_pct:.2f}%)")
with col2:
    st.metric("High", f"${data['High'].max():.2f}")
with col3:
    st.metric("Low", f"${data['Low'].min():.2f}")
with col4:
    st.metric("Volume", f"{data['Volume'].iloc[-1]:,.0f}")

st.markdown("---")

tab1, tab2, tab3 = st.tabs(["📊 Chart", "📈 Analysis", "ℹ️ Info"])

with tab1:
    st.subheader(f"{stock_symbol} Price Chart")
    
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name='Price'
    ))
    
    fig.update_layout(
        title=f'{stock_symbol} Stock Price',
        yaxis_title='Price (USD)',
        xaxis_rangeslider_visible=False,
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Average Price", f"${data['Close'].mean():.2f}")
    with col2:
        st.metric("Volatility", f"${data['Close'].std():.2f}")

with tab2:
    st.subheader("Technical Analysis")
    
    data['SMA20'] = data['Close'].rolling(window=20).mean()
    data['SMA50'] = data['Close'].rolling(window=50).mean()
    
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=data.index, y=data['Close'], name='Price', line=dict(color='blue')))
    fig2.add_trace(go.Scatter(x=data.index, y=data['SMA20'], name='SMA 20', line=dict(color='orange', dash='dash')))
    fig2.add_trace(go.Scatter(x=data.index, y=data['SMA50'], name='SMA 50', line=dict(color='red', dash='dash')))
    fig2.update_layout(title='Price with Moving Averages', height=400)
    st.plotly_chart(fig2, use_container_width=True)
    
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    data['RSI'] = rsi
    
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=data.index, y=data['RSI'], name='RSI', line=dict(color='purple')))
    fig3.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
    fig3.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
    fig3.update_layout(title='RSI Indicator', height=300)
    st.plotly_chart(fig3, use_container_width=True)
    
    current_rsi = data['RSI'].iloc[-1]
    if current_rsi > 70:
        st.warning(f"RSI: {current_rsi:.2f} - Overbought")
    elif current_rsi < 30:
        st.success(f"RSI: {current_rsi:.2f} - Oversold")
    else:
        st.info(f"RSI: {current_rsi:.2f} - Neutral")
    
    returns = data['Close'].pct_change().dropna() * 100
    fig4 = px.histogram(returns, nbins=50, title='Daily Returns Distribution')
    fig4.add_vline(x=0, line_dash="dash", line_color="red")
    st.plotly_chart(fig4, use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Mean Return", f"{returns.mean():.3f}%")
    with col2:
        st.metric("Return Volatility", f"{returns.std():.3f}%")

with tab3:
    st.subheader("Company Information")
    
    if info:
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Company:** {info.get('longName', 'N/A')}")
            st.write(f"**Sector:** {info.get('sector', 'N/A')}")
            st.write(f"**Industry:** {info.get('industry', 'N/A')}")
            st.write(f"**Market Cap:** ${info.get('marketCap', 0)/1e9:.2f}B")
            st.write(f"**P/E Ratio:** {info.get('trailingPE', 'N/A')}")
        
        with col2:
            summary = info.get('longBusinessSummary', 'No description available')
            st.write("**Description:**")
            st.write(summary[:500] + "..." if len(summary) > 500 else summary)

st.markdown("---")
st.markdown("*Educational purposes only. Not financial advice.*")