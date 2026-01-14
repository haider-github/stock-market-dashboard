 import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import ta
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="Stock Market Analytics",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {padding: 0rem 1rem;}
    .stMetric {background-color: #f0f2f6; padding: 15px; border-radius: 10px;}
    h1 {color: #1f77b4;}
    .positive {color: green; font-weight: bold;}
    .negative {color: red; font-weight: bold;}
    </style>
""", unsafe_allow_html=True)

# Title
st.title("📈 Real-Time Stock Market Analytics Dashboard")
st.markdown("*Powered by Machine Learning & Technical Analysis*")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Stock selection
    stock_symbol = st.text_input(
        "Enter Stock Symbol:",
        value="AAPL",
        help="Examples: AAPL, GOOGL, MSFT, TSLA, AMZN"
    ).upper()
    
    # Time period
    time_period = st.selectbox(
        "Select Time Period:",
        ["1mo", "3mo", "6mo", "1y", "2y", "5y", "max"],
        index=3
    )
    
    # Prediction days
    prediction_days = st.slider(
        "Prediction Days:",
        min_value=7,
        max_value=90,
        value=30,
        step=7
    )
    
    st.markdown("---")
    st.markdown("### 📊 Popular Stocks")
    popular = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN", "META", "NVDA", "JPM"]
    for symbol in popular:
        if st.button(symbol, key=f"btn_{symbol}"):
            stock_symbol = symbol

# Cache data loading
@st.cache_data(ttl=600)
def load_stock_data(symbol, period):
    try:
        import time
        time.sleep(1)  # Add small delay to avoid rate limiting
        stock = yf.Ticker(symbol)
        data = stock.history(period=period)
        info = stock.info
        return data, info, None
    except Exception as e:
        return None, None, str(e)

# Load data
with st.spinner(f'Loading {stock_symbol} data...'):
    data, info, error = load_stock_data(stock_symbol, time_period)

if data is None or data.empty:
    st.error(f"❌ Could not fetch data for {stock_symbol}.")
    if error:
        if "Rate limited" in error or "Too Many Requests" in error:
            st.warning("⏳ **Yahoo Finance Rate Limit Reached**")
            st.info("**Solutions:**\n- Wait 2-5 minutes and refresh\n- Try a different stock symbol\n- The free Yahoo Finance API has request limits")
        else:
            st.error(f"Error details: {error}")
    st.info("💡 **Try these popular symbols:** MSFT, GOOGL, TSLA, AMZN, META")
    st.stop()

# Stock info
if info:
    col1, col2, col3, col4, col5 = st.columns(5)
    
    current_price = data['Close'].iloc[-1]
    previous_close = info.get('previousClose', data['Close'].iloc[-2])
    change = current_price - previous_close
    change_pct = (change / previous_close) * 100
    
    with col1:
        st.metric(
            label="Current Price",
            value=f"${current_price:.2f}",
            delta=f"{change:.2f} ({change_pct:.2f}%)"
        )
    
    with col2:
        st.metric(
            label="Market Cap",
            value=f"${info.get('marketCap', 0)/1e9:.2f}B"
        )
    
    with col3:
        st.metric(
            label="52W High",
            value=f"${info.get('fiftyTwoWeekHigh', 0):.2f}"
        )
    
    with col4:
        st.metric(
            label="52W Low",
            value=f"${info.get('fiftyTwoWeekLow', 0):.2f}"
        )
    
    with col5:
        st.metric(
            label="P/E Ratio",
            value=f"{info.get('trailingPE', 0):.2f}"
        )

st.markdown("---")

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Price Chart",
    "📈 Technical Analysis",
    "🤖 ML Predictions",
    "📋 Statistics",
    "📰 Company Info"
])

with tab1:
    st.subheader(f"{stock_symbol} Stock Price History")
    
    # Candlestick chart
    fig = go.Figure(data=[go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name='Price'
    )])
    
    # Add volume
    fig.add_trace(go.Bar(
        x=data.index,
        y=data['Volume'],
        name='Volume',
        yaxis='y2',
        marker_color='rgba(100,100,250,0.3)'
    ))
    
    fig.update_layout(
        title=f'{stock_symbol} Price & Volume',
        yaxis_title='Price ($)',
        yaxis2=dict(
            title='Volume',
            overlaying='y',
            side='right'
        ),
        xaxis_rangeslider_visible=False,
        height=600,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Price statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Average Price", f"${data['Close'].mean():.2f}")
    with col2:
        st.metric("Volatility (Std Dev)", f"${data['Close'].std():.2f}")
    with col3:
        returns = data['Close'].pct_change().mean() * 100
        st.metric("Avg Daily Return", f"{returns:.3f}%")

with tab2:
    st.subheader("Technical Analysis Indicators")
    
    # Calculate technical indicators
    data['SMA_20'] = ta.trend.sma_indicator(data['Close'], window=20)
    data['SMA_50'] = ta.trend.sma_indicator(data['Close'], window=50)
    data['EMA_12'] = ta.trend.ema_indicator(data['Close'], window=12)
    data['RSI'] = ta.momentum.rsi(data['Close'], window=14)
    
    # MACD
    macd = ta.trend.MACD(data['Close'])
    data['MACD'] = macd.macd()
    data['MACD_Signal'] = macd.macd_signal()
    data['MACD_Hist'] = macd.macd_diff()
    
    # Bollinger Bands
    bollinger = ta.volatility.BollingerBands(data['Close'])
    data['BB_High'] = bollinger.bollinger_hband()
    data['BB_Low'] = bollinger.bollinger_lband()
    data['BB_Mid'] = bollinger.bollinger_mavg()
    
    # Price with Moving Averages
    fig_ma = go.Figure()
    fig_ma.add_trace(go.Scatter(x=data.index, y=data['Close'], name='Close', line=dict(color='blue', width=2)))
    fig_ma.add_trace(go.Scatter(x=data.index, y=data['SMA_20'], name='SMA 20', line=dict(color='orange', dash='dash')))
    fig_ma.add_trace(go.Scatter(x=data.index, y=data['SMA_50'], name='SMA 50', line=dict(color='red', dash='dash')))
    fig_ma.update_layout(title='Price with Moving Averages', height=400, hovermode='x unified')
    st.plotly_chart(fig_ma, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # RSI
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(x=data.index, y=data['RSI'], name='RSI', line=dict(color='purple')))
        fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
        fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
        fig_rsi.update_layout(title='RSI (Relative Strength Index)', height=300)
        st.plotly_chart(fig_rsi, use_container_width=True)
        
        # Current RSI status
        current_rsi = data['RSI'].iloc[-1]
        if current_rsi > 70:
            st.warning(f"⚠️ RSI: {current_rsi:.2f} - **Overbought** (Potential Sell Signal)")
        elif current_rsi < 30:
            st.success(f"✅ RSI: {current_rsi:.2f} - **Oversold** (Potential Buy Signal)")
        else:
            st.info(f"ℹ️ RSI: {current_rsi:.2f} - **Neutral**")
    
    with col2:
        # MACD
        fig_macd = go.Figure()
        fig_macd.add_trace(go.Scatter(x=data.index, y=data['MACD'], name='MACD', line=dict(color='blue')))
        fig_macd.add_trace(go.Scatter(x=data.index, y=data['MACD_Signal'], name='Signal', line=dict(color='red')))
        fig_macd.add_trace(go.Bar(x=data.index, y=data['MACD_Hist'], name='Histogram', marker_color='gray'))
        fig_macd.update_layout(title='MACD (Moving Average Convergence Divergence)', height=300)
        st.plotly_chart(fig_macd, use_container_width=True)
        
        # MACD signal
        if data['MACD'].iloc[-1] > data['MACD_Signal'].iloc[-1]:
            st.success("✅ MACD above Signal - **Bullish**")
        else:
            st.warning("⚠️ MACD below Signal - **Bearish**")
    
    # Bollinger Bands
    fig_bb = go.Figure()
    fig_bb.add_trace(go.Scatter(x=data.index, y=data['BB_High'], name='Upper Band', line=dict(color='red', dash='dash')))
    fig_bb.add_trace(go.Scatter(x=data.index, y=data['Close'], name='Close', line=dict(color='blue')))
    fig_bb.add_trace(go.Scatter(x=data.index, y=data['BB_Low'], name='Lower Band', line=dict(color='green', dash='dash')))
    fig_bb.add_trace(go.Scatter(x=data.index, y=data['BB_Mid'], name='Middle Band', line=dict(color='orange')))
    fig_bb.update_layout(title='Bollinger Bands', height=400, hovermode='x unified')
    st.plotly_chart(fig_bb, use_container_width=True)

with tab3:
    st.subheader("🤖 Machine Learning Price Predictions")
    
    try:
        from prophet import Prophet
        
        # Prepare data for Prophet
        df_prophet = data.reset_index()[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
        
        with st.spinner('Training ML model...'):
            # Prophet model
            model = Prophet(daily_seasonality=True, yearly_seasonality=True)
            model.fit(df_prophet)
            
            # Make predictions
            future = model.make_future_dataframe(periods=prediction_days)
            forecast = model.predict(future)
        
        # Plot predictions
        fig_pred = go.Figure()
        
        # Historical data
        fig_pred.add_trace(go.Scatter(
            x=df_prophet['ds'],
            y=df_prophet['y'],
            name='Actual Price',
            line=dict(color='blue', width=2)
        ))
        
        # Predictions
        fig_pred.add_trace(go.Scatter(
            x=forecast['ds'],
            y=forecast['yhat'],
            name='Predicted Price',
            line=dict(color='red', width=2, dash='dash')
        ))
        
        # Confidence interval
        fig_pred.add_trace(go.Scatter(
            x=forecast['ds'],
            y=forecast['yhat_upper'],
            fill=None,
            mode='lines',
            line_color='rgba(255,0,0,0)',
            showlegend=False
        ))
        
        fig_pred.add_trace(go.Scatter(
            x=forecast['ds'],
            y=forecast['yhat_lower'],
            fill='tonexty',
            mode='lines',
            line_color='rgba(255,0,0,0)',
            name='Confidence Interval',
            fillcolor='rgba(255,0,0,0.2)'
        ))
        
        fig_pred.update_layout(
            title=f'{stock_symbol} Price Prediction ({prediction_days} Days)',
            xaxis_title='Date',
            yaxis_title='Price ($)',
            height=600,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_pred, use_container_width=True)
        
        # Prediction summary
        col1, col2, col3 = st.columns(3)
        
        last_actual = df_prophet['y'].iloc[-1]
        future_prediction = forecast['yhat'].iloc[-1]
        
        predicted_change = ((future_prediction - last_actual) / last_actual) * 100
        
        with col1:
            st.metric(
                "Current Price",
                f"${last_actual:.2f}"
            )
        
        with col2:
            st.metric(
                f"Predicted ({prediction_days}d)",
                f"${future_prediction:.2f}",
                delta=f"{predicted_change:.2f}%"
            )
        
        with col3:
            if predicted_change > 0:
                st.success(f"📈 **Bullish Outlook** (+{predicted_change:.2f}%)")
            else:
                st.error(f"📉 **Bearish Outlook** ({predicted_change:.2f}%)")
        
        # Show prediction table
        st.subheader("Detailed Predictions")
        future_forecast = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(prediction_days)
        future_forecast.columns = ['Date', 'Predicted Price', 'Lower Bound', 'Upper Bound']
        future_forecast = future_forecast.reset_index(drop=True)
        st.dataframe(future_forecast.style.format({
            'Predicted Price': '${:.2f}',
            'Lower Bound': '${:.2f}',
            'Upper Bound': '${:.2f}'
        }), use_container_width=True)
        
    except ImportError:
        st.error("⚠️ Prophet library not installed. Install it with: pip install prophet")
        st.info("Using simple linear regression as alternative...")
        
        # Simple linear prediction as fallback
        from sklearn.linear_model import LinearRegression
        
        df_simple = data.reset_index()[['Date', 'Close']]
        df_simple['Days'] = (df_simple['Date'] - df_simple['Date'].min()).dt.days
        
        X = df_simple[['Days']].values
        y = df_simple['Close'].values
        
        model = LinearRegression()
        model.fit(X, y)
        
        # Predict future
        last_day = df_simple['Days'].iloc[-1]
        future_days = np.array([[last_day + i] for i in range(1, prediction_days + 1)])
        predictions = model.predict(future_days)
        
        # Plot
        fig_simple = go.Figure()
        fig_simple.add_trace(go.Scatter(x=df_simple['Date'], y=y, name='Actual', line=dict(color='blue')))
        
        future_dates = pd.date_range(start=df_simple['Date'].iloc[-1], periods=prediction_days+1)[1:]
        fig_simple.add_trace(go.Scatter(x=future_dates, y=predictions, name='Predicted', line=dict(color='red', dash='dash')))
        
        fig_simple.update_layout(title='Simple Linear Prediction', height=400)
        st.plotly_chart(fig_simple, use_container_width=True)
    
    except Exception as e:
        st.error(f"❌ Error in prediction: {str(e)}")
        st.info("Try selecting a different time period or stock symbol.")

with tab4:
    st.subheader("📊 Statistical Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Returns distribution
        returns = data['Close'].pct_change().dropna() * 100
        
        fig_hist = px.histogram(
            returns,
            nbins=50,
            title='Daily Returns Distribution',
            labels={'value': 'Return (%)', 'count': 'Frequency'}
        )
        fig_hist.add_vline(x=0, line_dash="dash", line_color="red")
        st.plotly_chart(fig_hist, use_container_width=True)
        
        st.metric("Mean Daily Return", f"{returns.mean():.3f}%")
        st.metric("Return Volatility", f"{returns.std():.3f}%")
    
    with col2:
        # Cumulative returns
        cumulative_returns = (1 + data['Close'].pct_change()).cumprod()
        
        fig_cum = go.Figure()
        fig_cum.add_trace(go.Scatter(
            x=data.index,
            y=cumulative_returns,
            name='Cumulative Returns',
            line=dict(color='green', width=2)
        ))
        fig_cum.update_layout(title='Cumulative Returns', height=400)
        st.plotly_chart(fig_cum, use_container_width=True)
        
        total_return = ((data['Close'].iloc[-1] / data['Close'].iloc[0]) - 1) * 100
        st.metric("Total Return", f"{total_return:.2f}%")
    
    # Correlation with volume
    st.subheader("Price-Volume Correlation")
    correlation = data['Close'].corr(data['Volume'])
    st.metric("Correlation Coefficient", f"{correlation:.3f}")
    
    if abs(correlation) > 0.5:
        st.info("Strong correlation between price and volume")
    else:
        st.info("Weak correlation between price and volume")

with tab5:
    st.subheader(f"📰 {stock_symbol} Company Information")
    
    if info:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Basic Info")
            st.write(f"**Company:** {info.get('longName', 'N/A')}")
            st.write(f"**Sector:** {info.get('sector', 'N/A')}")
            st.write(f"**Industry:** {info.get('industry', 'N/A')}")
            st.write(f"**Country:** {info.get('country', 'N/A')}")
            st.write(f"**Website:** {info.get('website', 'N/A')}")
            
            st.markdown("### Key Metrics")
            st.write(f"**Market Cap:** ${info.get('marketCap', 0)/1e9:.2f}B")
            st.write(f"**P/E Ratio:** {info.get('trailingPE', 'N/A')}")
            st.write(f"**EPS:** ${info.get('trailingEps', 'N/A')}")
            st.write(f"**Dividend Yield:** {info.get('dividendYield', 0)*100:.2f}%")
        
        with col2:
            st.markdown("### Business Summary")
            st.write(info.get('longBusinessSummary', 'No description available'))
            
            st.markdown("### Analyst Recommendation")
            recommendation = info.get('recommendationKey', 'N/A').upper()
            if recommendation in ['BUY', 'STRONG_BUY']:
                st.success(f"**{recommendation}** 📈")
            elif recommendation in ['SELL', 'STRONG_SELL']:
                st.error(f"**{recommendation}** 📉")
            else:
                st.info(f"**{recommendation}**")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "⚠️ Disclaimer: This is for educational purposes only. Not financial advice. | "
    "Built with Streamlit 🎈 & Yahoo Finance"
    "</div>",
    unsafe_allow_html=True
)