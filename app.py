#!/usr/bin/env python3
"""
Ultimate Trading System - Streamlit Dashboard
Web-based interface for trading analytics and monitoring
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Ultimate Trading System",
    page_icon="💹",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
        .main {
            padding-top: 2rem;
        }
        .metric-card {
            background-color: #f0f2f6;
            padding: 1.5rem;
            border-radius: 0.5rem;
            border-left: 4px solid #1f77b4;
        }
    </style>
""", unsafe_allow_html=True)


def load_stock_data(symbol: str, period: str = "1y") -> pd.DataFrame:
    """
    Load stock data from yFinance
    """
    try:
        data = yf.download(symbol, period=period, progress=False)
        return data
    except Exception as e:
        st.error(f"Error loading data for {symbol}: {str(e)}")
        return None


def calculate_technical_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate technical indicators
    """
    # SMA (Simple Moving Average)
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['SMA_200'] = data['Close'].rolling(window=200).mean()
    
    # RSI (Relative Strength Index)
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD (Moving Average Convergence Divergence)
    exp1 = data['Close'].ewm(span=12, adjust=False).mean()
    exp2 = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = exp1 - exp2
    data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()
    data['MACD_Histogram'] = data['MACD'] - data['Signal_Line']
    
    return data


def main():
    """
    Main Streamlit application
    """
    # Title
    st.title("🚀 Ultimate Trading System")
    st.markdown("Professional Algorithmic Trading Dashboard")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Stock selection
        symbol = st.text_input(
            "Enter Stock Symbol",
            value="NVDA",
            placeholder="e.g., NVDA, AAPL, TSLA"
        ).upper()
        
        # Time period selection
        period = st.selectbox(
            "Select Time Period",
            ["1mo", "3mo", "6mo", "1y", "2y", "5y"]
        )
        
        # Refresh button
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.rerun()
    
    if symbol:
        # Load data
        with st.spinner(f"Loading data for {symbol}..."):
            data = load_stock_data(symbol, period)
        
        if data is not None:
            # Calculate indicators
            data = calculate_technical_indicators(data)
            
            # Main metrics
            col1, col2, col3, col4 = st.columns(4)
            
            current_price = data['Close'].iloc[-1]
            prev_price = data['Close'].iloc[-2]
            price_change = current_price - prev_price
            price_change_pct = (price_change / prev_price) * 100
            
            with col1:
                st.metric(
                    "Current Price",
                    f"${current_price:.2f}",
                    f"{price_change_pct:+.2f}%"
                )
            
            with col2:
                st.metric(
                    "52 Week High",
                    f"${data['Close'].max():.2f}"
                )
            
            with col3:
                st.metric(
                    "52 Week Low",
                    f"${data['Close'].min():.2f}"
                )
            
            with col4:
                st.metric(
                    "Volume",
                    f"{data['Volume'].iloc[-1]/1e6:.2f}M"
                )
            
            st.markdown("---")
            
            # Price Chart
            st.subheader("📊 Price Chart")
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data['Close'],
                name=f"{symbol} Close Price",
                line=dict(color='#1f77b4', width=2)
            ))
            
            # Add Moving Averages
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data['SMA_20'],
                name='SMA 20',
                line=dict(color='orange', width=1)
            ))
            
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data['SMA_50'],
                name='SMA 50',
                line=dict(color='red', width=1)
            ))
            
            fig.update_layout(
                title=f"{symbol} Price Chart",
                xaxis_title="Date",
                yaxis_title="Price (USD)",
                template="plotly_white",
                hovermode="x unified",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Technical Indicators
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 RSI (Relative Strength Index)")
                
                fig_rsi = go.Figure()
                fig_rsi.add_trace(go.Scatter(
                    x=data.index,
                    y=data['RSI'],
                    fill='tozeroy',
                    name='RSI'
                ))
                
                # Add overbought/oversold lines
                fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
                fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
                
                fig_rsi.update_layout(
                    template="plotly_white",
                    height=300,
                    showlegend=False
                )
                
                st.plotly_chart(fig_rsi, use_container_width=True)
            
            with col2:
                st.subheader("📊 MACD (Moving Average Convergence Divergence)")
                
                fig_macd = go.Figure()
                fig_macd.add_trace(go.Scatter(
                    x=data.index,
                    y=data['MACD'],
                    name='MACD',
                    line=dict(color='blue')
                ))
                
                fig_macd.add_trace(go.Scatter(
                    x=data.index,
                    y=data['Signal_Line'],
                    name='Signal Line',
                    line=dict(color='red')
                ))
                
                fig_macd.update_layout(
                    template="plotly_white",
                    height=300,
                    hovermode="x unified"
                )
                
                st.plotly_chart(fig_macd, use_container_width=True)
            
            # Latest Data Table
            st.subheader("📄 Latest Data")
            
            display_data = data[['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_20', 'SMA_50', 'RSI']].tail(10)
            st.dataframe(
                display_data.style.format({
                    'Open': '${:.2f}',
                    'High': '${:.2f}',
                    'Low': '${:.2f}',
                    'Close': '${:.2f}',
                    'Volume': '{:,.0f}',
                    'SMA_20': '${:.2f}',
                    'SMA_50': '${:.2f}',
                    'RSI': '{:.2f}'
                }),
                use_container_width=True
            )
            
            # Footer
            st.markdown("---")
            st.markdown("""
            <div style='text-align: center'>
                <small>Data provided by yFinance | Last updated: {}</small>
            </div>
            """.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')), unsafe_allow_html=True)


if __name__ == "__main__":
    main()
