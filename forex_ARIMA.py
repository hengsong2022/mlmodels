import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from ta.trend import MACD
from ta.momentum import RSIIndicator
from sklearn.metrics import mean_squared_error
from datetime import datetime, timedelta
import pytz
import os
import pickle
import time

# Create models directory
MODELS_DIR = "saved_models"
os.makedirs(MODELS_DIR, exist_ok=True)

# Streamlit configuration
st.set_page_config(page_title="Forex Analysis", layout="wide")
st.title("Forex Market Analysis Tool")

# Sidebar inputs
forex_pair = st.sidebar.selectbox(
    "Forex Pair", 
    ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "USDCAD=X"]
)
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End Date", datetime.now().date())
interval = st.sidebar.selectbox("Data Interval", ["1h", "1d", "1wk"])

# Forex market hours (24/5)
def is_forex_open():
    now = datetime.now(pytz.timezone("UTC"))
    return now.weekday() < 5  # Forex is open 24/5

@st.cache_data
def load_forex_data(pair, start_date, end_date):
    try:
        data = yf.download(pair, start=start_date, end=end_date + timedelta(days=1))
        return data if not data.empty else None
    except Exception as e:
        st.error(f"Data load failed: {str(e)}")
        return None

@st.cache_data(ttl=60)
def get_live_forex_data(pair, interval):
    try:
        period = "5d" if interval == "1h" else "60d"
        data = yf.download(pair, period=period, interval=interval)
        return data if not data.empty else None
    except Exception as e:
        st.error(f"Live data failed: {str(e)}")
        return None

def preprocess_forex_data(data):
    try:
        # Ensure required columns
        req_cols = ['Open', 'High', 'Low', 'Close']
        for col in req_cols:
            if col not in data.columns:
                data[col] = data['Adj Close'] if col == 'Close' else 0
        
        # Calculate technical indicators
        close_series = pd.Series(data['Close'].values.flatten())
        data['RSI'] = RSIIndicator(close_series, window=14).rsi()
        data['MACD'] = MACD(close_series).macd()
        
        # Fill any remaining NaNs
        data.fillna(method='ffill', inplace=True)
        data.fillna(0, inplace=True)
        
        return data[['Open', 'High', 'Low', 'Close', 'RSI', 'MACD']]
    except Exception as e:
        st.error(f"Preprocessing error: {str(e)}")
        return None

def train_arima_model(data):
    try:
        processed = preprocess_forex_data(data.copy())
        if processed is None:
            return None, None
        
        # Train ARIMA model
        model = ARIMA(processed['Close'], order=(1,1,1))
        model_fit = model.fit()
        
        # Make predictions
        predictions = model_fit.predict(start=0, end=len(processed)-1)
        mse = mean_squared_error(processed['Close'], predictions)
        
        # Store metadata
        model_fit.training_date = datetime.now()
        model_fit.mse = mse
        
        return model_fit, predictions
    except Exception as e:
        st.error(f"Training error: {str(e)}")
        return None, None

# Main app
try:
    data = load_forex_data(forex_pair, start_date, end_date)
    if data is None:
        st.stop()
    
    # Historical Analysis
    st.header("Historical Analysis")
    fig, ax = plt.subplots(figsize=(12,4))
    data['Close'].plot(ax=ax)
    ax.set(xlabel="Date", ylabel="Price", title=f"{forex_pair} Price History")
    ax.grid()
    st.pyplot(fig)
    
    # Model Training
    st.sidebar.subheader("Model Training")
    if st.sidebar.checkbox("Train ARIMA Model"):
        with st.spinner("Training ARIMA model..."):
            model, predictions = train_arima_model(data)
            if model:
                model_file = f"arima_{forex_pair}_{datetime.now().strftime('%Y%m%d_%H%M')}.pkl"
                model_path = os.path.join(MODELS_DIR, model_file)
                
                with open(model_path, "wb") as f:
                    pickle.dump(model, f)
                
                st.sidebar.success(f"ARIMA model saved to {model_path}\nMSE: {model.mse:.6f}")
                
                # Plot predictions vs actual
                st.subheader("Model Predictions vs Actual")
                fig, ax = plt.subplots(figsize=(12,4))
                data['Close'].plot(ax=ax, label='Actual')
                pd.Series(predictions, index=data.index).plot(ax=ax, label='Predicted', alpha=0.7)
                ax.set(xlabel="Date", ylabel="Price", title="ARIMA Model Performance")
                ax.legend()
                ax.grid()
                st.pyplot(fig)
    
    # Live Analysis
    st.header("Live Forex Analysis")
    
    # Model Loading
    model = None
    if st.sidebar.checkbox("Load ARIMA Model"):
        try:
            model_files = [f for f in os.listdir(MODELS_DIR) if f.startswith(f"arima_{forex_pair}")]
            if not model_files:
                st.sidebar.warning(f"No ARIMA models found for {forex_pair}")
            else:
                latest = sorted(model_files)[-1]
                with open(os.path.join(MODELS_DIR, latest), "rb") as f:
                    model = pickle.load(f)
                st.sidebar.success(f"Loaded: {latest}\nMSE: {model.mse:.6f}")
        except Exception as e:
            st.sidebar.error(f"Load error: {str(e)}")
    
    # Live Prediction
    if model and st.sidebar.checkbox("Show Live Forecast"):
        stop_signal = st.sidebar.button("Stop Updates")
        live_display = st.empty()
        chart_display = st.empty()
        
        while not stop_signal:
            current_time = datetime.now(pytz.timezone("UTC"))
            forex_open = is_forex_open()
            live_data = get_live_forex_data(forex_pair, interval)
            
            if live_data is None:
                time.sleep(10)
                continue
                
            processed = preprocess_forex_data(live_data.copy())
            if processed is None:
                time.sleep(10)
                continue
                
            with live_display.container():
                cols = st.columns(3)
                cols[0].metric("Time", current_time.strftime("%H:%M:%S UTC"))
                
                # Get current price safely
                current_price = live_data['Close'].iloc[-1]
                if isinstance(current_price, (pd.Series, np.ndarray)):
                    current_price = current_price[0] if len(current_price) > 0 else 0
                cols[1].metric("Current Price", f"{float(current_price):.5f}")
                
                if forex_open:
                    try:
                        # Forecast next period
                        forecast = model.forecast(steps=1)[0]
                        change_pct = (forecast - current_price) / current_price * 100
                        
                        if forecast > current_price:
                            cols[2].success(f"↑ {forecast:.5f} (+{change_pct:.2f}%)")
                        else:
                            cols[2].error(f"↓ {forecast:.5f} ({change_pct:.2f}%)")
                    except Exception as e:
                        cols[2].error(f"Forecast error: {str(e)}")
                else:
                    cols[2].warning("Forex Closed (Weekend)")
            
            with chart_display.container():
                fig, ax = plt.subplots(figsize=(12,4))
                live_data['Close'].tail(48).plot(ax=ax)  # Show last 48 periods
                ax.set(xlabel="Time", ylabel="Price", title="Recent Price Action")
                ax.grid()
                st.pyplot(fig)
            
            time.sleep(30 if interval != "1h" else 60)
            if stop_signal:
                st.sidebar.success("Updates stopped")
                break
except Exception as e:
    st.error(f"Application error: {str(e)}")