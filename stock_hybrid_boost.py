import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from ta.trend import MACD
from ta.momentum import RSIIndicator
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import pickle
from datetime import datetime, time, timedelta
import time as time_module
import pytz
import os
from sklearn.metrics import accuracy_score
from sklearn.base import clone

# Create models directory
MODELS_DIR = "saved_models"
os.makedirs(MODELS_DIR, exist_ok=True)

# Streamlit configuration
st.set_page_config(page_title="Stock Analysis", layout="wide")
st.title("Stock Market Analysis Tool (Hybrid Model)")

# Sidebar inputs
ticker = st.sidebar.text_input("Ticker Symbol", "AAPL")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End Date", datetime.now().date())
interval = st.sidebar.selectbox("Data Interval", ["1m", "5m", "15m", "30m", "1h", "1d"])

# Model parameters
st.sidebar.subheader("Model Parameters")

st.sidebar.markdown("**Random Forest Parameters**")
rf_n_estimators = st.sidebar.slider("RF: Number of trees", 50, 500, 100)
rf_max_depth = st.sidebar.slider("RF: Max depth", 3, 15, 6)

st.sidebar.markdown("**XGBoost Parameters**")
xgb_n_estimators = st.sidebar.slider("XGB: Number of trees", 50, 500, 100)
xgb_max_depth = st.sidebar.slider("XGB: Max depth", 3, 15, 6)
xgb_learning_rate = st.sidebar.slider("XGB: Learning rate", 0.01, 0.5, 0.1)
xgb_early_stopping = st.sidebar.slider("XGB: Early stopping rounds", 0, 50, 10)

# Voting weights
st.sidebar.markdown("**Model Voting Weights**")
rf_weight = st.sidebar.slider("Random Forest Weight", 0.0, 1.0, 0.5)
xgb_weight = st.sidebar.slider("XGBoost Weight", 0.0, 1.0, 0.5)

# Market hours configuration
MARKET_OPEN = time(9, 30)
MARKET_CLOSE = time(16, 0)
timezone = pytz.timezone("America/New_York")

def is_market_open():
    now = datetime.now(timezone).time()
    return (MARKET_OPEN <= now <= MARKET_CLOSE) and (datetime.now(timezone).weekday() < 5)

@st.cache_data
def load_data(ticker, start_date, end_date):
    try:
        data = yf.download(ticker, start=start_date, end=end_date + timedelta(days=1))
        return data if not data.empty else None
    except Exception as e:
        st.error(f"Data load failed: {str(e)}")
        return None

@st.cache_data(ttl=60)
def get_live_data(ticker, interval):
    try:
        period = "1d" if interval in ["1m", "5m", "15m", "30m", "1h"] else "1mo"
        data = yf.download(ticker, period=period, interval=interval)
        return data if not data.empty else None
    except Exception as e:
        st.error(f"Live data failed: {str(e)}")
        return None

def preprocess_data(data):
    try:
        # Ensure required columns
        req_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in req_cols:
            if col not in data.columns:
                data[col] = data['Adj Close'] if col == 'Close' else 0
        
        # Technical indicators
        close_series = pd.Series(data['Close'].values.flatten())
        data['RSI'] = RSIIndicator(close_series, window=14).rsi()
        data['MACD'] = MACD(close_series).macd()
        data['MA_50'] = data['Close'].rolling(window=50).mean()
        data['MA_200'] = data['Close'].rolling(window=200).mean()
        
        # Price changes
        data['Daily_Return'] = data['Close'].pct_change()
        data['Volatility'] = data['Daily_Return'].rolling(window=14).std()
        
        # Fill NaNs
        data.fillna(0, inplace=True)
        
        features = ['Open', 'High', 'Low', 'Close', 'Volume', 
                  'RSI', 'MACD', 'MA_50', 'MA_200', 
                  'Daily_Return', 'Volatility']
        return data[features]
    except Exception as e:
        st.error(f"Preprocessing error: {str(e)}")
        return None

class HybridModel:
    def __init__(self, rf_model, xgb_model, rf_weight=0.5, xgb_weight=0.5):
        self.rf_model = rf_model
        self.xgb_model = xgb_model
        self.rf_weight = rf_weight
        self.xgb_weight = xgb_weight
        self.feature_names = rf_model.feature_names  # Assuming both models have same features
        self.training_date = datetime.now()
        
    def predict(self, X):
        rf_pred = self.rf_model.predict(X)
        xgb_pred = self.xgb_model.predict(X)
        
        # Weighted voting
        hybrid_pred = np.where(
            (rf_pred * self.rf_weight + xgb_pred * self.xgb_weight) >= 0.5, 
            1, 
            0
        )
        return hybrid_pred
    
    def predict_proba(self, X):
        rf_proba = self.rf_model.predict_proba(X)
        xgb_proba = self.xgb_model.predict_proba(X)
        
        # Weighted average of probabilities
        hybrid_proba = (rf_proba * self.rf_weight + xgb_proba * self.xgb_weight) / (self.rf_weight + self.xgb_weight)
        return hybrid_proba

def train_models(data, rf_params, xgb_params, rf_weight=0.5, xgb_weight=0.5):
    try:
        processed = preprocess_data(data.copy())
        if processed is None:
            return None, None, None
            
        # Create target (1 if price increases next day)
        processed['Target'] = (processed['Close'].shift(-1) > processed['Close']).astype(int)
        processed.dropna(inplace=True)
        
        # Feature selection
        X = processed.iloc[:, :-1]
        y = processed['Target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        # Train Random Forest
        rf_model = RandomForestClassifier(
            n_estimators=rf_params['n_estimators'],
            max_depth=rf_params['max_depth'],
            random_state=42
        )
        rf_model.fit(X_train, y_train)
        rf_model.feature_names = list(X.columns)
        
        # Train XGBoost
        xgb_model = XGBClassifier(
            n_estimators=xgb_params['n_estimators'],
            max_depth=xgb_params['max_depth'],
            learning_rate=xgb_params['learning_rate'],
            eval_metric='logloss',
            early_stopping_rounds=xgb_params['early_stopping'] if xgb_params['early_stopping'] > 0 else None,
            use_label_encoder=False,
            random_state=42
        )
        
        eval_set = [(X_test, y_test)] if xgb_params['early_stopping'] > 0 else None
        xgb_model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=False
        )
        xgb_model.feature_names = list(X.columns)
        
        # Create hybrid model
        hybrid_model = HybridModel(rf_model, xgb_model, rf_weight, xgb_weight)
        
        return hybrid_model, X_test, y_test
    except Exception as e:
        st.error(f"Training error: {str(e)}")
        return None, None, None

# Main app
try:
    data = load_data(ticker, start_date, end_date)
    if data is None:
        st.stop()
    
    # Historical Analysis
    st.header("Historical Analysis")
    fig, ax = plt.subplots(figsize=(12,4))
    data['Close'].plot(ax=ax)
    ax.set(xlabel="Date", ylabel="Price ($)", title=f"{ticker} Price History")
    ax.grid()
    st.pyplot(fig)
    
    # Model Training
    st.sidebar.subheader("Model Training")
    if st.sidebar.checkbox("Train New Hybrid Model"):
        with st.spinner("Training hybrid model (RF + XGBoost)..."):
            rf_params = {
                'n_estimators': rf_n_estimators,
                'max_depth': rf_max_depth
            }
            
            xgb_params = {
                'n_estimators': xgb_n_estimators,
                'max_depth': xgb_max_depth,
                'learning_rate': xgb_learning_rate,
                'early_stopping': xgb_early_stopping
            }
            
            model, X_test, y_test = train_models(
                data,
                rf_params=rf_params,
                xgb_params=xgb_params,
                rf_weight=rf_weight,
                xgb_weight=xgb_weight
            )
            
            if model:
                # Save model
                model_file = f"hybrid_model_{ticker}_{datetime.now().strftime('%Y%m%d_%H%M')}.pkl"
                model_path = os.path.join(MODELS_DIR, model_file)
                
                with open(model_path, "wb") as f:
                    pickle.dump(model, f)
                
                # Evaluation
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                
                st.sidebar.success(f"Hybrid model saved to {model_path}")
                st.sidebar.success(f"Accuracy: {accuracy:.1%}")
                
                # Feature importance from both models
                st.subheader("Feature Importance")
                
                # Random Forest importance
                rf_feat_imp = pd.DataFrame({
                    'Feature': model.rf_model.feature_names,
                    'RF Importance': model.rf_model.feature_importances_
                }).sort_values('RF Importance', ascending=False)
                
                # XGBoost importance
                xgb_feat_imp = pd.DataFrame({
                    'Feature': model.xgb_model.feature_names,
                    'XGB Importance': model.xgb_model.feature_importances_
                }).sort_values('XGB Importance', ascending=False)
                
                # Combine importances
                combined_imp = pd.merge(rf_feat_imp, xgb_feat_imp, on='Feature')
                combined_imp['Hybrid Importance'] = (
                    combined_imp['RF Importance'] * rf_weight + 
                    combined_imp['XGB Importance'] * xgb_weight
                )
                combined_imp = combined_imp.sort_values('Hybrid Importance', ascending=False)
                
                # Plot
                fig, ax = plt.subplots(figsize=(10,8))
                combined_imp.plot.barh(x='Feature', y=['RF Importance', 'XGB Importance', 'Hybrid Importance'], ax=ax)
                ax.set_title("Hybrid Model Feature Importance")
                ax.legend(['Random Forest', 'XGBoost', 'Hybrid'])
                st.pyplot(fig)
    
    # Live Signals
    st.header("Live Trading Signals")
    
    # Model Loading
    model = None
    if st.sidebar.checkbox("Load Existing Model"):
        try:
            model_files = [f for f in os.listdir(MODELS_DIR) if f.startswith(f"hybrid_model_{ticker}")]
            if not model_files:
                st.sidebar.warning(f"No hybrid models found for {ticker}")
            else:
                latest = sorted(model_files)[-1]
                with open(os.path.join(MODELS_DIR, latest), "rb") as f:
                    model = pickle.load(f)
                st.sidebar.success(f"Loaded: {latest}")
        except Exception as e:
            st.sidebar.error(f"Load error: {str(e)}")
    
    # Live Prediction
    if model and st.sidebar.checkbox("Show Live Signals"):
        stop_signal = st.sidebar.button("Stop Updates")
        live_display = st.empty()
        chart_display = st.empty()
        
        while not stop_signal:
            current_time = datetime.now(timezone)
            market_open = is_market_open()
            live_data = get_live_data(ticker, interval)
            
            if live_data is None:
                time_module.sleep(10)
                continue
                
            processed = preprocess_data(live_data.copy())
            if processed is None:
                time_module.sleep(10)
                continue
                
            with live_display.container():
                cols = st.columns(3)
                cols[0].metric("Time", current_time.strftime("%H:%M:%S"))
                
                # Current price
                current_price = live_data['Close'].iloc[-1]
                if isinstance(current_price, (pd.Series, np.ndarray)):
                    current_price = current_price[0] if len(current_price) > 0 else 0
                cols[1].metric("Price", f"${float(current_price):.2f}")
                
                if market_open:
                    try:
                        X_live = processed.iloc[-1:][model.feature_names]
                        pred = model.predict(X_live)[0]
                        proba = model.predict_proba(X_live)[0][1]
                        
                        confidence = float(proba if pred == 1 else (1-proba))
                        if pred == 1:
                            cols[2].success(f"BUY ({confidence:.1%} confidence)")
                        else:
                            cols[2].error(f"SELL ({confidence:.1%} confidence)")
                    except Exception as e:
                        cols[2].error(f"Prediction error: {str(e)}")
                else:
                    cols[2].warning("Market Closed")
            
            with chart_display.container():
                fig, ax = plt.subplots(figsize=(12,4))
                live_data['Close'].tail(20).plot(ax=ax)
                ax.set(xlabel="Time", ylabel="Price ($)", title="Recent Price Action")
                ax.grid()
                st.pyplot(fig)
            
            time_module.sleep(30 if interval != "1m" else 15)
            if stop_signal:
                st.sidebar.success("Updates stopped")
                break
except Exception as e:
    st.error(f"Application error: {str(e)}")