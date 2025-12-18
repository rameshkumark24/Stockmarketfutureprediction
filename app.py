import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import streamlit as st
from datetime import date, timedelta
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM

# --- Page Config ---
st.set_page_config(page_title="Indian Stock Predictor", layout="wide")
st.header('📈 Indian Stock Market Predictor')

# --- Market Lists ---
# NIFTY 50 (NSE)
nifty_50_dict = {
    'ADANIENT': 'ADANIENT.NS', 'ADANIPORTS': 'ADANIPORTS.NS', 'APOLLOHOSP': 'APOLLOHOSP.NS', 
    'ASIANPAINT': 'ASIANPAINT.NS', 'AXISBANK': 'AXISBANK.NS', 'BAJAJ-AUTO': 'BAJAJ-AUTO.NS', 
    'BAJFINANCE': 'BAJFINANCE.NS', 'BAJAJFINSV': 'BAJAJFINSV.NS', 'BEL': 'BEL.NS', 
    'BHARTIARTL': 'BHARTIARTL.NS', 'BPCL': 'BPCL.NS', 'BRITANNIA': 'BRITANNIA.NS', 
    'CIPLA': 'CIPLA.NS', 'COALINDIA': 'COALINDIA.NS', 'DIVISLAB': 'DIVISLAB.NS', 
    'DRREDDY': 'DRREDDY.NS', 'EICHERMOT': 'EICHERMOT.NS', 'GRASIM': 'GRASIM.NS', 
    'HCLTECH': 'HCLTECH.NS', 'HDFCBANK': 'HDFCBANK.NS', 'HDFCLIFE': 'HDFCLIFE.NS', 
    'HEROMOTOCO': 'HEROMOTOCO.NS', 'HINDALCO': 'HINDALCO.NS', 'HINDUNILVR': 'HINDUNILVR.NS', 
    'ICICIBANK': 'ICICIBANK.NS', 'INDUSINDBK': 'INDUSINDBK.NS', 'INFY': 'INFY.NS', 
    'ITC': 'ITC.NS', 'JSWSTEEL': 'JSWSTEEL.NS', 'KOTAKBANK': 'KOTAKBANK.NS', 
    'LT': 'LT.NS', 'LTIM': 'LTIM.NS', 'M&M': 'M&M.NS', 'MARUTI': 'MARUTI.NS', 
    'NESTLEIND': 'NESTLEIND.NS', 'NTPC': 'NTPC.NS', 'ONGC': 'ONGC.NS', 
    'POWERGRID': 'POWERGRID.NS', 'RELIANCE': 'RELIANCE.NS', 'SBILIFE': 'SBILIFE.NS', 
    'SBIN': 'SBIN.NS', 'SUNPHARMA': 'SUNPHARMA.NS', 'TATACONSUM': 'TATACONSUM.NS', 
    'TATAMOTORS': 'TATAMOTORS.NS', 'TATASTEEL': 'TATASTEEL.NS', 'TCS': 'TCS.NS', 
    'TECHM': 'TECHM.NS', 'TITAN': 'TITAN.NS', 'ULTRACEMCO': 'ULTRACEMCO.NS', 'WIPRO': 'WIPRO.NS'
}

# SENSEX 30 (BSE)
sensex_30_dict = {
    'ASIANPAINT': 'ASIANPAINT.BO', 'AXISBANK': 'AXISBANK.BO', 'BAJAJ-AUTO': 'BAJAJ-AUTO.BO',
    'BAJFINANCE': 'BAJFINANCE.BO', 'BAJAJFINSV': 'BAJAJFINSV.BO', 'BHARTIARTL': 'BHARTIARTL.BO',
    'DRREDDY': 'DRREDDY.BO', 'HCLTECH': 'HCLTECH.BO', 'HDFCBANK': 'HDFCBANK.BO',
    'HINDUNILVR': 'HINDUNILVR.BO', 'ICICIBANK': 'ICICIBANK.BO', 'INDUSINDBK': 'INDUSINDBK.BO',
    'INFY': 'INFY.BO', 'ITC': 'ITC.BO', 'KOTAKBANK': 'KOTAKBANK.BO', 'LT': 'LT.BO',
    'M&M': 'M&M.BO', 'MARUTI': 'MARUTI.BO', 'NESTLEIND': 'NESTLEIND.BO', 'NTPC': 'NTPC.BO',
    'ONGC': 'ONGC.BO', 'POWERGRID': 'POWERGRID.BO', 'RELIANCE': 'RELIANCE.BO',
    'SBIN': 'SBIN.BO', 'SUNPHARMA': 'SUNPHARMA.BO', 'TATASTEEL': 'TATASTEEL.BO',
    'TCS': 'TCS.BO', 'TECHM': 'TECHM.BO', 'TITAN': 'TITAN.BO', 'ULTRACEMCO': 'ULTRACEMCO.BO',
    'WIPRO': 'WIPRO.BO'
}

# --- Sidebar ---
st.sidebar.subheader('Select Market')
market_choice = st.sidebar.radio("Market Index", ('NIFTY 50 (NSE)', 'SENSEX 30 (BSE)'))

if market_choice == 'NIFTY 50 (NSE)':
    stock_dict = nifty_50_dict
    suffix = '.NS'
else:
    stock_dict = sensex_30_dict
    suffix = '.BO'

st.sidebar.subheader('Select Stock')
selected_stock_name = st.sidebar.selectbox(f"Choose a {market_choice} Stock", ['Custom'] + list(stock_dict.keys()))

if selected_stock_name == 'Custom':
    user_input = st.sidebar.text_input('Enter Custom Symbol (e.g., TATASTEEL)', 'TATASTEEL')
    user_input = user_input.upper().strip() 
    if not user_input.endswith('.NS') and not user_input.endswith('.BO'):
        stock_symbol = f"{user_input}{suffix}"
    else:
        stock_symbol = user_input
else:
    stock_symbol = stock_dict[selected_stock_name]

# --- Data Fetching ---
st.write(f"Fetching data for: **{stock_symbol}**")

@st.cache_data
def load_data(symbol):
    try:
        # '10y' is better for training models (more relevant recent data)
        data = yf.download(symbol, period="10y") 
        if data.empty or len(data) < 200: 
            # Fallback for new stocks (Zomato/Paytm)
            data = yf.download(symbol, period="max")
        return data
    except Exception as e:
        return pd.DataFrame()

try:
    data = load_data(stock_symbol)
    
    if data.empty:
        st.error(f"❌ No data found for {stock_symbol}. Please check if the ticker is correct.")
        st.stop()

    st.subheader(f'{stock_symbol} - Stock Data (INR)')
    st.write(data.tail())

    # --- Analysis (MA) ---
    st.subheader('Price vs Moving Averages')
    ma_50 = data.Close.rolling(50).mean()
    ma_200 = data.Close.rolling(200).mean()
    fig1 = plt.figure(figsize=(10,6))
    plt.plot(data.Close, 'g', label='Close Price')
    plt.plot(ma_50, 'r', label='50 Day MA')
    plt.plot(ma_200, 'b', label='200 Day MA')
    plt.title(f'{stock_symbol} Price History')
    plt.legend()
    st.pyplot(fig1)

    # --- LSTM Model Training ---
    # Fix: Train on ALL available data for better scaling accuracy
    scaler = MinMaxScaler(feature_range=(0,1))
    
    # Use 'values' to convert pandas Series to numpy array to avoid index issues
    data_close_values = data.Close.values.reshape(-1, 1)
    
    data_train_array = scaler.fit_transform(data_close_values)

    @st.cache_resource
    def train_model(data_train_array):
        x_train = []
        y_train = []
        
        # Lookback Period
        lookback = 100
        if len(data_train_array) < 200:
             lookback = 30 
        
        for i in range(lookback, len(data_train_array)):
            x_train.append(data_train_array[i-lookback: i])
            y_train.append(data_train_array[i, 0])
            
        x_train, y_train = np.array(x_train), np.array(y_train)

        model = Sequential()
        model.add(LSTM(units=50, activation='relu', return_sequences=True, input_shape=(x_train.shape[1], 1)))
        model.add(Dropout(0.2))
        model.add(LSTM(units=60, activation='relu', return_sequences=True))
        model.add(Dropout(0.3))
        model.add(LSTM(units=80, activation='relu', return_sequences=True))
        model.add(Dropout(0.4))
        model.add(LSTM(units=120, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(units=1))
        
        model.compile(optimizer='adam', loss='mean_squared_error')
        # Increased epochs to 50 for better accuracy (cached so it's fast after 1st run)
        model.fit(x_train, y_train, epochs=50, batch_size=32, verbose=0)
        
        return model, lookback

    with st.spinner('Training AI Model (50 Epochs for better accuracy)...'):
        model, lookback = train_model(data_train_array)

    # --- Test Accuracy (Past vs Predicted) ---
    # We take the last 30% of data to visualize how well the model learned
    test_start_index = int(len(data_close_values) * 0.7)
    
    # Need 'lookback' days before the test start to predict the first test day
    input_data = data_close_values[test_start_index - lookback:]
    input_data = scaler.transform(input_data)

    x_test = []
    y_test = data_close_values[test_start_index:] # Actual future values we want to predict

    for i in range(lookback, len(input_data)):
        x_test.append(input_data[i-lookback: i])

    x_test = np.array(x_test)
    
    # Predict
    y_predicted = model.predict(x_test)
    
    # Inverse Scale
    scale_factor = 1/scaler.scale_[0]
    y_predicted = y_predicted * scale_factor
    # y_test is already in original scale, no need to inverse

    st.subheader('Model Accuracy: Predicted vs Original (Test Set)')
    fig2 = plt.figure(figsize=(10,6))
    plt.plot(y_test, 'g', label='Original Price')
    plt.plot(y_predicted, 'r', label='AI Predicted Price')
    plt.xlabel('Time')
    plt.ylabel('Price (INR)')
    plt.legend()
    st.pyplot(fig2)

    # --- FUTURE PREDICTION (Next 10 Days) ---
    st.subheader('🔮 Future Price Prediction (Next 10 Days)')
    
    # Get the last 'lookback' days of data to predict tomorrow
    last_days_data = data_close_values[-lookback:]
    last_days_scaled = scaler.transform(last_days_data)
    
    future_predictions = []
    current_batch = last_days_scaled.reshape(1, lookback, 1) # Reshape for LSTM [1, 100, 1]

    for i in range(10): # Predict 10 days
        pred = model.predict(current_batch)[0] # Get prediction (scaled)
        future_predictions.append(pred[0])
        
        # Update batch: remove first day, add new predicted day
        current_batch = np.append(current_batch[:, 1:, :], [[pred]], axis=1)

    # Inverse scale future predictions
    future_predictions = np.array(future_predictions) * scale_factor
    
    # Create future dates
    future_dates = pd.date_range(start=date.today() + timedelta(days=1), periods=10)
    
    # Display Future Data
    future_df = pd.DataFrame({'Date': future_dates, 'Predicted Price (INR)': future_predictions})
    st.write(future_df)
    
    # Plot Future
    fig3 = plt.figure(figsize=(10,6))
    plt.plot(data.index[-50:], data.Close[-50:], 'g', label='Past 50 Days')
    plt.plot(future_dates, future_predictions, 'r--', marker='o', label='Next 10 Days Prediction')
    plt.title(f'{stock_symbol} Future Forecast')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    st.pyplot(fig3)

except Exception as e:
    st.error(f"Error: {e}")
