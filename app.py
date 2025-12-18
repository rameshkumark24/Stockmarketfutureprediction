import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import streamlit as st
from datetime import date
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM

# --- Page Config ---
st.set_page_config(page_title="Indian Stock Predictor", layout="wide")
st.header('📈 Indian Stock Market Predictor')

# --- Market Lists ---
# NIFTY 50 (NSE) - Tickers end with .NS
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

# SENSEX 30 (BSE) - Tickers end with .BO
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
    user_input = st.sidebar.text_input('Enter Custom Symbol (e.g., ZOMATO)', 'ZOMATO')
    
    # FIX 1: Force Uppercase and Remove Spaces
    user_input = user_input.upper().strip() 
    
    # Auto-fix suffix if missing
    if not user_input.endswith('.NS') and not user_input.endswith('.BO'):
        stock_symbol = f"{user_input}{suffix}"
    else:
        stock_symbol = user_input
    if not user_input.endswith('.NS') and not user_input.endswith('.BO'):
        stock_symbol = f"{user_input}{suffix}"
    else:
        stock_symbol = user_input
else:
    stock_symbol = stock_dict[selected_stock_name]

# --- Data Fetching ---
start = '2015-01-01'
end = date.today().strftime("%Y-%m-%d")

st.write(f"Fetching data for: **{stock_symbol}** from {start} to {end}")

@st.cache_data
def load_data(symbol, start, end):
    data = yf.download(symbol, start, end)
    return data

try:
    data = load_data(stock_symbol, start, end)
    
    if data.empty:
        st.error(f"No data found for {stock_symbol}. Try checking the symbol.")
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

    # --- LSTM Model ---
    scaler = MinMaxScaler(feature_range=(0,1))
    data_train = pd.DataFrame(data.Close[0: int(len(data)*0.80)])
    data_test = pd.DataFrame(data.Close[int(len(data)*0.80): len(data)])
    data_train_array = scaler.fit_transform(data_train)

    @st.cache_resource
    def train_model(data_train_array):
        x_train = []
        y_train = []
        for i in range(100, data_train_array.shape[0]):
            x_train.append(data_train_array[i-100: i])
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
        model.fit(x_train, y_train, epochs=5, batch_size=32, verbose=0)
        return model

    with st.spinner('Training AI Model...'):
        model = train_model(data_train_array)

    # --- Predictions ---
    pas_100_days = data_train.tail(100)
    final_df = pd.concat([pas_100_days, data_test], ignore_index=True)
    input_data = scaler.transform(final_df)

    x_test = []
    y_test = []
    for i in range(100, input_data.shape[0]):
        x_test.append(input_data[i-100: i])
        y_test.append(input_data[i, 0])

    x_test, y_test = np.array(x_test), np.array(y_test)
    y_predicted = model.predict(x_test)
    
    scale_factor = 1/scaler.scale_[0]
    y_predicted = y_predicted * scale_factor
    y_test = y_test * scale_factor

    st.subheader('Prediction vs Original')
    fig2 = plt.figure(figsize=(10,6))
    plt.plot(y_test, 'g', label='Original Price')
    plt.plot(y_predicted, 'r', label='AI Predicted Price')
    plt.xlabel('Time')
    plt.ylabel('Price (INR)')
    plt.legend()
    st.pyplot(fig2)

except Exception as e:
    st.error(f"Error: {e}")
