import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import streamlit as st
from datetime import date
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM

# --- Indian Market Customizations ---
st.set_page_config(page_title="Nifty 50 Predictor", layout="wide")
st.header('📈 Indian Stock Market Predictor')

# Sidebar for Indian Stock Selection
st.sidebar.subheader('Select Stock')
nifty_50 = [
    'RELIANCE', 'TCS', 'HDFCBANK', 'ICICIBANK', 'INFY', 'SBIN', 'BHARTIARTL', 'ITC', 
    'LTIM', 'TATAMOTORS', 'LT', 'HCLTECH', 'AXISBANK', 'MARUTI', 'TITAN', 'ZOMATO'
]
stock_selection = st.sidebar.selectbox("Choose a NIFTY 50 Stock", ['Custom'] + nifty_50)

if stock_selection == 'Custom':
    user_input = st.sidebar.text_input('Enter Custom Symbol (e.g., ZOMATO)', 'ZOMATO')
else:
    user_input = stock_selection

# Auto-append .NS
if not user_input.endswith('.NS') and not user_input.endswith('.BO'):
    stock_symbol = f"{user_input}.NS"
else:
    stock_symbol = user_input

# --- FIX 1: USE LIVE DATES ---
start = '2015-01-01'
end = date.today().strftime("%Y-%m-%d") # This gets TODAY'S date automatically

st.write(f"Fetching data for: **{stock_symbol}** from {start} to {end}")

# --- FIX 2: CACHING THE DATA FETCHING ---
@st.cache_data
def load_data(symbol, start, end):
    data = yf.download(symbol, start, end)
    return data

try:
    data = load_data(stock_symbol, start, end)
    
    if data.empty:
        st.error("No data found. Please check the stock symbol.")
        st.stop()

    st.subheader('Stock Data (INR)')
    st.write(data.tail()) # Shows the most recent dates (Up to yesterday/today)

    # Prepare Data
    data_train = pd.DataFrame(data.Close[0: int(len(data)*0.80)])
    data_test = pd.DataFrame(data.Close[int(len(data)*0.80): len(data)])
    
    scaler = MinMaxScaler(feature_range=(0,1))

    st.subheader('Price vs Moving Averages')
    ma_50 = data.Close.rolling(50).mean()
    ma_200 = data.Close.rolling(200).mean()
    fig1 = plt.figure(figsize=(10,6))
    plt.plot(data.Close, 'g', label='Close Price')
    plt.plot(ma_50, 'r', label='50 Day MA')
    plt.plot(ma_200, 'b', label='200 Day MA')
    plt.legend()
    st.pyplot(fig1)

    # --- FIX 3: CACHING THE MODEL TRAINING ---
    # This prevents the "Training..." spinner from appearing if you just switch tabs or reload
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

    # Scale data
    data_train_array = scaler.fit_transform(data_train)
    
    with st.spinner('Training AI Model... (Only happens once per stock)'):
        model = train_model(data_train_array)

    # Prediction Logic
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
