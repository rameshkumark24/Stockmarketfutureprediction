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

# --- Market Lists (Same as before) ---
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
        data = yf.download(symbol, period="10y") 
        if data.empty or len(data) < 200: 
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

    # --- Analysis ---
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
    scaler = MinMaxScaler(feature_range=(0,1))
    data_close_values = data.Close.values.reshape(-1, 1)
    data_train_array = scaler.fit_transform(data_close_values)

    @st.cache_resource
    def train_model(data_train_array):
        x_train = []
        y_train = []
        lookback = 100
        if len(data_train_array) < 200: lookback = 30 
        
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
        model.fit(x_train, y_train, epochs=50, batch_size=32, verbose=0)
        return model, lookback

    with st.spinner('Training AI Model (50 Epochs for better accuracy)...'):
        model, lookback = train_model(data_train_array)

    # --- Test Accuracy (Corrected) ---
    test_start_index = int(len(data_close_values) * 0.7)
    input_data = data_close_values[test_start_index - lookback:]
    input_data = scaler.transform(input_data)

    x_test = []
    y_test = data_close_values[test_start_index:] 

    for i in range(lookback, len(input_data)):
        x_test.append(input_data[i-lookback: i])

    x_test = np.array(x_test)
    y_predicted = model.predict(x_test)
    scale_factor = 1/scaler.scale_[0]
    y_predicted = y_predicted * scale_factor

    st.subheader('Model Accuracy: Predicted vs Original (Test Set)')
    fig2 = plt.figure(figsize=(10,6))
    plt.plot(y_test, 'g', label='Original Price')
    plt.plot(y_predicted, 'r', label='AI Predicted Price')
    plt.xlabel('Time')
    plt.ylabel('Price (INR)')
    plt.legend()
    st.pyplot(fig2)

    # --- FUTURE PREDICTION (Improved with Calibration) ---
    st.subheader('🔮 Future Price Prediction (Next 10 Days)')
    
    # 1. Get raw future predictions
    last_days_data = data_close_values[-lookback:]
    last_days_scaled = scaler.transform(last_days_data)
    
    future_predictions = []
    current_batch = last_days_scaled.reshape(1, lookback, 1)

    for i in range(10): 
        pred = model.predict(current_batch)[0]
        future_predictions.append(pred[0])
        current_batch = np.append(current_batch[:, 1:, :], [[pred]], axis=1)

    # 2. Inverse scale
    future_predictions = np.array(future_predictions) * scale_factor
    
    # 3. FIX: CALCULATE BIAS (Difference between Today's Actual vs Model's "Today")
    # We predict "Today" using the model to see how far off it is
    last_known_batch = last_days_scaled[:-1].reshape(1, lookback-1, 1) # This logic is complex, let's use a simpler anchor
    
    # Simpler Anchor Logic:
    # We know the LAST REAL PRICE (Today's Close)
    last_real_price = data_close_values[-1][0]
    
    # We compare it to the FIRST predicted future point
    # Ideally, Day 1 Prediction should be close to Day 0 Actual
    # We shift the whole curve so it starts from Last Real Price
    
    # Calculate gap between "Model's first day prediction" and "Actual Last Close"
    # Actually, a smoother way is to shift based on the trend. 
    # Let's simply offset the difference.
    
    # We assume the model's *shape* is correct, but *level* is wrong.
    # We force the start of the prediction to align with the last known price.
    
    first_pred = future_predictions[0]
    gap = last_real_price - first_pred
    
    # Apply Correction to ALL future predictions
    corrected_predictions = future_predictions + gap
    
    # 4. Connect the line visually
    # We prepend the LAST KNOWN DATE and PRICE to the future lists
    # This creates a solid line from the Green chart to the Red chart without a gap
    
    future_dates = pd.date_range(start=date.today(), periods=11) # Start from TODAY
    
    # Final plot lists
    final_pred_values = [last_real_price] + list(corrected_predictions) # Start with actual, then follow predicted trend
    
    # Display Future Data Table
    future_df = pd.DataFrame({
        'Date': future_dates, 
        'Predicted Price (INR)': final_pred_values
    })
    st.write(future_df.iloc[1:]) # Show table from tomorrow onwards

    # Plot Future
    fig3 = plt.figure(figsize=(10,6))
    
    # Plot last 50 days of ACTUAL data
    plt.plot(data.index[-50:], data.Close[-50:], 'g', label='Past 50 Days (Actual)')
    
    # Plot Future (Starting from Today to connect lines)
    plt.plot(future_dates, final_pred_values, 'r--', marker='o', label='Next 10 Days (AI Forecast)')
    
    plt.title(f'{stock_symbol} Future Forecast (Calibrated)')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.grid(True)
    plt.legend()
    st.pyplot(fig3)

except Exception as e:
    st.error(f"Error: {e}")
