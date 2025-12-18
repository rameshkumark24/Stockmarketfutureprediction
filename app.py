import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM

# --- UI Setup ---
st.header('Stock Market Predictor')

stock = st.text_input('Enter Stock Symbol', 'ZOMATO.NS')
start = '2015-01-01'
end = '2025-01-01'

data = yf.download(stock, start, end)

st.subheader('Stock Data')
st.write(data)

data_train = pd.DataFrame(data.Close[0: int(len(data)*0.80)])
data_test = pd.DataFrame(data.Close[int(len(data)*0.80): len(data)])

scaler = MinMaxScaler(feature_range=(0,1))

# --- Moving Averages (Your existing logic) ---
st.subheader('Price vs MA50')
ma_50_days = data.Close.rolling(50).mean()
fig1 = plt.figure(figsize=(8,6))
plt.plot(ma_50_days, 'r', label='MA50')
plt.plot(data.Close, 'g', label='Price')
plt.legend()
st.pyplot(fig1)

st.subheader('Price vs MA50 vs MA100')
ma_100_days = data.Close.rolling(100).mean()
fig2 = plt.figure(figsize=(8,6))
plt.plot(ma_50_days, 'r', label='MA50')
plt.plot(ma_100_days, 'b', label='MA100')
plt.plot(data.Close, 'g', label='Price')
plt.legend()
st.pyplot(fig2)

# --- Prediction Model (The "Resume Worthy" Part) ---
# Prepare data for LSTM
pas_100_days = data_train.tail(100)
final_df = pd.concat([pas_100_days, data_test], ignore_index=True)
input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100: i])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)

# Build LSTM Model (Simplified for deployment speed)
model = Sequential()
model.add(LSTM(units=50, activation='relu', return_sequences=True, input_shape=(x_test.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=60, activation='relu', return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(units=80, activation='relu', return_sequences=True))
model.add(Dropout(0.4))
model.add(LSTM(units=120, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=1))

# Compile and Train (In a real app, load a pre-trained model to save time)
model.compile(optimizer='adam', loss='mean_squared_error')
with st.spinner('Training Model... (this may take a minute)'):
    model.fit(x_test, y_test, epochs=5, batch_size=32, verbose=0) # Low epochs for demo speed

# Predict
predictions = model.predict(x_test)
scale_factor = 1/scaler.scale_[0]
predictions = predictions * scale_factor
y_test = y_test * scale_factor

# Plot Predictions
st.subheader('Original Price vs Predicted Price')
fig4 = plt.figure(figsize=(8,6))
plt.plot(y_test, 'g', label='Original Price')
plt.plot(predictions, 'r', label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig4)
