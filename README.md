📈 Indian Stock Market Predictor (AI-Powered)

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B)
![TensorFlow](https://img.shields.io/badge/TensorFlow-LSTM-orange)
![License](https://img.shields.io/badge/License-MIT-green)

A comprehensive Web Application that predicts stock prices for the **Indian Stock Market (NSE & BSE)** using Deep Learning (**LSTM**). It visualizes historical trends and forecasts the stock price for the **next 10 days** with high accuracy calibration.

## 🚀 Live Demo
👉 **[Click here to try the App](https://stockmarketfutureprediction-aob6kekuk4vfywpdytbwvd.streamlit.app/)**

## 📌 Features

* **Multi-Market Support:** Pre-loaded with **NIFTY 50** (NSE) and **SENSEX 30** (BSE) stocks.
* **Custom Ticker Search:** Analyzes *any* stock listed on Yahoo Finance (e.g., ZOMATO, PAYTM, TATAPOWER).
* **Deep Learning Model:** Uses **Long Short-Term Memory (LSTM)** neural networks for accurate time-series forecasting.
* **Future Forecasting:** Predicts stock prices for the **Next 10 Days** with a visual trend line.
* **Smart Calibration (Anchor & Shift):** Implements a post-processing logic to align AI predictions with the latest real-time closing price, ensuring zero-gap continuous visualization.
* **Technical Analysis:** Interactive charts for **50-Day** and **200-Day Moving Averages** (MA).
* **Performance Optimized:** Uses `Streamlit Caching` to train models once and reuse them for instant reloading.

## 🛠️ Tech Stack

* **Frontend:** Streamlit (Python)
* **Machine Learning:** TensorFlow, Keras (LSTM Layers)
* **Data Processing:** Pandas, NumPy, Scikit-Learn (MinMax Scaling)
* **Data Source:** Yahoo Finance (`yfinance`) API
* **Visualization:** Matplotlib

## 📂 Project Structure

```bash
├── app.py                # Main application code (UI + Model Logic)
├── requirements.txt      # List of dependencies
├── README.md             # Project documentation
└── .gitignore            # Files to ignore (e.g., venv, __pycache__)

```

## ⚙️ Installation & Run Locally

1. **Clone the Repository**
```bash
git clone [https://github.com/rameshkumark24/Stockmarketfutureprediction.git](https://github.com/rameshkumark24/Stockmarketfutureprediction.git)
cd Stockmarketfutureprediction

```


2. **Create a Virtual Environment (Optional but Recommended)**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

```


3. **Install Dependencies**
```bash
pip install -r requirements.txt

```


4. **Run the App**
```bash
streamlit run app.py

```



## 🧠 How It Works (The AI Part)

1. **Data Fetching:** The app dynamically fetches 10+ years of historical daily data using `yfinance`.
2. **Preprocessing:** Data is normalized between 0 and 1 using `MinMaxScaler` to help the LSTM converge faster.
3. **Model Architecture:**
* **LSTM Layers:** Captures long-term dependencies and patterns in price movements.
* **Dropout Layers:** Prevents overfitting to ensure the model generalizes well to new data.
* **Dense Output Layer:** Predicts the next numerical value.


4. **Forecasting Logic:**
* The model uses the last 100 days of data to predict Day 101.
* This predicted value is fed back into the input to predict Day 102, and so on (Recursive Forecasting).


5. **Bias Correction (Anchor Strategy):** * Raw LSTM predictions often capture the *trend* but may have a slight offset in *value*.
* The app calculates the difference between the "Model's Today" and "Actual Today" prices.
* It shifts the entire future prediction curve to anchor it to the Last Traded Price (LTP), ensuring a realistic and seamless chart connection.



## 🤝 Contributing

Contributions are welcome!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/NewFeature`)
3. Commit your Changes (`git commit -m 'Add some NewFeature'`)
4. Push to the Branch (`git push origin feature/NewFeature`)
5. Open a Pull Request

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.

---

*Created by [Ramesh Kumar*](https://www.google.com/search?q=https://github.com/rameshkumark24)

```

```
