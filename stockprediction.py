import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

st.set_page_config(page_title="Stock Price Predictor", layout="centered")
st.title("📈 Stock Price Prediction")

uploaded_file = st.file_uploader("Upload your stock CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("Raw Data")
    st.write(df.head())

    if 'Date' not in df.columns or 'Close' not in df.columns:
        st.error("❌ The CSV must contain 'Date' and 'Close' columns.")
    else:
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
        df = df.dropna(subset=['Date']) 
        df = df.sort_values('Date')

        df['Date_ordinal'] = df['Date'].map(pd.Timestamp.toordinal)
        X = df['Date_ordinal'].values.reshape(-1, 1)
        y = df['Close'].values

        model = LinearRegression()
        model.fit(X, y)

        future_days = st.slider("Predict how many days into the future", 1, 100, 30)
        last_date = df['Date'].max()
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=future_days)
        future_ordinals = future_dates.map(pd.Timestamp.toordinal).values.reshape(-1, 1)
        predictions = model.predict(future_ordinals)

        st.subheader("📊 Stock Price Prediction Chart")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(df['Date'], y, label='Actual Prices', color='blue')
        ax.plot(future_dates, predictions, label='Predicted Prices', color='red')
        ax.set_xlabel("Date")
        ax.set_ylabel("Stock Price")
        ax.legend()
        st.pyplot(fig)

        st.subheader("📄 Predicted Prices Table")
        prediction_df = pd.DataFrame({
            "Date": future_dates,
            "Predicted Price": predictions
        })
        st.write(prediction_df)
