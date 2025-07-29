import streamlit as st
import base64
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor, IsolationForest
import joblib
import os

import streamlit as st
import base64

# --- Set background image from local file with a dark tint overlay ---
def set_bg_from_local(image_file):
    with open(image_file, "rb") as image:
        encoded = base64.b64encode(image.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: linear-gradient(rgba(0, 0, 0, 0.5), rgba(0, 0, 0, 0.5)),
                              url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# ✅ Call this with your image filename (must be in the same folder as app.py)
set_bg_from_local("food-wine-wallpaper-preview.jpg")

# --- Caching for data/model ---
@st.cache_data
def load_data():
    df = pd.read_csv("winequality.csv")
    df = df.dropna()
    return df

@st.cache_resource
def load_or_train_model(df, features, target, model_path="wine_rf_model.joblib"):
    if os.path.exists(model_path):
        model = joblib.load(model_path)
    else:
        X = df[features]
        y = df[target]
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        joblib.dump(model, model_path)
    return model

# --- Feature list ---
PHYSICOCHEMICAL_FEATURES = [
    "fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides",
    "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol"
]

# --- Load data/model ---
st.set_page_config(page_title="Wine Quality Analysis", layout="wide")
st.title("🍷 Wine Quality Analysis")
df = load_data()
model = load_or_train_model(df, PHYSICOCHEMICAL_FEATURES, "quality")

# --- Sidebar: User Input ---
st.sidebar.header("Input Wine Properties")
user_input = {}
for feature in PHYSICOCHEMICAL_FEATURES:
    min_val = float(df[feature].min())
    max_val = float(df[feature].max())
    mean_val = float(df[feature].mean())
    step = (max_val - min_val) / 100
    user_input[feature] = st.sidebar.slider(
        feature.title(), min_value=min_val, max_value=max_val, value=mean_val, step=step
    )
user_input_df = pd.DataFrame([user_input])

# --- Tabs ---
tabs = st.tabs(["Prediction", "Visualizations", "Feature Importance", "Outlier Detection"])

# --- Prediction Tab ---
with tabs[0]:
    st.header("🍷 Predict Wine Quality")
    pred_quality = model.predict(user_input_df)[0]
    pred_quality_rounded = int(round(pred_quality))
    if pred_quality_rounded <= 4:
        label = "Poor"
        color = "#e74c3c"
    elif pred_quality_rounded <= 6:
        label = "Average"
        color = "#f1c40f"
    else:
        label = "Good"
        color = "#27ae60"
    st.metric("Predicted Quality Score", f"{pred_quality:.2f}", label)
    st.markdown(
        f"<span style='color:{color}; font-size:1.5em;'>Quality: <b>{label}</b></span>",
        unsafe_allow_html=True,
    )
    # IQR-based warning
    warnings = []
    for feature in ["alcohol", "volatile acidity"]:
        Q1 = df[feature].quantile(0.25)
        Q3 = df[feature].quantile(0.75)
        if user_input[feature] < Q1 or user_input[feature] > Q3:
            warnings.append(f"{feature.title()} is outside the normal (IQR) range!")
    if warnings:
        st.warning(" ".join(warnings))
    st.write("---")
    st.write("#### Your Input:")
    st.dataframe(user_input_df)

# --- Visualizations Tab ---
with tabs[1]:
    st.header("📊 Visualizations (EDA)")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Histogram of Wine Quality")
        fig = px.histogram(
            df, 
            x="quality", 
            nbins=11, 
            color="quality", 
            color_discrete_sequence=px.colors.sequential.Blues
        )
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.subheader("Correlation Heatmap")
        corr = df[PHYSICOCHEMICAL_FEATURES + ["quality"]].corr()
        fig = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale="RdBu", origin="lower")
        st.plotly_chart(fig, use_container_width=True)
    st.subheader("Alcohol vs Quality (Scatter)")
    fig = px.scatter(df, x="alcohol", y="quality", color="quality", color_continuous_scale="Viridis")
    st.plotly_chart(fig, use_container_width=True)
    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Volatile Acidity vs Quality (Boxplot)")
        fig = px.box(df, x="quality", y="volatile acidity", color="quality", color_discrete_sequence=px.colors.sequential.Plasma)
        st.plotly_chart(fig, use_container_width=True)
    with col4:
        st.subheader("pH vs Quality (Boxplot)")
        fig = px.box(df, x="quality", y="pH", color="quality", color_discrete_sequence=px.colors.sequential.Plasma)
        st.plotly_chart(fig, use_container_width=True)

# --- Feature Importance Tab ---
with tabs[2]:
    st.header("🔬 Feature Importance")
    importances = model.feature_importances_
    feat_imp = pd.Series(importances, index=PHYSICOCHEMICAL_FEATURES).sort_values(ascending=True)
    st.subheader("Top Features (Random Forest)")
    fig = go.Figure(go.Bar(
        x=feat_imp.values[-10:],
        y=feat_imp.index[-10:],
        orientation='h',
        marker=dict(color=feat_imp.values[-10:], colorscale='Blues')
    ))
    fig.update_layout(xaxis_title="Importance", yaxis_title="Feature", height=400)
    st.plotly_chart(fig, use_container_width=True)
    st.write("Full Feature Importances:")
    st.dataframe(feat_imp.sort_values(ascending=False))

# --- Outlier Detection Tab ---
with tabs[3]:
    st.header("🚨 Outlier Detection")
    st.write("Detecting outliers using Isolation Forest and IQR method for 'alcohol' and 'volatile acidity'.")
    # Isolation Forest
    iso = IsolationForest(contamination=0.05, random_state=42)
    selected = df[["alcohol", "volatile acidity"]]
    outlier_pred = iso.fit_predict(selected)
    df_outliers = df.copy()
    df_outliers["outlier"] = outlier_pred
    outliers = df_outliers[df_outliers["outlier"] == -1]
    st.subheader("Outliers Detected (Isolation Forest)")
    st.dataframe(outliers[["alcohol", "volatile acidity", "quality"]])
    # IQR method
    st.subheader("IQR Outlier Ranges")
    for feature in ["alcohol", "volatile acidity"]:
        Q1 = df[feature].quantile(0.25)
        Q3 = df[feature].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        st.write(f"**{feature.title()}**: Normal range = [{lower:.2f}, {upper:.2f}]")
        outlier_rows = df[(df[feature] < lower) | (df[feature] > upper)]
        st.write(f"Number of outliers: {outlier_rows.shape[0]}")
    # User input warning
    user_warnings = []
    for feature in ["alcohol", "volatile acidity"]:
        Q1 = df[feature].quantile(0.25)
        Q3 = df[feature].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        if user_input[feature] < lower or user_input[feature] > upper:
            user_warnings.append(f"{feature.title()} is outside the normal (IQR) range!")
    if user_warnings:
        st.warning(" ".join(user_warnings))
    st.write("---")
    st.write("Raw Data (first 5 rows):")
    st.dataframe(df.head())
