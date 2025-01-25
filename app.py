import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Streamlit Page Configuration
st.set_page_config(page_title="Network Anomaly Detection & Congestion Prediction", page_icon="📶", layout="wide")

# Custom Styling
st.markdown("""
    <style>
        .reportview-container { background: #f5f7fa; }
        .css-1d391kg { padding: 2rem; }
        h1 { color: #333366; }
    </style>
""", unsafe_allow_html=True)

# App Title
st.title("📶 Network Anomaly Detection & Congestion Prediction")
st.write("Upload network traffic data to detect anomalies and predict congestion.")

# File Uploader
uploaded_file = st.file_uploader("📂 Upload Network Traffic Data (CSV)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("🔍 Preview of Uploaded Data")
    st.dataframe(df.head())

    # Preprocessing
    df['Time'] = pd.to_datetime(df['Time'], unit='s')
    df['Hour'] = df['Time'].dt.hour
    label_encoder = LabelEncoder()
    df['Protocol_Encoded'] = label_encoder.fit_transform(df['Protocol'])
    features = ['Length', 'Protocol_Encoded', 'Hour', 'No.']
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df[features])
    
    # Anomaly Detection using Isolation Forest
    if st.button("🚨 Run Anomaly Detection"):
        model = IsolationForest(contamination=0.1, random_state=42)
        df['Anomaly'] = model.fit_predict(df_scaled)
        df['Anomaly'] = df['Anomaly'].map({1: 'Normal', -1: 'Anomaly'})
        
        st.subheader("🚀 Anomaly Detection Results")
        st.write(f"Total Anomalies Detected: {df[df['Anomaly'] == 'Anomaly'].shape[0]}")
        
        fig = px.histogram(df, x='Protocol', color='Anomaly', title="Anomalies by Protocol", barmode='group')
        st.plotly_chart(fig, use_container_width=True)
    
    # Congestion Prediction Model
    if st.button("📊 Predict Network Congestion"):
        X_train, X_test, y_train, y_test = train_test_split(df_scaled, df['Anomaly'], test_size=0.2, random_state=42)
        model = IsolationForest(contamination=0.05, random_state=42)
        model.fit(X_train)
        y_pred = model.predict(X_test)
        y_pred = np.where(y_pred == 1, "Normal", "Congested")
        
        st.subheader("📈 Congestion Prediction Results")
        st.write("Confusion Matrix:")
        st.write(confusion_matrix(y_test, y_pred))
        st.text(classification_report(y_test, y_pred))
        
        congestion_fig = px.pie(names=['Normal', 'Congested'], values=[np.sum(y_pred == 'Normal'), np.sum(y_pred == 'Congested')], title="Traffic Congestion Distribution")
        st.plotly_chart(congestion_fig, use_container_width=True)
