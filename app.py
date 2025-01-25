import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest

# Set Streamlit Page Configurations
st.set_page_config(page_title="AI Network Security", layout="wide", page_icon="🛡️")

# Custom Styling
st.markdown("""
    <style>
        .big-font { font-size:24px !important; font-weight: bold; }
        .stButton>button { background-color: #ff4b4b; color: white; font-size:18px; }
        .stNumberInput>div>input { font-size: 18px; }
    </style>
""", unsafe_allow_html=True)

# Sidebar - Upload Data
st.sidebar.header("📂 Upload Your CSV File")
uploaded_file = st.sidebar.file_uploader("Upload a dataset (CSV format)", type=["csv"])

# Global variables
df = None
df_scaled = None
anomaly_model = None
anomaly_detected = False  # Track if anomalies are detected

# Function to load and preprocess data
def load_data(file):
    df = pd.read_csv(file)
    st.write("### 🔍 Sample Data")
    st.dataframe(df.head())

    # Standardizing Data
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df.select_dtypes(include=np.number)), columns=df.select_dtypes(include=np.number).columns)

    return df, df_scaled

# Function to detect anomalies using Isolation Forest
def detect_anomalies(df_scaled):
    global df, anomaly_model, anomaly_detected

    st.write("### 🚨 Running Anomaly Detection...")
    anomaly_model = IsolationForest(contamination=0.1, random_state=42)
    df['Anomaly'] = anomaly_model.fit_predict(df_scaled)
    
    anomaly_detected = True  # Mark as detected

    # Count anomalies and normal records
    anomaly_count = (df['Anomaly'] == -1).sum()
    normal_count = (df['Anomaly'] == 1).sum()

    # Display results
    st.markdown(f"🔴 **Total Anomalies Detected:** <span style='color:red;font-size:22px;font-weight:bold'>{anomaly_count}</span>", unsafe_allow_html=True)
    st.markdown(f"✅ **Total Normal Records:** <span style='color:green;font-size:22px;font-weight:bold'>{normal_count}</span>", unsafe_allow_html=True)

    return anomaly_count, normal_count

# Function to visualize anomaly detection results
def plot_anomalies(anomalies, normal_records):
    categories = ["Anomalies", "Normal Records"]
    values = [anomalies, normal_records]
    colors = ["red", "blue"]

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.barplot(x=categories, y=values, palette=colors, ax=ax)

    ax.set_ylabel("Count", fontsize=14)
    ax.set_title("Anomaly Detection Overview", fontsize=16, fontweight='bold')
    
    for i, v in enumerate(values):
        ax.text(i, v + 5000, str(v), ha='center', fontsize=12, fontweight='bold', color='black')

    st.pyplot(fig)

# Function to predict congestion using Train-Test Split
def congestion_prediction():
    global df, df_scaled

    if not anomaly_detected:
        st.error("⚠️ Please run anomaly detection first!")
        return

    st.write("### 📊 Predicting Network Congestion...")
    
    try:
        # Ensure 'Anomaly' column exists
        if 'Anomaly' not in df.columns:
            st.error("⚠️ 'Anomaly' column is missing. Please run anomaly detection first.")
            return
        
        # Train-Test Split
        X_train, X_test, y_train, y_test = train_test_split(df_scaled, df['Anomaly'], test_size=0.2, random_state=42)
        
        st.success("✅ Congestion Prediction Successful!")
        st.write(f"Training Data: {X_train.shape[0]} rows")
        st.write(f"Testing Data: {X_test.shape[0]} rows")

    except Exception as e:
        st.error(f"❌ Error in Prediction: {e}")

# Main Content
st.title("🛡️ AI-Powered Network Security Dashboard")
st.markdown("Anomaly Detection and Congestion Prediction for Securing Networks")

# Check if file is uploaded
if uploaded_file is not None:
    df, df_scaled = load_data(uploaded_file)

    # Run Anomaly Detection
    if st.button("🚨 Run Anomaly Detection"):
        anomalies, normal_records = detect_anomalies(df_scaled)
        plot_anomalies(anomalies, normal_records)

    # Predict Congestion
    if st.button("📊 Predict Network Congestion"):
        congestion_prediction()
else:
    st.info("📂 Please upload a CSV file to proceed.")

