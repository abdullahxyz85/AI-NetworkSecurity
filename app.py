import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Streamlit Page Configuration
st.set_page_config(page_title="Network Anomaly Detection & Congestion Prediction", page_icon="📡", layout="wide")

# Custom Styling for Advanced UI
st.markdown("""
    <style>
        body {
            font-family: 'Roboto', sans-serif;
            background-color: #f4f6f9;
            color: #333333;
        }
        .stApp {
            margin: 0;
            padding: 0;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
        }
        .reportview-container {
            background-color: #f4f6f9;
            padding: 0;
        }
        .css-1d391kg {
            padding: 2rem;
            width: 70%;
        }
        h1 {
            font-size: 3rem;
            color: #1e2a47;
            font-weight: bold;
            text-align: center;
            margin-top: 2rem;
            letter-spacing: 1px;
        }
        h2 {
            font-size: 2.2rem;
            color: #3a4f78;
            font-weight: bold;
        }
        .stButton button {
            background-color: #007BFF;
            color: white;
            font-size: 18px;
            font-weight: bold;
            padding: 12px 30px;
            border-radius: 12px;
            border: none;
            box-shadow: 0px 8px 12px rgba(0, 0, 0, 0.1);
            transition: 0.3s ease;
        }
        .stButton button:hover {
            background-color: #0056b3;
            transform: translateY(-2px);
        }
        .stButton button:active {
            transform: translateY(0);
            box-shadow: 0px 5px 8px rgba(0, 0, 0, 0.1);
        }
        .stFileUploader {
            margin-top: 2rem;
            background-color: #ffffff;
            padding: 1rem;
            border-radius: 12px;
            box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
        }
        .stAlert {
            background-color: #f8d7da;
            color: #721c24;
            border-radius: 8px;
            padding: 1rem;
            margin-top: 20px;
            font-size: 1.1rem;
        }
        .sidebar .sidebar-content {
            background-color: #283593;
            color: #ffffff;
            padding: 20px;
            border-radius: 12px;
        }
        .stSelectbox div, .stTextInput div {
            margin-bottom: 15px;
        }
        .stBarChart {
            margin-top: 20px;
            background-color: #ffffff;
            border-radius: 10px;
            padding: 1rem;
            box-shadow: 0px 8px 12px rgba(0, 0, 0, 0.1);
        }
        .stDataFrame {
            margin-top: 2rem;
            border-radius: 10px;
            box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
            background-color: #ffffff;
        }
        .stAlert {
            font-size: 1.1rem;
        }
        .stPlotlyChart {
            background-color: #ffffff;
            border-radius: 10px;
            padding: 1rem;
            box-shadow: 0px 8px 10px rgba(0, 0, 0, 0.15);
        }
        .stTextInput div {
            margin-top: 1rem;
        }
    </style>
""", unsafe_allow_html=True)

# App Title
st.title("📡 **Network Anomaly Detection & Congestion Prediction** 🔍")
st.write("""
    **Step 1**: Upload your network traffic data.
    **Step 2**: Run **Anomaly Detection** to identify unusual patterns.
    **Step 3**: Predict **Network Congestion** based on traffic.
""")
st.sidebar.header("📑 **Navigation Panel**")
st.sidebar.markdown("""
    - **Step 1**: Upload your network traffic data.
    - **Step 2**: Run **Anomaly Detection** to identify unusual patterns.
    - **Step 3**: Predict **Network Congestion** based on traffic.
""")
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/4/46/Network-icon.svg/1200px-Network-icon.svg.png", width=100)

# File Uploader with Stylish Layout
uploaded_file = st.file_uploader("📂 **Upload Network Traffic Data (CSV)**", type=["csv"])

# Load and preprocess functions
@st.cache_data
def load_data(uploaded_file):
    df = pd.read_csv(uploaded_file)
    return df

@st.cache_data
def preprocess_data(df):
    df['Time'] = pd.to_datetime(df['Time'], unit='s')
    df['Hour'] = df['Time'].dt.hour
    label_encoder = LabelEncoder()
    df['Protocol_Encoded'] = label_encoder.fit_transform(df['Protocol'])
    features = ['Length', 'Protocol_Encoded', 'Hour', 'No.']
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df[features])
    return df, df_scaled

# Anomaly Detection
@st.cache_data
def detect_anomalies(df, df_scaled):
    model = IsolationForest(contamination=0.1, random_state=42)
    df['Anomaly'] = model.fit_predict(df_scaled)
    df['Anomaly'] = df['Anomaly'].map({1: 'Normal', -1: 'Anomaly'})
    return df

# Display results function
def display_results(df):
    st.write("### 🚨 **Anomaly Detection Results** 🚨")
    st.dataframe(df.head())
    anomaly_count = df['Anomaly'].value_counts()
    st.write("🛑 **Total Anomalies Detected**:", anomaly_count.get('Anomaly', 0))
    st.write("✅ **Total Normal Records**:", anomaly_count.get('Normal', 0))
    st.bar_chart(df['Anomaly'].value_counts(), use_container_width=True)

# Congestion Prediction
def congestion_prediction(df_scaled, df):
    if 'Anomaly' not in df.columns:
        st.error("Anomaly column is missing. Please run anomaly detection first.")
        return

    X_train, X_test, y_train, y_test = train_test_split(df_scaled, df['Anomaly'], test_size=0.2, random_state=42)
    model = IsolationForest(contamination=0.05, random_state=42)
    model.fit(X_train)
    y_pred = model.predict(X_test)
    y_pred = np.where(y_pred == 1, "Normal", "Congested")
    
    st.subheader("📈 **Congestion Prediction Results** 📉")
    st.write("📊 **Confusion Matrix**:")
    st.write(confusion_matrix(y_test, y_pred))
    st.text(classification_report(y_test, y_pred))
    
    congestion_fig = px.pie(names=['Normal', 'Congested'], values=[np.sum(y_pred == 'Normal'), np.sum(y_pred == 'Congested')], title="🚦 Traffic Congestion Distribution 🌐")
    st.plotly_chart(congestion_fig, use_container_width=True)

# Main Execution
if uploaded_file:
    df = load_data(uploaded_file)
    st.subheader("🔍 **Preview of Uploaded Data**")
    st.dataframe(df.head())

    df, df_scaled = preprocess_data(df)
    
    # Session state to store the 'Anomaly' column after anomaly detection
    if st.button("🚨 **Run Anomaly Detection** 🚨"):
        df = detect_anomalies(df, df_scaled)
        st.session_state.df = df  # Save df to session state
        display_results(df)
    
    if st.button("📊 **Predict Network Congestion** 📉"):
        # Ensure Anomaly column is available in the session state
        if 'df' in st.session_state and 'Anomaly' in st.session_state.df.columns:
            congestion_prediction(df_scaled, st.session_state.df)
        else:
            st.error("Anomaly column is missing. Please run anomaly detection first.")
