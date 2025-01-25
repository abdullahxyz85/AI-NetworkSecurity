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

# Custom Styling for UI Enhancements
st.markdown("""
    <style>
        .reportview-container { background: #f5f7fa; }
        .css-1d391kg { padding: 2rem; }
        h1 { color: #333366; font-size: 3rem; text-align: center; font-weight: bold; }
        h2 { color: #1E3A8A; font-size: 2rem; font-weight: bold; }
        .stButton button { 
            background-color: #4CAF50; 
            color: white; 
            font-size: 16px; 
            padding: 10px 20px;
            border-radius: 10px;
            box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
        }
        .stButton button:hover { 
            background-color: #45a049; 
            transform: scale(1.05); 
            transition: all 0.3s ease-in-out;
        }
        .sidebar .sidebar-content { background-color: #1E3A8A; color: white; padding: 20px; }
        .stSelectbox div, .stTextInput div { margin-bottom: 20px; }
        .stAlert { background-color: #f8d7da; color: #721c24; border-radius: 10px; padding: 10px; }
        .stBarChart { margin-top: 20px; }
    </style>
""", unsafe_allow_html=True)

# App Title
st.title("📡 **Network Anomaly Detection & Congestion Prediction** 🔍")
st.write("Upload your network traffic data to detect anomalies and predict congestion patterns. 🚨📊")

# Sidebar for Navigation
st.sidebar.header("📑 **Navigation Panel**")
st.sidebar.markdown("""
    - **Step 1**: Upload your network traffic data.
    - **Step 2**: Run **Anomaly Detection** to identify unusual patterns.
    - **Step 3**: Predict **Network Congestion** for traffic analysis.
""")
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/4/46/Network-icon.svg", width=100)

# File Uploader
uploaded_file = st.file_uploader("📂 **Upload Network Traffic Data (CSV)**", type=["csv"])

# Load data function
@st.cache_data
def load_data(uploaded_file):
    df = pd.read_csv(uploaded_file)
    return df

# Preprocess data
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

# Anomaly Detection function
@st.cache_data
def detect_anomalies(df, df_scaled):
    model = IsolationForest(contamination=0.1, random_state=42)
    df['Anomaly'] = model.fit_predict(df_scaled)
    df['Anomaly'] = df['Anomaly'].map({1: 'Normal', -1: 'Anomaly'})
    return df

# Display results with improved UI
def display_results(df):
    st.write("### 🚨 **Anomaly Detection Results** 🚨")
    st.write(df.head()) 
    anomaly_count = df['Anomaly'].value_counts()
    st.write("🛑 **Total Anomalies Detected**:", anomaly_count.get('Anomaly', 0))
    st.write("✅ **Total Normal Records**:", anomaly_count.get('Normal', 0))
    st.bar_chart(df['Anomaly'].value_counts(), use_container_width=True)

# Congestion Prediction function
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
