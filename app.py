import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import dask.dataframe as dd
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
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/4/46/Network-icon.svg/1200px-Network-icon.svg.png", width=100)

# File Uploader
uploaded_file = st.file_uploader("📂 **Upload Network Traffic Data (CSV)**", type=["csv"])

# Use Dask for faster file loading
@st.cache
def load_data(uploaded_file):
    # Load the data using Dask for faster performance
    df = dd.read_csv(uploaded_file)
    return df.compute()  # Convert Dask dataframe to Pandas for further processing

# Preprocess data
@st.cache
def preprocess_data(df):
    # Convert 'Time' to datetime and extract the hour
    df['Time'] = pd.to_datetime(df['Time'], unit='s')
    df['Hour'] = df['Time'].dt.hour
    
    # Encode 'Protocol' column
    label_encoder = LabelEncoder()
    df['Protocol_Encoded'] = label_encoder.fit_transform(df['Protocol'])
    
    # Select features for scaling
    features = ['Length', 'Protocol_Encoded', 'Hour', 'No.']
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df[features])
    
    return df, df_scaled

# Anomaly Detection using Isolation Forest
@st.cache
def detect_anomalies(df, df_scaled):
    # Using Isolation Forest to detect anomalies
    model = IsolationForest(contamination=0.1, random_state=42)
    df['Anomaly'] = model.fit_predict(df_scaled)
    df['Anomaly'] = df['Anomaly'].map({1: 'Normal', -1: 'Anomaly'})
    return df

# Display results with improved UI
def display_results(df):
    st.write("### 🚨 **Anomaly Detection Results** 🚨")
    st.write(df.head())  # Display first few rows of the dataframe
    
    # Count anomalies and normal records
    anomaly_count = df['Anomaly'].value_counts()
    st.write("🛑 **Total Anomalies Detected**:", anomaly_count.get('Anomaly', 0))
    st.write("✅ **Total Normal Records**:", anomaly_count.get('Normal', 0))
    
    # Display a bar chart of anomaly counts
    st.bar_chart(df['Anomaly'].value_counts(), use_container_width=True)

# Congestion Prediction using Isolation Forest (for model training and testing)
def congestion_prediction(df_scaled, df):
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

if uploaded_file:
    df = load_data(uploaded_file)
    st.subheader("🔍 **Preview of Uploaded Data**")
    st.dataframe(df.head())

    # Preprocess the data
    df, df_scaled = preprocess_data(df)
    
    # Anomaly Detection
    if st.button("🚨 **Run Anomaly Detection** 🚨"):
        df = detect_anomalies(df, df_scaled)
        st.subheader("🚀 **Anomaly Detection Results** 🚀")
        st.write(f"**Total Anomalies Detected:** {df[df['Anomaly'] == 'Anomaly'].shape[0]} 🛑")
        
        fig = px.histogram(df, x='Protocol', color='Anomaly', title="📊 **Anomalies by Protocol**", barmode='group')
        st.plotly_chart(fig, use_container_width=True)
    
    # Predict Congestion
    if st.button("📊 **Predict Network Congestion** 📉"):
        congestion_prediction(df_scaled, df)
