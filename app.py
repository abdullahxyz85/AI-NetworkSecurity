import streamlit as st
import pandas as pd
import dask.dataframe as dd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

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
    df['hour'] = df['Time'].dt.hour
    
    # Encode 'Protocol' column
    label_encoder = LabelEncoder()
    df['protocol_encoded'] = label_encoder.fit_transform(df['Protocol'])
    
    # Select features for scaling
    features = ['Length', 'protocol_encoded', 'hour', 'No.']
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df[features])
    
    return df, df_scaled

# Detect anomalies using Isolation Forest
@st.cache
def detect_anomalies(df, df_scaled):
    # Using Isolation Forest to detect anomalies
    model = IsolationForest(contamination=0.1, random_state=42)
    df['is_anomaly'] = model.fit_predict(df_scaled)
    
    # Mapping the anomaly prediction to 'Normal' and 'Anomaly'
    df['Anomaly'] = df['is_anomaly'].map({1: 'Normal', -1: 'Anomaly'})
    
    return df

def display_results(df):
    st.write("### Anomaly Detection Results")
    st.write(df.head())  # Display first few rows of the dataframe
    
    # Count anomalies and normal records
    anomaly_count = df['Anomaly'].value_counts()
    st.write("Total Anomalies:", anomaly_count.get('Anomaly', 0))
    st.write("Total Normal Records:", anomaly_count.get('Normal', 0))
    
    # Display a bar chart of anomaly counts
    st.bar_chart(df['Anomaly'].value_counts())

def main():
    st.title("Network Anomaly Detection and Congestion Prediction")
    
    # File uploader widget
    uploaded_file = st.file_uploader("Upload Network Traffic Data (CSV)", type=["csv"])
    
    if uploaded_file is not None:
        # Load and preprocess the data
        df = load_data(uploaded_file)
        df, df_scaled = preprocess_data(df)
        
        # Apply anomaly detection
        df = detect_anomalies(df, df_scaled)
        
        # Train-test split for model training (if needed)
        X_train, X_test, y_train, y_test = train_test_split(df_scaled, df['Anomaly'], test_size=0.2, random_state=42)
        
        # Display the results of the anomaly detection
        display_results(df)

if __name__ == "__main__":
    main()
