import streamlit as st
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics
import numpy as np
from datetime import date
import requests
import pandas as pd
from io import StringIO
from flight_price_prediction import load_and_preprocess_data, split_data, scale_data

# URL of the dataset on GitHub
GITHUB_URL = "https://github.com/Data-Detectives-Loyalist/fatemi-loyalist/Airline-Price-Prediction/main/data/dataset.csv"

def fetch_data(url):
    response = requests.get(url)
    if response.status_code == 200:
        data = response.content.decode('utf-8')
        return pd.read_csv(StringIO(data))
    else:
        st.error("Failed to fetch data from GitHub")
        return None

def run_app():
    st.title("Airline Fare Prediction Tool")

    # Fetch and prepare the dataset
    dataset = fetch_data(GITHUB_URL)
    if dataset is not None:
        df = load_and_preprocess_data(dataset)
        X_train, X_test, y_train, y_test = split_data(df, 'Fare')
        X_train_scaled, X_test_scaled = scale_data(X_train, X_test)

        # Initialize and train the model
        regressor = DecisionTreeRegressor()
        regressor.fit(X_train_scaled, y_train)

        # User inputs for prediction
        st.header("Predict Your Flight Fare")

        today = date.today()
        dep_date = st.date_input("Departure Date", min_value=today)
        arr_date = st.date_input("Arrival Date", min_value=today)

        source_cols = [col for col in df.columns if col.startswith('Source_')]
        source_list = [col.replace('Source_', '') for col in source_cols]
        source_selection = st.selectbox("Source", options=source_list)

        dest_cols = [col for col in df.columns if col.startswith('Destination_')]
        dest_list = [col.replace('Destination_', '') for col in dest_cols]
        dest_selection = st.selectbox("Destination", options=dest_list)

        airline_cols = [col for col in df.columns if col.startswith('Airline_')]
        airline_list = [col.replace('Airline_', '') for col in airline_cols]
        airline_selection = st.selectbox("Airline", options=airline_list)

        if st.button("Predict Fare"):
            # Make a prediction (placeholder logic)
            predictions = regressor.predict(X_test_scaled)
            r2_accuracy = metrics.r2_score(y_test, predictions)
            st.write(f"Model Accuracy: {r2_accuracy * 100:.2f}%")

            # Display model evaluation metrics
            st.write("Model evaluation results:")
            st.write(f"Training Score: {regressor.score(X_train_scaled, y_train)}")
            st.write(f"Testing Score: {regressor.score(X_test_scaled, y_test)}")
            st.write(f"Mean Absolute Error: {metrics.mean_absolute_error(y_test, predictions)}")
            st.write(f"Mean Squared Error: {metrics.mean_squared_error(y_test, predictions)}")
            st.write(f"Root Mean Squared Error: {np.sqrt(metrics.mean_squared_error(y_test, predictions))}")
            st.write(f"R2 Score: {r2_accuracy}")

            # Generate and display scatter plot
            fig, ax = plt.subplots()
            ax.scatter(y_test, predictions, alpha=0.5)
            ax.set_xlabel("Actual Fare")
            ax.set_ylabel("Predicted Fare")
            st.pyplot(fig)

if __name__ == "__main__":
    run_app()
