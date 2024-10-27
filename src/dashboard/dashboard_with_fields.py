import streamlit as st
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics
import numpy as np
from datetime import date
from flight_price_prediction import load_and_preprocess_data, split_data, scale_data

# Path to the predefined dataset
DATA_PATH = "/Users/fatemi/Desktop/Loyalist study data/SEM2/Step Presentation/Airline-Price-Prediction/data/dataset.csv"

def main():
    st.title("Airline Fare Prediction Dashboard")

    # Load and preprocess the data
    df = load_and_preprocess_data(DATA_PATH)
    X_train, X_test, y_train, y_test = split_data(df, 'Fare')
    X_train_scaled, X_test_scaled = scale_data(X_train, X_test)

    # st.write(df.head())

    # Train the model
    model = DecisionTreeRegressor()
    model.fit(X_train_scaled, y_train)

    # Input fields for prediction
    st.header("Flight Fare Prediction")

    current_date = date.today()
    departure_date = st.date_input("Departure Date", min_value=current_date)
    arrival_date = st.date_input("Arrival Date", min_value=current_date)

    source_columns = [col for col in df.columns if col.startswith('Source_')]
    source_options = [col.replace('Source_', '') for col in source_columns]
    source = st.selectbox("Source", options=source_options)

    destination_columns = [col for col in df.columns if col.startswith('Destination_')]
    destination_options = [col.replace('Destination_', '') for col in destination_columns]
    destination = st.selectbox("Destination", options=destination_options)

    airline_columns = [col for col in df.columns if col.startswith('Airline_')]
    airline_options = [col.replace('Airline_', '') for col in airline_columns]
    airline = st.selectbox("Airline", options=airline_options)

    if st.button("Predict Fare"):
        # Example prediction using the model
        # Note: Actual prediction logic needs the input features to match model's training features
        # For example, let's just use the model's score as a placeholder prediction
        y_pred = model.predict(X_test_scaled)
        accuracy = metrics.r2_score(y_test, y_pred)
        st.write(f"Model Accuracy: {accuracy * 100:.2f}%")

        # Evaluate Model
        st.write("Model evaluated successfully!")
        st.write(f"Train Score: {model.score(X_train_scaled, y_train)}")
        st.write(f"Test Score: {model.score(X_test_scaled, y_test)}")
        st.write(f"Mean Absolute Error: {metrics.mean_absolute_error(y_test, y_pred)}")
        st.write(f"Mean Squared Error: {metrics.mean_squared_error(y_test, y_pred)}")
        st.write(f"Root Mean Squared Error: {np.sqrt(metrics.mean_squared_error(y_test, y_pred))}")
        st.write(f"R2 Score: {metrics.r2_score(y_test, y_pred)}")

        # Create scatter plot
        fig, ax = plt.subplots()
        ax.scatter(y_test, y_pred, alpha=0.5)
        ax.set_xlabel("y_test")
        ax.set_ylabel("y_pred")
        st.pyplot(fig)

if __name__ == "__main__":
    main()
