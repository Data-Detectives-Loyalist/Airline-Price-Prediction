import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics
import numpy as np
from flight_price_prediction import load_and_preprocess_data, split_data, scale_data


def main():
    st.title("Airline Fare Prediction Dashboard")

    st.sidebar.title("Options")
    option = st.sidebar.selectbox("Choose a step", ("Load Data", "Train Model", "Evaluate Model"))

    if option == "Load Data":
        st.header("Load Data")
        data_path = st.text_input("Enter the path to the dataset", "data/dataset.csv")
        if st.button("Load and Preprocess Data"):
            df = load_and_preprocess_data(data_path)
            st.write("Data loaded and preprocessed successfully!")
            st.write(df.head())
    
    elif option == "Train Model":
        st.header("Train Model")
        data_path = st.text_input("Enter the path to the dataset", "data/dataset.csv")
        if st.button("Load Data and Train Model"):
            df = load_and_preprocess_data(data_path)
            X_train, X_test, y_train, y_test = split_data(df, 'Fare')
            X_train_scaled, X_test_scaled = scale_data(X_train, X_test)
            model = DecisionTreeRegressor()
            model.fit(X_train_scaled, y_train)
            st.write("Model trained successfully!")
    
    elif option == "Evaluate Model":
        st.header("Evaluate Model")
        data_path = st.text_input("Enter the path to the dataset", "data/dataset.csv")
        if st.button("Load Data, Train and Evaluate Model"):
            df = load_and_preprocess_data(data_path)
            X_train, X_test, y_train, y_test = split_data(df, 'Fare')
            X_train_scaled, X_test_scaled = scale_data(X_train, X_test)
            model = DecisionTreeRegressor()
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
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
