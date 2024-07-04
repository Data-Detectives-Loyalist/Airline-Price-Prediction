import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load and preprocess data
def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)

    # Convert date columns to datetime
    data['Date'] = pd.to_datetime(data['Date'])
    data['Departure_24hr'] = pd.to_datetime(data['Departure_24hr'], format='%H:%M').dt.time
    data['Arrival_24hr'] = pd.to_datetime(data['Arrival_24hr'], format='%H:%M').dt.time

    # Extract additional features from dates
    data['Day_of_Week'] = data['Date'].dt.dayofweek
    data['Month'] = data['Date'].dt.month

    # Encode categorical variables
    label_encoders = {}
    categorical_features = ['Airline', 'Source', 'Destination', 'Class']
    for feature in categorical_features:
        le = LabelEncoder()
        data[feature] = le.fit_transform(data[feature])
        label_encoders[feature] = le

    return data, label_encoders

# Get features and target
def get_features_and_target(data):
    X = data.drop(columns=['price in CAD', 'Date', 'Departure_24hr', 'Arrival_24hr'])
    y = data['price in CAD']
    return X, y

# Encode input data
def encode_input_data(input_data, label_encoders):
    for feature, le in label_encoders.items():
        try:
            input_data[feature] = le.transform(input_data[feature])
        except ValueError:
            # Handle unseen labels by setting them to a default value or skipping encoding
            input_data[feature] = -1  # This is a placeholder; choose an appropriate value
    return input_data

# Train model and evaluate
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = GradientBoostingRegressor(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return model, mse, mae, r2

# Main execution
if __name__ == "__main__":
    # Load and preprocess data
    data, label_encoders = load_and_preprocess_data('AirlineData.csv')
    X, y = get_features_and_target(data)

    # Train the model
    model, mse, mae, r2 = train_model(X, y)

    # Print evaluation metrics
    print(f"Mean Squared Error: {mse}")
    print(f"Mean Absolute Error: {mae}")
    print(f"R-squared: {r2}")
