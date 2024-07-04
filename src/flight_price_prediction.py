
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.tree import DecisionTreeRegressor


def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath)
    
    df.dropna(inplace=True)
    
    # Convert Date_of_Journey to datetime
    df['Date_of_journey'] = pd.to_datetime(df['Date_of_journey'], format='%Y-%m-%d')
    df['Day'] = df['Date_of_journey'].dt.day
    df['Month'] = df['Date_of_journey'].dt.month
    df.drop(columns=["Date_of_journey"], inplace=True)
    
    # One-hot encoding
    one_hot_encoder = OneHotEncoder(sparse_output=False, drop='first')
    columns_to_encode = ['Class', 'Source', 'Airline', 'Journey_day', 'Destination']
    one_hot_encoded_data = one_hot_encoder.fit_transform(df[columns_to_encode])
    one_hot_encoded_df = pd.DataFrame(one_hot_encoded_data, columns=one_hot_encoder.get_feature_names_out(columns_to_encode))
    df.drop(columns=columns_to_encode, inplace=True)
    df = pd.concat([df, one_hot_encoded_df], axis=1)
    
    # Encode and map Departure and Arrival
    df = encode_and_map(df, 'Departure', ['Before 6 AM', '6 AM - 12 PM', '12 PM - 6 PM', 'After 6 PM'])
    df = encode_and_map(df, 'Arrival', ['Before 6 AM', '6 AM - 12 PM', '12 PM - 6 PM', 'After 6 PM'])
    
    # Label encode Total_stops
    df["Total_stops"] = LabelEncoder().fit_transform(df["Total_stops"])
    
    # Drop unnecessary columns
    df.drop(columns=['Flight_code', 'Departure_time', 'Arrival_time'], inplace=True)
    
    return df

def encode_and_map(df, column_name, order):
    label_encoder = LabelEncoder()
    label_encoder.fit(order)
    encoded_column_name = f"{column_name}_encoded"
    df[encoded_column_name] = label_encoder.transform(df[column_name])
    mapping = dict(zip(label_encoder.transform(order), order))
    decoded_column_name = f"{column_name}_time"
    df[decoded_column_name] = df[encoded_column_name].map(mapping)
    df.drop(columns=[column_name], inplace=True)
    return df

def evaluate_model(X_train, X_test, y_train, y_test):
    model = DecisionTreeRegressor()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    print("Train Score = ", model.score(X_train, y_train))
    print("Test Score = ", model.score(X_test, y_test))
    
    print('MAE:', metrics.mean_absolute_error(y_test, y_pred))
    print('MSE:', metrics.mean_squared_error(y_test, y_pred))
    print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    print('R2 Score:', metrics.r2_score(y_test, y_pred))
    
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.xlabel("y_test")
    plt.ylabel("y_pred")
    plt.show()


def train_decision_tree(df):
    X_train, X_test, y_train, y_test = split_data(df, 'Fare')
    X_train_scaled, X_test_scaled = scale_data(X_train, X_test)
    
    # Train the model
    model = DecisionTreeRegressor()
    model.fit(X_train_scaled, y_train)
    
    return X_train_scaled, X_test_scaled, y_train, y_test

def split_data(df, target_column, test_size=0.25, random_state=42):
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def scale_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

def main():
    # Load and preprocess the data
    df = load_and_preprocess_data('data/dataset.csv')
    
    # Train the model
    X_train, X_test, y_train, y_test = train_decision_tree(df)
    
    # Evaluate the model
    evaluate_model(X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    main()

