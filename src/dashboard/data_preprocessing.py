import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(file_path):
    data = pd.read_csv(file_path)
    return data


def inspect_data(data):
    print("Columns in the dataset:", data.columns)
    print("First few rows of the dataset:\n", data.head())


def preprocess_data(data, target_column):
    # Convert date columns to datetime format with specified format
    for col in data.columns:
        if data[col].dtype == 'object':
            try:
                data[col] = pd.to_datetime(data[col], format='%Y-%m-%d')  # Specify the correct date format
            except ValueError:
                pass  # If conversion fails, keep the column as is

    # Drop non-numeric columns
    data = data.select_dtypes(include=[float, int])

    X = data.drop(target_column, axis=1)
    y = data[target_column]

    # Splitting the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scaling the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test