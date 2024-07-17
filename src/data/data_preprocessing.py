# src/data_preprocessing.py
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os


class DataPreprocessor:
    """Preprocesses the flight price prediction dataset."""

    def __init__(self, df):
        """Initializes with a DataFrame."""
        self.df = df
        self.label_encoders = {}

    def convert_date(self):
        """Converts 'Date' column to datetime objects."""
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        return self

    def create_time_features(self):
        """Extracts month and day of the week from 'Date'."""
        self.df['Month'] = self.df['Date'].dt.month
        self.df['Day_of_Week'] = self.df['Date'].dt.dayofweek
        return self

    def time_to_minutes(self, time_str):
        """Converts 'HH:MM' time to total minutes."""
        hours, minutes = map(int, time_str.split(':'))
        return hours * 60 + minutes

    def convert_time_to_minutes(self):
        """Converts departure & arrival time to total minutes."""
        self.df['Departure_Minutes'] = self.df['Departure_24hr'].apply(self.time_to_minutes)
        self.df['Arrival_Minutes'] = self.df['Arrival_24hr'].apply(self.time_to_minutes)
        return self

    def encode_categorical_variables(self):
        """Encodes categorical columns using LabelEncoder."""
        categorical_columns = ['Airline', 'Source', 'Destination', 'Class']
        for col in categorical_columns:
            le = LabelEncoder()
            self.df[col + '_Encoded'] = le.fit_transform(self.df[col])
            self.label_encoders[col] = le
        return self

    def drop_unnecessary_columns(self):
        """Drops original unneeded columns after preprocessing."""
        self.df = self.df.drop(
            columns=['Airline', 'Source', 'Destination', 'Class', 'Date', 'Departure_24hr', 'Arrival_24hr'])
        return self

    def preprocess_data(self):
        """Applies all preprocessing steps."""
        self.convert_date()
        self.create_time_features()
        self.convert_time_to_minutes()
        self.encode_categorical_variables()
        self.drop_unnecessary_columns()
        return self.df, self.label_encoders

    def save_preprocessed_data(self, output_path):
        """Saves the preprocessed data to a CSV file."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self.df.to_csv(output_path, index=False)
        print('Preprocessed data saved to {}'.format(output_path))