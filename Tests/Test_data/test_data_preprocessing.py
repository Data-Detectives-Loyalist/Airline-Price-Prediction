# test_data_preprocessing.py

import os
import pytest
import pandas as pd
from src.data.data_preprocessing import DataPreprocessor

# Sample data for testing
TEST_CSV = 'test_preprocessed_data.csv'

# Create a sample DataFrame for testing
@pytest.fixture(scope='module')
def sample_dataframe():
    data = {
        'Date': ['2024-07-08', '2024-07-09', '2024-07-10'],
        'Departure_24hr': ['14:30', '16:45', '09:15'],
        'Arrival_24hr': ['16:00', '18:30', '11:00'],
        'Airline': ['AirlineA', 'AirlineB', 'AirlineA'],
        'Source': ['CityA', 'CityB', 'CityA'],
        'Destination': ['CityB', 'CityA', 'CityB'],
        'Class': ['Economy', 'Business', 'Economy']
    }
    df = pd.DataFrame(data)
    return df

def test_convert_date(sample_dataframe):
    preprocessor = DataPreprocessor(sample_dataframe.copy())
    preprocessor.convert_date()
    assert pd.api.types.is_datetime64_any_dtype(preprocessor.df['Date']), "Date column should be datetime"

def test_create_time_features(sample_dataframe):
    preprocessor = DataPreprocessor(sample_dataframe.copy())
    preprocessor.convert_date().create_time_features()
    assert 'Month' in preprocessor.df.columns, "Month column should be created"
    assert 'Day_of_Week' in preprocessor.df.columns, "Day_of_Week column should be created"

def test_convert_time_to_minutes(sample_dataframe):
    preprocessor = DataPreprocessor(sample_dataframe.copy())
    preprocessor.convert_time_to_minutes()
    assert 'Departure_Minutes' in preprocessor.df.columns, "Departure_Minutes column should be created"
    assert 'Arrival_Minutes' in preprocessor.df.columns, "Arrival_Minutes column should be created"
    assert preprocessor.df['Departure_Minutes'].iloc[0] == 870, "Departure_Minutes should be 870 for 14:30"
    assert preprocessor.df['Arrival_Minutes'].iloc[0] == 960, "Arrival_Minutes should be 960 for 16:00"

def test_encode_categorical_variables(sample_dataframe):
    preprocessor = DataPreprocessor(sample_dataframe.copy())
    preprocessor.encode_categorical_variables()
    for col in ['Airline', 'Source', 'Destination', 'Class']:
        assert col + '_Encoded' in preprocessor.df.columns, col + "_Encoded column should be created"

def test_drop_unnecessary_columns(sample_dataframe):
    preprocessor = DataPreprocessor(sample_dataframe.copy())
    preprocessor.drop_unnecessary_columns()
    for col in ['Airline', 'Source', 'Destination', 'Class', 'Date', 'Departure_24hr', 'Arrival_24hr']:
        assert col not in preprocessor.df.columns, col + " column should be dropped"

def test_preprocess_data(sample_dataframe):
    preprocessor = DataPreprocessor(sample_dataframe.copy())
    df, label_encoders = preprocessor.preprocess_data()
    assert df is not None, "Preprocessed DataFrame should not be None"
    assert label_encoders is not None, "Label encoders should not be None"
    for col in ['Airline', 'Source', 'Destination', 'Class']:
        assert col + '_Encoded' in df.columns, col + "_Encoded column should be in preprocessed DataFrame"

def test_save_preprocessed_data(sample_dataframe):
    preprocessor = DataPreprocessor(sample_dataframe.copy())
    df, _ = preprocessor.preprocess_data()
    preprocessor.save_preprocessed_data(TEST_CSV)
    assert os.path.exists(TEST_CSV), "Preprocessed data file should be saved"
    os.remove(TEST_CSV)

# Run the tests
if __name__ == "__main__":
    pytest.main()