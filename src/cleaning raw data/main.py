import pandas as pd
from data_cleaning import clean_stopover_details, calculate_total_stopover_time, \
    split_airline_column, clean_price_column
from feature_engineering import calculate_days_left, categorize_flight_features, \
    convert_time_features

def load_data(filename):
    """Loads the flight data from a CSV file."""
    df = pd.read_csv(filename)
    return df

def preprocess_data(df):
    """Applies all data preprocessing steps."""
    df = clean_stopover_details(df)
    df = calculate_total_stopover_time(df)
    df = split_airline_column(df)
    df = clean_price_column(df)
    df = calculate_days_left(df)
    df = categorize_flight_features(df)
    df = convert_time_features(df)
    return df

def save_data(df, filename):
    """Saves the processed data to a CSV file."""
    df = df.drop(columns=['Departure', 'Arrival', 'Stopover_1_Time', 'Stopover_1_Airport',
                           'Stopover_2_Time', 'Stopover_2_Airport', 'Stopover_3_Time',
                           'Stopover_3_Airport', 'Operated'], axis=1)
    df['Date'] = pd.to_datetime(df['Date'])
    df['price in CAD'] = df['price in CAD'].astype(float)
    df['days_left'] = df['days_left'].astype(int)
    df['Number of Stops'] = df['Number of Stops'].astype(int)
    df['Arrival_Day_Offset'] = df['Arrival_Day_Offset'].astype(int)
    df.to_csv(filename, index=False)

if __name__ == "__main__":
    data = load_data('../../data/raw/scraped_data.csv')
    processed_data = preprocess_data(data)
    save_data(processed_data, '../../data/cleaned/AirlineData1.csv')