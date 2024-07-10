import pandas as pd
from datetime import datetime, timedelta

def calculate_days_left(df):
    """Calculates the number of days left until the flight date."""
    def calculate_days_left(flight_date):
        start_date = datetime(2024, 6, 22)
        current_date = datetime.strptime(flight_date, '%Y-%m-%d')
        return (current_date - start_date).days + 1
    df['days_left'] = df['Date'].apply(calculate_days_left)
    return df

def categorize_flight_features(df):
    """Categorizes and cleans up flight-related features."""
    df['Airline'] = df['Airline'].apply(lambda x: 'Multiple Airlines' if ',' in x else x)
    df['Number of Stops'] = df['Number of Stops'].str.extract('(\d+)').fillna(0).astype(int)

    def categorize_class(class_name):
        class_name = class_name.lower()
        if 'business' in class_name or any(keyword in class_name for keyword in ['executive','upper class']):
            return 'Business Class'
        elif 'economy' in class_name or any(keyword in class_name for keyword in ['classic','flex','comfort','latitude','light','basic', 'best', 'eco', 'discount','promotion','best buy', 'plus','saver','best offer', 'eco saver', 'ultrasaver', 'standard']):
            return 'Economy Class'
        elif 'first' in class_name:
            return 'First Class'
        elif 'premium' in class_name:
            return 'Premium Economy'
        else:
            return 'Other'

    df['Class'] = df['Class'].apply(categorize_class)
    return df

def convert_time_features(df):
    """Converts and extracts time-related features."""
    def convert_to_24hr(time_str):
        return datetime.strptime(time_str, '%I:%M %p').strftime('%H:%M')

    def extract_arrival_info(arrival_str):
        if '+' in arrival_str:
            time_part, day_increment = arrival_str.split('+')
            day_increment = int(day_increment)
        else:
            time_part = arrival_str
            day_increment = 0
        arrival_24hr = datetime.strptime(time_part.strip(), '%I:%M %p').strftime('%H:%M')
        return arrival_24hr, day_increment

    df['Departure_24hr'] = df['Departure'].apply(convert_to_24hr)
    df['Arrival_24hr'], df['Arrival_Day_Offset'] = zip(*df['Arrival'].apply(extract_arrival_info))

    def convert_to_minutes(time_str):
        if pd.isna(time_str):
            return 0
        hours, minutes = 0, 0
        if 'h' in time_str:
            hours = int(time_str.split('h')[0].strip())
            time_str = time_str.split('h')[1].strip()
        if 'm' in time_str:
            minutes = int(time_str.split('m')[0].strip())
        return hours * 60 + minutes

    df['Total_Stopover_Time'] = df['Total_Stopover_Time'].apply(convert_to_minutes)
    return df

# Example usage (you would typically load your data here)
def main():
    df = pd.DataFrame({'Date': ['2024-07-01', '2024-07-15', '2024-08-01'],
                       'Departure': ['08:00 AM', '06:30 PM', '11:45 PM'],
                       'Arrival': ['09:30 PM', '07:15 AM +1', '02:00 PM +2'],
                       'Total_Stopover_Time': ['1h 30m', '2h 45m', '5h 15m'],
                       'Number of Stops': ['1 Stop', '2 Stops', 'No Stops'],
                       'Class': ['Economy', 'Business Class', 'Premium Economy'],
                       'Airline': ['Airline A, Airline B', 'Airline C', 'Airline D']})

    df = calculate_days_left(df)
    df = categorize_flight_features(df)
    df = convert_time_features(df)

    print(df)

if __name__ == "__main__":
    main()