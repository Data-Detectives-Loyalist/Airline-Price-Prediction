import pandas as pd
import numpy as np

def clean_stopover_details(df):
    """
    Cleans and splits the 'Stopover Details' column into separate columns
    for stopover time and airport.
    """
    def clean_split_stopover(details):
        if isinstance(details, str):
            parts = details.split(', , ')
            cleaned_parts = [(part.split(' layover, <b>')[0].strip(),
                              part.split(' layover, <b>')[1].replace('</b>', '').strip())
                             for part in parts]
            return cleaned_parts
        else:
            return []

    df['Cleaned Stopover Details'] = df['Stopover Details'].apply(clean_split_stopover)
    stopover_df = df['Cleaned Stopover Details'].apply(pd.Series)

    flat_df = pd.DataFrame()
    for col in stopover_df.columns:
        temp_list = stopover_df[col].apply(lambda x: x if isinstance(x, tuple) else (np.nan, np.nan))
        temp_df = pd.DataFrame(temp_list.tolist(), columns=[f'Stopover_{col+1}_Time', f'Stopover_{col+1}_Airport'])
        flat_df = pd.concat([flat_df, temp_df], axis=1)

    flat_df['Stopover_1_Time'] = flat_df['Stopover_1_Time'].str.replace('^, ', '', regex=True)
    df = pd.concat([df, flat_df], axis=1)
    df = df.drop(columns=['Cleaned Stopover Details', 'Stopover Details'])
    return df

def calculate_total_stopover_time(df):
    """Calculates the total stopover time in minutes and formats it."""
    def time_to_minutes(time_str):
        if pd.isna(time_str):
            return 0
        hours, minutes = 0, 0
        if 'h' in time_str:
            hours = int(time_str.split('h')[0].strip())
            time_str = time_str.split('h')[1].strip()
        if 'm' in time_str:
            minutes = int(time_str.split('m')[0].strip())
        return hours * 60 + minutes

    # Use map instead of applymap
    for col in df.columns:
        if "Time" in col:
            df[col] = df[col].map(time_to_minutes)
    df['Total_Stopover_Time'] = df[[col for col in df.columns if 'Time' in col]].sum(axis=1)

    def minutes_to_time(minutes):
        hours = minutes // 60
        minutes = minutes % 60
        return f'{hours}h {minutes}m'

    df['Total_Stopover_Time'] = df['Total_Stopover_Time'].apply(minutes_to_time)
    return df

    def minutes_to_time(minutes):
        hours = minutes // 60
        minutes = minutes % 60
        return f'{hours}h {minutes}m'

    df['Total_Stopover_Time'] = df['Total_Stopover_Time'].apply(minutes_to_time)
    return df

def split_airline_column(df):
    """Splits the 'Airline' column and handles missing values."""
    split_df = df['Airline'].str.split('• Operated by', n=1, expand=True)
    df['Airline'] = split_df[0].str.strip()
    df['Operated'] = split_df[1].str.strip().fillna('None')
    return df

def clean_price_column(df):
    """Cleans the 'Price' column and renames it to 'price in CAD'."""
    df['price in CAD'] = df['Price'].str.replace('C$', '').str.replace(',', '').astype(int)
    df = df.drop(columns=['Price'])
    return df

def main():
    # Example usage (you would typically load your data here)
    df = pd.DataFrame({'Stopover Details': ['1h 20m layover, <b>Airport A</b>, , 2h 45m layover, <b>Airport B</b>',
                                         '3h 10m layover, <b>Airport C</b>', None],
                       'Airline': ['Airline A • Operated by Partner Airline',
                                   'Airline B', 'Airline C • Operated by Airline C'],
                       'Price': ['C$1,234.56', 'C$987.65', 'C$500']})

    df = clean_stopover_details(df)
    df = calculate_total_stopover_time(df)
    df = split_airline_column(df)
    df = clean_price_column(df)

    print(df)

if __name__ == "__main__":
    main()