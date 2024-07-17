# Import the necessary function for scraping flight data from an Excel file
from scrape_data import scrape_flights_from_excel

# Main execution block to ensure the script runs only when executed directly
if __name__ == "__main__":
    # Define the path to the input Excel file containing flight data
    input_excel_file = "flight_data.xlsx"

    # Define the path to the output CSV file where the scraped data will be saved
    output_csv_file = "scraped_flight_data.csv"

    # Define the path to the ChromeDriver executable required for web scraping
    chromedriver_path = "../scraping/chromedriver-win64/chromedriver.exe"

    # Call the function to scrape flight data from the Excel file and save it to a CSV file
    scrape_flights_from_excel(input_excel_file, output_csv_file, chromedriver_path)