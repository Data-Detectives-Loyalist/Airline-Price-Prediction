from scrape_data import scrape_flights_from_excel

if __name__ == "__main__":
    input_excel_file = "flight_data.xlsx"
    output_csv_file = "scraped_flight_data.csv"
    chromedriver_path = "../scraping/chromedriver-win64/chromedriver.exe"

    scrape_flights_from_excel(input_excel_file, output_csv_file, chromedriver_path)