import pandas as pd
from datetime import datetime, timedelta
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException, TimeoutException, ElementClickInterceptedException
from bs4 import BeautifulSoup
import time
import os

# Airport code mapping
AIRPORT_CODES = {
    "Toronto Pearson International (YYZ)": "YYZ",
    "Bengaluru (BLR)": "BLR",
    "Indira Gandhi International (DEL)": "DEL",
    "Mumbai (BOM)": "BOM",
    "Hyderabad (HYD)": "HYD",
    "Chennai (MAA)": "MAA",
    "Ahmedabad (AMD)": "AMD",
    "Kochi (COK)": "COK",
    "Colombo Bandaranayake (CMB)": "CMB",
    "Kathmandu (KTM)": "KTM",
    "Mexico City Juarez International (MEX)": "MEX",
    "Sao Paulo Guarulhos (GRU)": "GRU",
    "Aminu Kano Intl (KAN)": "KAN"
}

def generate_flight_url(source, destination, date, travel_class="economy"):
    """Generates the Kayak flight search URL."""
    base_url = "https://www.ca.kayak.com/flights/"
    class_segment = travel_class if travel_class in ["premium", "business"] else ""
    return f"{base_url}{source}-{destination}/{date}/{class_segment}?sort=bestflight_a"

def click_show_more_button(driver):
    """Clicks the 'Show more results' button on the Kayak page."""
    try:
        show_more_button = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, 'div.ULvh-button.show-more-button'))
        )
        driver.execute_script("arguments[0].click();", show_more_button)
        return True
    except (NoSuchElementException, TimeoutException, ElementClickInterceptedException):
        return False

def scrape_flight_data(driver, url):
    """Scrapes flight data from a given Kayak search results page."""
    driver.get(url)
    time.sleep(5)

    # Click 'Show more results' until no more results are available
    n = 5
    while n != 0:
        time.sleep(3)
        if not click_show_more_button(driver):
            break
        n -= 1
    time.sleep(5)

    page_source = driver.page_source
    soup = BeautifulSoup(page_source, 'html.parser')
    flights = soup.find_all('div', class_='nrc6 nrc6-mod-pres-default')

    data = []
    date_in_url = url.split('/')[5]
    for flight in flights:
        airline = flight.find('div', class_='J0g6-operator-text').text.strip() if flight.find('div', class_='J0g6-operator-text') else ''
        source = flight.find_all('div', class_='c_cgF c_cgF-mod-variant-full-airport-wide')[0]['title'].strip() if len(flight.find_all('div', 'c_cgF c_cgF-mod-variant-full-airport-wide')) > 0 else ''
        destination = flight.find_all('div', class_='c_cgF c_cgF-mod-variant-full-airport-wide')[1]['title'].strip() if len(flight.find_all('div', 'c_cgF c_cgF-mod-variant-full-airport-wide')) > 1 else ''

        departure_arrival_div = flight.find('div', class_='vmXl vmXl-mod-variant-large')
        departure_span, arrival_span = departure_arrival_div.find_all('span')[:3:2] if departure_arrival_div else (None, None)
        departure = departure_span.text.strip() if departure_span else ''
        arrival = arrival_span.text.strip() if arrival_span else ''

        jweo_div = flight.find('div', class_='JWEO')
        num_stops_div = jweo_div.find('div', class_='vmXl vmXl-mod-variant-default') if jweo_div else None
        num_stops_text = num_stops_div.find('span', 'JWEO-stops-text').text.strip() if num_stops_div else ''

        stopover_div = jweo_div.find('div', class_='c_cgF c_cgF-mod-variant-full-airport') if jweo_div else None
        stopover_details = ', '.join([span.get('title', '') for span in stopover_div.find_all('span')]) if stopover_div else ''

        price = flight.find('div', class_='f8F1-price-text').text.strip() if flight.find('div', class_='f8F1-price-text') else ''
        travel_class = flight.find('div', class_='aC3z-name')['title'].strip() if flight.find('div', class_='aC3z-name') else ''

        data.append({
            'Airline': airline,
            'Source': source,
            'Destination': destination,
            'Departure': departure,
            'Arrival': arrival,
            'Number of Stops': num_stops_text,
            'Stopover Details': stopover_details,
            'Price': price,
            'Class': travel_class,
            'Date': date_in_url
        })

    return pd.DataFrame(data)

def scrape_flights_from_excel(input_file, output_file, driver_path, days_to_scrape=50):
    """Reads flight data from an Excel file and scrapes flight details from Kayak."""

    input_data = pd.read_excel(input_file)
    today = datetime.today()
    file_exists = os.path.isfile(output_file)

    service = Service(driver_path)
    driver = webdriver.Chrome(service=service)

    for index, row in input_data.iterrows():
        source = AIRPORT_CODES.get(row['Source'], "")
        destination = AIRPORT_CODES.get(row['Destination'], "")

        if source and destination:
            for day in range(days_to_scrape):
                date = today + timedelta(days=day)
                formatted_date = date.strftime("%Y-%m-%d")

                for travel_class in ["economy", "premium", "business"]:
                    url = generate_flight_url(source, destination, formatted_date, travel_class)
                    df = scrape_flight_data(driver, url)

                    if not file_exists:
                        df.to_csv(output_file, index=False)
                        file_exists = True
                    else:
                        df.to_csv(output_file, mode='a', header=False, index=False)

                    print(f"Data from {url} saved to {output_file}")
                    time.sleep(10)  # Wait between requests

    driver.quit()
    print("Data scraping complete and saved to scraped_flight_data.csv")