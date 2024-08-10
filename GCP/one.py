import os
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
from google.cloud import storage
from io import StringIO
import functions_framework
import base64

# Airport code mapping and other functions go here...
# Airport code mapping
airport_codes = {
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

# Input data
input_data = [

        {"Source": "Toronto Pearson International (YYZ)", "Destination": "Indira Gandhi International (DEL)"},
        {"Source": "Toronto Pearson International (YYZ)", "Destination": "Bengaluru (BLR)"},
        {"Source": "Toronto Pearson International (YYZ)",
         "Destination": "Chhatrapati Shivaji Maharaj International (BOM)"},
        {"Source": "Toronto Pearson International (YYZ)", "Destination": "Chennai International (MAA)"},
        {"Source": "Toronto Pearson International (YYZ)", "Destination": "Kempegowda International (BLR)"},
        {"Source": "Toronto Pearson International (YYZ)", "Destination": "Tribhuvan International (KTM)"},
        {"Source": "Toronto Pearson International (YYZ)", "Destination": "Murtala Muhammed International (LOS)"},
        {"Source": "Toronto Pearson International (YYZ)", "Destination": "Nnamdi Azikiwe International (ABV)"},
        {"Source": "Toronto Pearson International (YYZ)", "Destination": "Mexico City International (MEX)"},
        {"Source": "Toronto Pearson International (YYZ)", "Destination": "Cancún International (CUN)"},
        {"Source": "Toronto Pearson International (YYZ)", "Destination": "Guadalajara International (GDL)"},
        {"Source": "Toronto Pearson International (YYZ)", "Destination": "Monterrey International (MTY)"},
        {"Source": "Toronto Pearson International (YYZ)", "Destination": "São Paulo/Guarulhos International (GRU)"},
        {"Source": "Toronto Pearson International (YYZ)", "Destination": "Rio de Janeiro/Galeão International (GIG)"},
        {"Source": "Toronto Pearson International (YYZ)", "Destination": "Bandaranaike International (CMB)"}

    # Add more source-destination pairs as needed
]
@functions_framework.cloud_event
def scrape_flights(event):
    all_urls = generate_urls()
    bucket_name = '1flight_data_analysis'
    for url in all_urls[:10]:  # Process a subset of URLs to fit within time limits
        df = scrape(url)
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        destination_blob_name = f'scraped_flight_data_{timestamp}.csv'
        upload_to_gcs(bucket_name, df, destination_blob_name)
        time.sleep(10)  # Adjust or remove as needed
    return 'Scraping job executed successfully.'

def generate_flight_url(source, destination, date, travel_class="economy"):
    base_url = "https://www.ca.kayak.com/flights/"
    if travel_class in ["premium", "business"]:
        class_segment = travel_class
    else:
        class_segment = ""
    return f"{base_url}{source}-{destination}/{date}/{class_segment}?sort=bestflight_a"

def generate_urls():
    today = datetime.today()
    all_urls = []
    for row in input_data:
        source = airport_codes.get(row['Source'], "")
        destination = airport_codes.get(row['Destination'], "")

        if source and destination:
            for day in range(50):
                date = today + timedelta(days=day)
                formatted_date = date.strftime("%Y-%m-%d")
                flight_url_economy = generate_flight_url(source, destination, formatted_date, travel_class="economy")
                all_urls.append(flight_url_economy)
                flight_url_premium_economy = generate_flight_url(source, destination, formatted_date, travel_class="premium")
                all_urls.append(flight_url_premium_economy)
                flight_url_business = generate_flight_url(source, destination, formatted_date, travel_class="business")
                all_urls.append(flight_url_business)
    return all_urls

def click_show_more_button(driver):
    try:
        show_more_button = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, 'div.ULvh-button.show-more-button'))
        )
        driver.execute_script("arguments[0].click();", show_more_button)
        return True
    except (NoSuchElementException, TimeoutException, ElementClickInterceptedException):
        return False

def scrape(url):
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    driver_path = 'C:\webdriver\chromedriver.exe'  # Ensure this path is correct
    service = Service(driver_path)
    driver = webdriver.Chrome(service=service, options=options)
    driver.get(url)
    time.sleep(5)
    n = 5
    while n != 0:
        time.sleep(3)
        if not click_show_more_button(driver):
            break
        n -= 1
    time.sleep(5)
    page_source = driver.page_source
    driver.quit()
    soup = BeautifulSoup(page_source, 'html.parser')
    flights = soup.find_all('div', class_='nrc6 nrc6-mod-pres-default')

    airlines = []
    sources = []
    destinations = []
    departures = []
    arrivals = []
    num_stops = []
    stopover_details_list = []
    prices = []
    classes = []
    dates = []

    date_in_url = url.split('/')[5]

    for flight in flights:
        airline = flight.find('div', class_='J0g6-operator-text').text.strip() if flight.find('div', 'J0g6-operator-text') else ''
        source = flight.find_all('div', class_='c_cgF c_cgF-mod-variant-full-airport-wide')[0]['title'].strip() if len(flight.find_all('div', 'c_cgF c_cgF-mod-variant-full-airport-wide')) > 0 else ''
        destination = flight.find_all('div', 'c_cgF c_cgF-mod-variant-full-airport-wide')[1]['title'].strip() if len(flight.find_all('div', 'c_cgF c_cgF-mod-variant-full-airport-wide')) > 1 else ''
        departure_arrival_div = flight.find('div', class_='vmXl vmXl-mod-variant-large')
        departure_span, arrival_span = departure_arrival_div.find_all('span')[:3:2] if departure_arrival_div else (None, None)
        departure = departure_span.text.strip() if departure_span else ''
        arrival = arrival_span.text.strip() if arrival_span else ''

        jweo_div = flight.find('div', 'JWEO')
        num_stops_div = jweo_div.find('div', 'vmXl vmXl-mod-variant-default') if jweo_div else None
        num_stops_text = num_stops_div.find('span', 'JWEO-stops-text').text.strip() if num_stops_div else ''
        num_stops.append(num_stops_text)

        stopover_div = jweo_div.find('div', 'c_cgF c_cgF-mod-variant-full-airport') if jweo_div else None
        stopover_details = ', '.join([span.get('title', '') for span in stopover_div.find_all('span')]) if stopover_div else ''
        stopover_details_list.append(stopover_details)

        price = flight.find('div', class_='f8F1-price-text').text.strip() if flight.find('div', 'f8F1-price-text') else ''
        travel_class = flight.find('div', 'aC3z-name')['title'].strip() if flight.find('div', 'aC3z-name') else ''

        airlines.append(airline)
        sources.append(source)
        destinations.append(destination)
        departures.append(departure)
        arrivals.append(arrival)
        prices.append(price)
        classes.append(travel_class)
        dates.append(date_in_url)

    df = pd.DataFrame({
        'Airline': airlines,
        'Source': sources,
        'Destination': destinations,
        'Departure': departures,
        'Arrival': arrivals,
        'Number of Stops': num_stops,
        'Stopover Details': stopover_details_list,
        'Price': prices,
        'Class': classes,
        'Date': dates
    })

    return df

def upload_to_gcs(bucket_name, dataframe, destination_blob_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    csv_buffer = StringIO()
    dataframe.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)
    blob.upload_from_string(csv_buffer.getvalue(), content_type='text/csv')
    print(f"Data uploaded to {destination_blob_name}.")
