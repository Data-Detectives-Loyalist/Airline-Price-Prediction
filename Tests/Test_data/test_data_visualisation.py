# test_data_visualization.py

import os
import pytest
import pandas as pd
from src.data.data_visualisation import DataVisualizer

# Sample data for testing
TEST_OUTPUT_PATH = 'test_visuals'

# Create a sample DataFrame for testing
@pytest.fixture(scope='module')
def sample_dataframe():
    data = {
        'Number of Stops': [1, 2, 1, 0, 1],
        'Class': ['Economy', 'Business', 'Economy', 'First', 'Economy'],
        'days_left': [10, 20, 30, 40, 50],
        'Airline': ['AirlineA', 'AirlineB', 'AirlineA', 'AirlineC', 'AirlineA'],
        'price in CAD': [200, 300, 250, 400, 220]
    }
    df = pd.DataFrame(data)
    return df

@pytest.fixture(scope='module')
def visualizer(sample_dataframe):
    return DataVisualizer(sample_dataframe, TEST_OUTPUT_PATH)

def test_plot_number_of_stops(visualizer):
    visualizer.plot_number_of_stops()
    assert os.path.exists(os.path.join(TEST_OUTPUT_PATH, 'number_of_stops.png')), "number_of_stops.png should be created"

def test_plot_travel_classes_distribution(visualizer):
    visualizer.plot_travel_classes_distribution()
    assert os.path.exists(os.path.join(TEST_OUTPUT_PATH, 'travel_classes_distribution.png')), "travel_classes_distribution.png should be created"

def test_plot_days_left_distribution(visualizer):
    visualizer.plot_days_left_distribution()
    assert os.path.exists(os.path.join(TEST_OUTPUT_PATH, 'days_left_distribution.png')), "days_left_distribution.png should be created"

def test_plot_top_10_airlines(visualizer):
    visualizer.plot_top_10_airlines()
    assert os.path.exists(os.path.join(TEST_OUTPUT_PATH, 'top_10_airlines.png')), "top_10_airlines.png should be created"

def test_plot_average_price_by_airline(visualizer):
    visualizer.plot_average_price_by_airline()
    assert os.path.exists(os.path.join(TEST_OUTPUT_PATH, 'average_price_by_airline.png')), "average_price_by_airline.png should be created"

# Clean up the test output directory after tests
@pytest.fixture(scope='module', autouse=True)
def cleanup():
    yield
    if os.path.exists(TEST_OUTPUT_PATH):
        for file in os.listdir(TEST_OUTPUT_PATH):
            os.remove(os.path.join(TEST_OUTPUT_PATH, file))
        os.rmdir(TEST_OUTPUT_PATH)

# Run the tests
if __name__ == "__main__":
    pytest.main()