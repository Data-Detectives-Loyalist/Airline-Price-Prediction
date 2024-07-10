# test_config.py

import pytest

# Import the configuration file
import src.utils.configs as config

def test_data_path():
    assert config.DATA_PATH == '../data/cleaned/AirlineData.csv', "DATA_PATH is incorrect"

def test_processed_data_path():
    assert config.PROCESSED_DATA_PATH == '../data/Processed/AirlineData_preprocessed.csv', "PROCESSED_DATA_PATH is incorrect"

def test_model_path():
    assert config.MODEL_PATH == '../src/models/saved_models', "MODEL_PATH is incorrect"

def test_visuals_path():
    assert config.VISUALS == '../images-charts/EDA', "VISUALS is incorrect"

# Run the tests
if __name__ == "__main__":
    pytest.main()