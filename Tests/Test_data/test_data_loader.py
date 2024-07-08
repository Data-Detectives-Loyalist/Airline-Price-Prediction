# test_data_loader.py

import os
import pytest
import pandas as pd
from src.data.data_loader import DataLoader

# Sample data for testing
TEST_CSV = 'test_data.csv'

# Create a sample CSV file for testing
@pytest.fixture(scope='module')
def create_test_csv():
    data = {
        'col1': [1, 2, 3],
        'col2': ['A', 'B', 'C']
    }
    df = pd.DataFrame(data)
    df.to_csv(TEST_CSV, index=False)
    yield
    os.remove(TEST_CSV)

def test_load_data_success(create_test_csv):
    loader = DataLoader(TEST_CSV)
    data = loader.load_data()
    assert data is not None, "Data should not be None"
    assert not data.empty, "Data should not be empty"
    assert list(data.columns) == ['col1', 'col2'], "Columns do not match"

def test_load_data_file_not_found():
    loader = DataLoader('non_existent_file.csv')
    data = loader.load_data()
    assert data is None, "Data should be None for non-existent file"

# Run the tests
if __name__ == "__main__":
    pytest.main()