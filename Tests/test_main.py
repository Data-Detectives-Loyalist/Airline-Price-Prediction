# test_main.py

import pytest
import pandas as pd
import os  # Import the os module
from src.data.data_loader import DataLoader
from src.data.data_preprocessing import DataPreprocessor
from src.data.data_visualisation import DataVisualizer
from src.models.models_training import ModelTrainer
from src.models.models_evaluation import ModelEvaluator
from src.utils.configs import DATA_PATH, PROCESSED_DATA_PATH, VISUALS

@pytest.fixture(scope='module')
def data_loader():
    return DataLoader(DATA_PATH)

@pytest.fixture(scope='module')
def data_preprocessor():
    loader = DataLoader(DATA_PATH)
    df = loader.load_data()
    return DataPreprocessor(df)

@pytest.fixture(scope='module')
def preprocessed_data(data_preprocessor):
    df1, label_encoders = data_preprocessor.preprocess_data()
    return df1, label_encoders

@pytest.fixture(scope='module')
def data_visualizer():
    loader = DataLoader(DATA_PATH)
    df = loader.load_data()
    return DataVisualizer(df, VISUALS)

@pytest.fixture(scope='module')
def model_trainer(preprocessed_data):
    df1, _ = preprocessed_data
    X = df1.drop('price in CAD', axis=1)
    y = df1['price in CAD']
    return ModelTrainer(X, y)

@pytest.fixture(scope='module')
def models_results(model_trainer):
    selected_models = ['random_forest', 'decision_tree', 'linear_regression', 'xgboost', 'lasso', 'ridge',
                       'elastic_net', 'gradient_boosting']
    return model_trainer.train_selected_models(selected_models)

@pytest.fixture(scope='module')
def model_evaluator(models_results):
    return ModelEvaluator(models_results)

def test_data_loading(data_loader):
    df = data_loader.load_data()
    assert isinstance(df, pd.DataFrame), "Loaded data should be a DataFrame"
    assert not df.empty, "Loaded data should not be empty"

def test_data_preprocessing(data_preprocessor):
    df1, label_encoders = data_preprocessor.preprocess_data()
    assert isinstance(df1, pd.DataFrame), "Preprocessed data should be a DataFrame"
    assert 'price in CAD' in df1.columns, "'price in CAD' should be a column in the preprocessed data"

def test_save_preprocessed_data(data_preprocessor):
    data_preprocessor.save_preprocessed_data(PROCESSED_DATA_PATH)
    assert os.path.exists(PROCESSED_DATA_PATH), "Processed data file should exist"

def test_data_visualization(data_visualizer):
    data_visualizer.plot_number_of_stops()
    data_visualizer.plot_travel_classes_distribution()
    data_visualizer.plot_days_left_distribution()
    data_visualizer.plot_top_10_airlines()
    data_visualizer.plot_average_price_by_airline()
    # Assuming the visualizations are saved as files, you can check for their existence
    for visual in VISUALS:
        assert os.path.exists(visual), f"{visual} should exist"

def test_model_training(model_trainer):
    selected_models = ['random_forest', 'decision_tree', 'linear_regression', 'xgboost', 'lasso', 'ridge',
                       'elastic_net', 'gradient_boosting']
    models_results = model_trainer.train_selected_models(selected_models)
    assert isinstance(models_results, dict), "models_results should be a dictionary"
    for model_type in selected_models:
        assert model_type in models_results, f"{model_type} should be in models_results"

def test_model_evaluation(model_evaluator):
    evaluation_results = model_evaluator.evaluate_models()
    assert isinstance(evaluation_results, dict), "evaluation_results should be a dictionary"
    for model_type, metrics in evaluation_results.items():
        assert 'train_mse' in metrics, f"train_mse should be in metrics for {model_type}"
        assert 'train_r2' in metrics, f"train_r2 should be in metrics for {model_type}"
        assert 'test_mse' in metrics, f"test_mse should be in metrics for {model_type}"
        assert 'test_r2' in metrics, f"test_r2 should be in metrics for {model_type}"

# Run the tests
if __name__ == "__main__":
    pytest.main()