# test_model_training.py

import os
import pytest
import pandas as pd
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import joblib
from src.models.models_training import ModelTrainer
from src.models.models import MODELS, MODEL_PARAMS
from src.utils.configs import MODEL_PATH

# Create a sample dataset for testing
@pytest.fixture(scope='module')
def sample_data():
    X, y = make_regression(n_samples=100, n_features=10, noise=0.1, random_state=42)
    return X, y

@pytest.fixture(scope='module')
def model_trainer(sample_data):
    X, y = sample_data
    return ModelTrainer(X, y)

def test_train_model(model_trainer):
    model_type = 'linear_regression'
    model, X_train, y_train, X_test, y_test = model_trainer.train_model(model_type)
    assert model_type in model_trainer.trained_models, f"{model_type} should be in trained_models"
    assert os.path.exists(f'{MODEL_PATH}/{model_type}.pkl'), f"{model_type}.pkl should be saved in {MODEL_PATH}"
    loaded_model = joblib.load(f'{MODEL_PATH}/{model_type}.pkl')
    assert isinstance(loaded_model, type(model)), f"Loaded model should be an instance of {type(model)}"

def test_train_selected_models(model_trainer):
    selected_models = ['linear_regression', 'decision_tree']
    results = model_trainer.train_selected_models(selected_models)
    for model_type in selected_models:
        assert model_type in results, f"{model_type} should be in results"
        model, X_train, y_train, X_test, y_test = results[model_type]
        assert model_type in model_trainer.trained_models, f"{model_type} should be in trained_models"
        assert os.path.exists(f'{MODEL_PATH}/{model_type}.pkl'), f"{model_type}.pkl should be saved in {MODEL_PATH}"
        loaded_model = joblib.load(f'{MODEL_PATH}/{model_type}.pkl')
        assert isinstance(loaded_model, type(model)), f"Loaded model should be an instance of {type(model)}"

# Clean up the model path directory after tests
@pytest.fixture(scope='module', autouse=True)
def cleanup():
    yield
    if os.path.exists(MODEL_PATH):
        for file in os.listdir(MODEL_PATH):
            os.remove(os.path.join(MODEL_PATH, file))
        os.rmdir(MODEL_PATH)

# Run the tests
if __name__ == "__main__":
    pytest.main()