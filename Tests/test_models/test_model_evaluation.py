# test_model_evaluate.py

import pytest
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from src.models.models_evaluation import ModelEvaluator


# Create a sample dataset for testing
@pytest.fixture(scope='module')
def sample_data():
    X, y = make_regression(n_samples=100, n_features=10, noise=0.1, random_state=42)
    return X, y


@pytest.fixture(scope='module')
def trained_model(sample_data):
    X, y = sample_data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model, X_train, y_train, X_test, y_test


@pytest.fixture(scope='module')
def models_results(trained_model):
    model, X_train, y_train, X_test, y_test = trained_model
    return {'linear_regression': (model, X_train, y_train, X_test, y_test)}


@pytest.fixture(scope='module')
def model_evaluator(models_results):
    return ModelEvaluator(models_results)


def test_evaluate_model(model_evaluator, trained_model):
    model, X_train, y_train, X_test, y_test = trained_model
    train_mse, train_r2 = model_evaluator.evaluate_model(model, X_train, y_train)
    test_mse, test_r2 = model_evaluator.evaluate_model(model, X_test, y_test)

    assert isinstance(train_mse, float), "train_mse should be a float"
    assert isinstance(train_r2, float), "train_r2 should be a float"
    assert isinstance(test_mse, float), "test_mse should be a float"
    assert isinstance(test_r2, float), "test_r2 should be a float"


def test_evaluate_models(model_evaluator):
    evaluation_results = model_evaluator.evaluate_models()
    assert 'linear_regression' in evaluation_results, "linear_regression should be in evaluation_results"

    results = evaluation_results['linear_regression']
    assert 'train_mse' in results, "train_mse should be in results"
    assert 'train_r2' in results, "train_r2 should be in results"
    assert 'test_mse' in results, "test_mse should be in results"
    assert 'test_r2' in results, "test_r2 should be in results"

    assert isinstance(results['train_mse'], float), "train_mse should be a float"
    assert isinstance(results['train_r2'], float), "train_r2 should be a float"
    assert isinstance(results['test_mse'], float), "test_mse should be a float"
    assert isinstance(results['test_r2'], float), "test_r2 should be a float"


# Run the tests
if __name__ == "__main__":
    pytest.main()