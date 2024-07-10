# test_model.py

import pytest
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
import xgboost as xgb
from src.models.models import MODELS, MODEL_PARAMS

def test_models_dict():
    assert 'random_forest' in MODELS, "random_forest should be in MODELS"
    assert 'decision_tree' in MODELS, "decision_tree should be in MODELS"
    assert 'linear_regression' in MODELS, "linear_regression should be in MODELS"
    assert 'xgboost' in MODELS, "xgboost should be in MODELS"
    assert 'lasso' in MODELS, "lasso should be in MODELS"
    assert 'ridge' in MODELS, "ridge should be in MODELS"
    assert 'elastic_net' in MODELS, "elastic_net should be in MODELS"
    assert 'gradient_boosting' in MODELS, "gradient_boosting should be in MODELS"

def test_model_params_dict():
    assert 'random_forest' in MODEL_PARAMS, "random_forest should be in MODEL_PARAMS"
    assert 'decision_tree' in MODEL_PARAMS, "decision_tree should be in MODEL_PARAMS"
    assert 'linear_regression' in MODEL_PARAMS, "linear_regression should be in MODEL_PARAMS"
    assert 'xgboost' in MODEL_PARAMS, "xgboost should be in MODEL_PARAMS"
    assert 'lasso' in MODEL_PARAMS, "lasso should be in MODEL_PARAMS"
    assert 'ridge' in MODEL_PARAMS, "ridge should be in MODEL_PARAMS"
    assert 'elastic_net' in MODEL_PARAMS, "elastic_net should be in MODEL_PARAMS"
    assert 'gradient_boosting' in MODEL_PARAMS, "gradient_boosting should be in MODEL_PARAMS"

def test_model_instantiation():
    for model_name, model_class in MODELS.items():
        model_params = MODEL_PARAMS.get(model_name, {})
        model_instance = model_class(**model_params)
        assert isinstance(model_instance, model_class), f"{model_name} should be an instance of {model_class}"

# Run the tests
if __name__ == "__main__":
    pytest.main()