# src/model.py
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
import xgboost as xgb

# Dictionary to store model names and their corresponding classes
MODELS = {
    'random_forest': RandomForestRegressor,
    'decision_tree': DecisionTreeRegressor,
    'linear_regression': LinearRegression,
    'xgboost': xgb.XGBRegressor,
    'lasso': Lasso,
    'ridge': Ridge,
    'elastic_net': ElasticNet,
    'gradient_boosting': GradientBoostingRegressor
}

# Dictionary to store default parameters for each model
MODEL_PARAMS = {
    'random_forest': {'n_estimators': 100},
    'decision_tree': {},
    'linear_regression': {},
    'xgboost': {'n_estimators': 100, 'learning_rate': 0.1},
    'lasso': {'alpha': 0.1},
    'ridge': {'alpha': 1.0},
    'elastic_net': {'alpha': 0.1, 'l1_ratio': 0.5},
    'gradient_boosting': {'n_estimators': 100, 'learning_rate': 0.1}
}