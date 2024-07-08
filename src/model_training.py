# model_training.py

import numpy as np
from sklearn.model_selection import train_test_split
from linear_regression_model import train_linear_regression, evaluate_linear
from xgboost_model import train_xgboost, evaluate_xgboost
from regularized_regression import (
    train_lasso_regression, train_ridge_regression, train_elastic_net, evaluate_model
)

def train_and_evaluate_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    results = {}

    # Train and evaluate Linear Regression
    linear_model = train_linear_regression(X_train, y_train)
    linear_metrics = evaluate_linear(linear_model, X_test, y_test)
    results['Linear Regression'] = linear_metrics

    # Train and evaluate XGBoost Regression
    xgboost_model = train_xgboost(X_train, y_train)
    xgboost_metrics = evaluate_xgboost(xgboost_model, X_test, y_test)
    results['XGBoost'] = xgboost_metrics

    # Train and evaluate Lasso Regression
    lasso_model = train_lasso_regression(X_train, y_train)
    lasso_metrics = evaluate_model(lasso_model, X_test, y_test)
    results['Lasso Regression'] = lasso_metrics

    # Train and evaluate Ridge Regression
    ridge_model = train_ridge_regression(X_train, y_train)
    ridge_metrics = evaluate_model(ridge_model, X_test, y_test)
    results['Ridge Regression'] = ridge_metrics

    # Train and evaluate Elastic Net Regression
    elastic_net_model = train_elastic_net(X_train, y_train)
    elastic_net_metrics = evaluate_model(elastic_net_model, X_test, y_test)
    results['Elastic Net'] = elastic_net_metrics

    return results