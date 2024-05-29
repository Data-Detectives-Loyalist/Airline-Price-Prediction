from linear_regression_model import train_linear_regression, evaluate_linear
from xgboost_model import train_xgboost, evaluate_xgboost


def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    results = {}

    # Train and evaluate Linear Regression
    linear_model = train_linear_regression(X_train, y_train)
    linear_metrics = evaluate_linear(linear_model, X_test, y_test)
    results['Linear Regression'] = linear_metrics

    # Train and evaluate XGBoost
    xgboost_model = train_xgboost(X_train, y_train)
    xgboost_metrics = evaluate_xgboost(xgboost_model, X_test, y_test)
    results['XGBoost'] = xgboost_metrics

    return results