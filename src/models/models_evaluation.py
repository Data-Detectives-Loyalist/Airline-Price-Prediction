# src/model_evaluate.py
from sklearn.metrics import mean_squared_error, r2_score

class ModelEvaluator:
    def __init__(self, models_results):
        self.models_results = models_results

    def evaluate_model(self, model, X, y):
        predictions = model.predict(X)
        mse = mean_squared_error(y, predictions)
        r2 = r2_score(y, predictions)
        return mse, r2

    def evaluate_models(self):
        evaluation_results = {}
        for model_type, (model, X_train, y_train, X_test, y_test) in self.models_results.items():
            train_mse, train_r2 = self.evaluate_model(model, X_train, y_train)
            test_mse, test_r2 = self.evaluate_model(model, X_test, y_test)
            evaluation_results[model_type] = {
                'train_mse': train_mse,
                'train_r2': train_r2,
                'test_mse': test_mse,
                'test_r2': test_r2
            }
        return evaluation_results