# src/model_training.py
from sklearn.model_selection import train_test_split
import joblib
from src.models.models import MODELS, MODEL_PARAMS
from src.utils.configs import MODEL_PATH

class ModelTrainer:
    def __init__(self, X, y, model_params=None):
        self.X = X
        self.y = y
        self.model_params = model_params if model_params else MODEL_PARAMS
        self.models = {name: cls(**self.model_params.get(name, {})) for name, cls in MODELS.items()}
        self.trained_models = {}

    def train_model(self, model_type):
        if model_type not in self.models:
            raise ValueError('Model type {} is not supported'.format(model_type))

        model = self.models[model_type]
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

        model.fit(X_train, y_train)
        #joblib.dump(model, f'{MODEL_PATH}/{model_type}.pkl')
        self.trained_models[model_type] = model
        return model, X_train, y_train, X_test, y_test

    def train_selected_models(self, selected_models):
        results = {}
        for model_type in selected_models:
            model, X_train, y_train, X_test, y_test = self.train_model(model_type)
            results[model_type] = (model, X_train, y_train, X_test, y_test)
        return results