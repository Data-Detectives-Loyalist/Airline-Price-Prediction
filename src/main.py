# main.py
from src.data.data_loader import DataLoader
from src.data.data_preprocessing import DataPreprocessor
from src.data.data_visualisation import DataVisualizer
from src.models.models_training import ModelTrainer
from src.models.models_evaluation import ModelEvaluator
from src.utils.configs import DATA_PATH, PROCESSED_DATA_PATH, VISUALS
import time


def main():
    # Load data
    loader = DataLoader(DATA_PATH)
    df = loader.load_data()

    # Preprocess data
    preprocessor = DataPreprocessor(df)
    df1, label_encoders = preprocessor.preprocess_data()

    # Save preprocessed data
    preprocessor.save_preprocessed_data(PROCESSED_DATA_PATH)

    # Visualize data
    visualizer = DataVisualizer(df, VISUALS)
    visualizer.plot_number_of_stops()
    visualizer.plot_travel_classes_distribution()
    visualizer.plot_days_left_distribution()
    visualizer.plot_top_10_airlines()
    visualizer.plot_average_price_by_airline()

    # Split features and target
    X = df1.drop('price in CAD', axis=1)
    y = df1['price in CAD']

    # Define selected models
    selected_models = ['random_forest', 'decision_tree', 'gradient_boosting', 'xgboost', 'linear_regression', 'lasso', 'ridge',
                       'elastic_net']

    # Train models
    trainer = ModelTrainer(X, y)
    models_results = trainer.train_selected_models(selected_models)

    # Evaluate models
    evaluator = ModelEvaluator(models_results)
    evaluation_results = evaluator.evaluate_models()

    for model_type, metrics in evaluation_results.items():
        print(f'Model: {model_type}')
        print(f'  Train MSE: {metrics["train_mse"]:.4f}, Train RMSE: {metrics["train_rmse"]:.4f}, Train MAE: {metrics["train_mae"]:.4f}, Train R2: {metrics["train_r2"]:.4f}')
        print(f'  Test MSE: {metrics["test_mse"]:.4f}, Test RMSE: {metrics["test_rmse"]:.4f}, Test MAE: {metrics["test_mae"]:.4f}, Test R2: {metrics["test_r2"]:.4f}')
        print('-' * 50)


if __name__ == '__main__':
    main()