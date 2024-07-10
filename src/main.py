# main.py
from src.data.data_loader import DataLoader
from src.data.data_preprocessing import DataPreprocessor
from src.data.data_visualisation import DataVisualizer
from src.models.models_training import ModelTrainer
from src.models.models_evaluation import ModelEvaluator
from src.utils.configs import DATA_PATH, PROCESSED_DATA_PATH, VISUALS


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
    selected_models = ['random_forest', 'decision_tree', 'linear_regression', 'xgboost', 'lasso', 'ridge',
                       'elastic_net', 'gradient_boosting']

    # Train models
    trainer = ModelTrainer(X, y)
    models_results = trainer.train_selected_models(selected_models)

    # Evaluate models
    evaluator = ModelEvaluator(models_results)
    evaluation_results = evaluator.evaluate_models()

    for model_type, metrics in evaluation_results.items():
        print('Model: {}'.format(model_type))
        print('  Train MSE: {:.4f}, Train R2: {:.4f}'.format(metrics['train_mse'], metrics['train_r2']))
        print('  Test MSE: {:.4f}, Test R2: {:.4f}'.format(metrics['test_mse'], metrics['test_r2']))


if __name__ == '__main__':
    main()