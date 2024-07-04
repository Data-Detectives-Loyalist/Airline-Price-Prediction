from data_preprocessing import load_data, inspect_data, preprocess_data
from model_training import train_and_evaluate_models


def main():
    data = load_data('../notebooks/flightPrice.csv')

    # Inspect the data to understand its structure
    inspect_data(data)

    # Update the target column name as per your dataset
    target_column = 'Fare'

    X_train, X_test, y_train, y_test = preprocess_data(data, target_column)

    # Train and evaluate models
    results = train_and_evaluate_models(X_train, y_train)

    # Print or save results
    print(results)

    # Print results
    for model_name, metrics in results.items():
        mae, mse, rmse, r2 = metrics
        print(f'{model_name} - MAE: {mae}, MSE: {mse}, RMSE: {rmse}, R2: {r2}')


if __name__ == "__main__":
    main()