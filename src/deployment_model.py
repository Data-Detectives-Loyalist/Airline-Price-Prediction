import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score

class FlightPricePredictor:
    def __init__(self, filepath):
        self.filepath = filepath
        self.df = pd.read_csv(filepath)
        self.model = None
        self.scaler = StandardScaler()
        self.one_hot_encoder = OneHotEncoder(sparse=False, drop='first')
        self.columns_to_encode = ['Class', 'Source', 'Airline', 'Journey_day', 'Destination']

    def preprocess(self):
        self.df['Date_of_journey'] = pd.to_datetime(self.df['Date_of_journey'])
        self.df["Day"] = self.df["Date_of_journey"].dt.day
        self.df["Month"] = self.df["Date_of_journey"].dt.month
        self.df.drop(columns=["Date_of_journey"], inplace=True)

        one_hot_encoded_data = self.one_hot_encoder.fit_transform(self.df[self.columns_to_encode])
        one_hot_encoded_df = pd.DataFrame(one_hot_encoded_data, columns=self.one_hot_encoder.get_feature_names_out())
        self.df.drop(columns=self.columns_to_encode, inplace=True)
        self.df = pd.concat([self.df, one_hot_encoded_df], axis=1)

        self.encode_and_map('Departure', ['Before 6 AM', '6 AM - 12 PM', '12 PM - 6 PM', 'After 6 PM'])
        self.encode_and_map('Arrival', ['Before 6 AM', '6 AM - 12 PM', '12 PM - 6 PM', 'After 6 PM'])
        self.df["Total_stops_encoded"] = LabelEncoder().fit_transform(self.df["Total_stops"])
        self.df.drop(columns=['Flight_code', 'Departure_time', 'Arrival_time', 'Total_stops'], inplace=True)

    def encode_and_map(self, column_name, order):
        label_encoder = LabelEncoder()
        label_encoder.fit(order)
        encoded_column_name = f"\{column_name\}_encoded"
        self.df[encoded_column_name] = label_encoder.transform(self.df[column_name])
        mapping = dict(zip(label_encoder.transform(order), order))
        decoded_column_name = f"\{column_name\}_time"
        self.df[decoded_column_name] = self.df[encoded_column_name].map(mapping)
        self.df.drop(columns=[column_name], inplace=True)

    def split_data(self, test_size=0.25, random_state=42):
        X = self.df.drop(columns=['Fare'])
        y = self.df['Fare']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        return X_train, X_test, y_train, y_test

    def train_model(self, X_train, y_train):
        X_train_scaled = self.scaler.fit_transform(X_train)
        self.model = DecisionTreeRegressor()
        self.model.fit(X_train_scaled, y_train)

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def model_performance(self, X_train, y_train, X_test, y_test):
        y_pred_train = self.predict(X_train)
        y_pred_test = self.predict(X_test)
        train_score = r2_score(y_train, y_pred_train)
        test_score = r2_score(y_test, y_pred_test)
        print("Train Score = ", train_score)
        print("Test Score = ", test_score)

if __name__ == "__main__":
    predictor = FlightPricePredictor('flightPrice.csv')
    predictor.preprocess()
    X_train, X_test, y_train, y_test = predictor.split_data()
    predictor.train_model(X_train, y_train)
    predictor.model_performance(X_train, y_train, X_test, y_test)
