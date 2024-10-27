# src/data_visualization.py
import matplotlib.pyplot as plt
import seaborn as sns
import os

class DataVisualizer:
    def __init__(self, df, output_path):
        self.df = df
        self.output_path = output_path
        os.makedirs(self.output_path, exist_ok=True)

    def plot_number_of_stops(self):
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 3, 1)
        sns.countplot(x='Number of Stops', data=self.df)
        plt.title('Number of Stops')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_path, 'number_of_stops.png'))
        plt.close()

    def plot_travel_classes_distribution(self):
        plt.figure(figsize=(10, 6))
        sns.countplot(x='Class', data=self.df, order=self.df['Class'].value_counts().index)
        plt.title('Distribution of Travel Classes')
        plt.xlabel('Travel Class')
        plt.ylabel('Count')
        plt.savefig(os.path.join(self.output_path, 'travel_classes_distribution.png'))
        plt.close()

    def plot_days_left_distribution(self):
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 3, 4)
        sns.histplot(self.df['days_left'], kde=True, bins=30)
        plt.title('Days Left Distribution')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_path, 'days_left_distribution.png'))
        plt.close()

    def plot_top_10_airlines(self):
        top_10_airlines = self.df['Airline'].value_counts().head(10)
        plt.figure(figsize=(12, 10))
        plt.subplot(3, 2, 1)
        sns.countplot(y=self.df[self.df['Airline'].isin(top_10_airlines.index)]['Airline'], order=top_10_airlines.index, palette='viridis')
        plt.title('Top 10 Airlines by Count')
        plt.xlabel('Count')
        plt.ylabel('Airline')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_path, 'top_10_airlines.png'))
        plt.close()

    def plot_average_price_by_airline(self):
        average_price_per_airline = self.df.groupby('Airline')['price in CAD'].mean().sort_values()
        plt.figure(figsize=(12, 10))
        sns.barplot(x=average_price_per_airline.values, y=average_price_per_airline.index, palette='viridis')
        plt.title('Average Price by Airline')
        plt.xlabel('Average Price in CAD')
        plt.ylabel('Airline')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_path, 'average_price_by_airline.png'))
        plt.close()