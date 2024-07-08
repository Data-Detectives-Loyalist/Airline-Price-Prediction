# src/utils.py
import os
import pandas as pd

class Utils:
    @staticmethod
    def save_to_csv(df, filepath):
        df.to_csv(filepath, index=False)

    @staticmethod
    def load_from_csv(filepath):
        return pd.read_csv(filepath)