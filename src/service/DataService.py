import pickle
from pathlib import Path

import pandas as pd


class DataService:

    def __init__(self):
        pass

    def get_listings(self):
        pass

    def get_reviews(self):
        pass

    def get_calendar(self):
        pass

    def get_neighbourhoods(self):
        pass

    @staticmethod
    def load_data(file_path, chunk_size=1000):
        chunk_iterator = None
        try:
            chunk_iterator = pd.read_csv(
                file_path, chunksize=chunk_size, compression='gzip')

        except FileNotFoundError:
            print(f"Error: File '{file_path}' not found. Please provide the correct file path.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

        return chunk_iterator

    @staticmethod
    def merge_dataset(df1, df2, on=''):
        # Get the column names that are present in both dataframes
        common_columns = list(set(df1.columns) & set(df2.columns))

        # Identify the duplicate columns in df2
        duplicate_columns = [col for col in common_columns if col != on]

        # Drop the duplicate columns from df2
        df2 = df2.drop(columns=duplicate_columns)

        # Merge the dataframes based on the specified column
        m_df = pd.merge(df1, df2, on=on)

        return m_df

    @staticmethod
    def convert_object_to_string(df):
        # Iterate over each column in the DataFrame
        for column in df.columns:
            # Check if the column is of object type
            if df[column].dtypes == 'object':
                # Convert the column to string type
                df[column] = df[column].astype('string')

        return df

    @staticmethod
    def save_model(model, parent=0, apex='default'):
        model_name = f"{apex}_{model.__class__.__name__}"
        pickle.dump(model, open(
            Path.cwd().parents[parent].joinpath(
                "models", model_name), 'wb'))

    @staticmethod
    def load_model(model: str, parent=0, apex='default'):
        model_name = f"{apex}_{model.__class__.__name__}"
        loaded_model = pickle.load(open(Path.cwd().parents[parent].joinpath(
            "models", model_name), 'rb'))
        return loaded_model
