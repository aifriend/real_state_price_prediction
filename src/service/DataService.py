import pickle
from pathlib import Path
from typing import Any

import pandas as pd
from attr.validators import optional
from pandas import DataFrame

from src.service.RentalLogger import logger


class DataService:
    """
    DataService is a class that provides methods for loading and processing data.
    """

    @staticmethod
    def load_data(file_path: str, chunk_size: int = 1000) -> (int, optional):
        """
        Load data from a CSV file in chunks.

        Args:
            file_path (str): The path to the CSV file.
            chunk_size (int, optional): The number of rows to read in each chunk. Defaults to 1000.

        Returns:
            (int, optional): The number of chunks processed.
        """
        chunk_iterator = None
        try:
            chunk_iterator = pd.read_csv(
                file_path, chunksize=chunk_size, compression='gzip')

        except FileNotFoundError:
            logger.info(f"Error: File '{file_path}' not found. Please provide the correct file path.")
        except Exception as e:
            logger.info(f"An unexpected error occurred: {e}")

        return chunk_iterator

    @staticmethod
    def merge_dataset(df1: DataFrame, df2: DataFrame, on: str = '') -> DataFrame:
        """
        Merge two dataframes based on a specified column.

        Args:
            df1 (DataFrame): The first dataframe.
            df2 (DataFrame): The second dataframe.
            on (str, optional): The column to merge on. Defaults to ''.

        Returns:
            (DataFrame): The merged dataframe.
        """

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
    def convert_object_to_string(df: DataFrame) -> DataFrame:
        """
        Convert object type columns to string type.

        Args:
            df (DataFrame): The dataframe to convert.

        Returns:
            (DataFrame): The converted dataframe.
        """

        # Iterate over each column in the DataFrame
        for idx, column in enumerate(df.columns):
            # logger.info(f"colum {idx} type {df.iloc[:, idx].dtype}")
            # Check if the column is of object type
            if df[column].dtypes == 'object':
                # Convert the column to string type
                df[column] = df.iloc[:, idx].astype(pd.StringDtype())
                # logger.info(f"colum {idx} changed type to {df.iloc[:, idx].dtype}")

        return df

    @staticmethod
    def save_model(model: object, project_dir: Path, parent: int = 0, apex: str = 'default') -> None:
        """
        Save a model to a pickle file.

        Args:
            model (object): The model to save.
            project_dir (Path): The project directory.
            parent (int, optional): The parent directory index. Defaults to 0.
            apex (str, optional): The apex name. Defaults to 'default'.

        Returns:
            None
        """
        model_name = f"{apex}_{model.__class__.__name__}"
        pickle.dump(model, open(
            Path.cwd().parents[parent].joinpath(
                project_dir, "models", model_name), 'wb'))

    @staticmethod
    def load_model(model: object, project_dir: Path, parent: int = 0, apex: str = 'default') -> Any:
        """
        Load a model from a pickle file.

        Args:
            model (object): The model name.
            project_dir (Path): The project directory.
            parent (int, optional): The parent directory index. Defaults to 0.
            apex (str, optional): The apex name. Defaults to 'default'.

        Returns:
            object: The loaded model.
        """
        model_name = f"{apex}_{model.__class__.__name__}"
        loaded_model = pickle.load(open(Path.cwd().parents[parent].joinpath(
            project_dir, "models", model_name), 'rb'))

        return loaded_model
