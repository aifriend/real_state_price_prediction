"""
Data Loading and Preprocessing:
   - Load the data files into Python using libraries like pandas.
   - Perform data cleaning and preprocessing:
     - Handle missing values
     - Convert data types if necessary
     - Merge relevant datasets based on common columns
   - Explore the data to gain initial insights
"""
from pathlib import Path

import pandas as pd

from src.service.DataService import DataService


def load_calendar_desc():
    df2 = pd.read_csv(
        Path.cwd().parents[0].joinpath("data/raw", "calendar.csv.gz"))
    print(df2.head())

    # Convert object type columns to string
    df2 = DataService.convert_object_to_string(df2)
    print(df2.info())

    return df2


def process_calendar():
    # Load the merged calendar
    calendar_df = load_calendar_desc()

    # for each column in the DataFrame that is null
    for column in calendar_df.columns:
        print(f"\n{column.upper()}\n{'=' * 20}\n{calendar_df[column].describe()}")
        print(f"Number of null values: {calendar_df[column].isna().sum()}\n")

        # if the column has null values
        if (calendar_df[column].isna().any() or
                calendar_df[column].isnull().any() or
                calendar_df[column].eq('<NA>').any()):
            # fill the column with 0.0
            if column in ['minimum_nights', 'maximum_nights']:
                calendar_df[column] = calendar_df[column].fillna(calendar_df[column].mean())
                print("==> Filled with mean value\n")
            # fill the column with $0.0
            elif column in ['price', 'adjusted_price']:
                calendar_df[column] = calendar_df[column].bfill()
                print("==> Filled with previous value\n")

    # Convert objects columns to category if necessary
    for column in ['available']:
        calendar_df[column] = calendar_df[column].astype("category")
        print(f"{column} ==> Converted to category type\n")

    # Convert string to datetime if necessary
    for column in ['date']:
        calendar_df[column] = pd.to_datetime(calendar_df[column], errors='coerce')
        # Fill missing values with desired method (e.g., fill forward)
        calendar_df[column] = calendar_df[column].bfill()
        print(f"{column} ==> Converted to datetime and filling missing values\n")

    # Convert 't' and 'f' to boolean
    for column in ['available']:
        calendar_df[column] = calendar_df[column].map({'t': True, 'f': False})
        print(f"{column} ==> Converted 't' and 'f' to boolean\n")

    for column in calendar_df.columns:
        print(f"\n{column.upper()}\n{'=' * 20}\n{calendar_df[column].describe()}")
        print(f"Number of null values: {calendar_df[column].isna().sum()}\n")


if __name__ == '__main__':
    process_calendar()
