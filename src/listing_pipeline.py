"""
Data Loading and Preprocessing:
   - Load the data files into Python using libraries like pandas.
   - Perform data cleaning and preprocessing:
     - Handle missing values
     - Convert data types if necessary
     - Merge relevant datasets based on common columns
   - Explore the data to gain initial insights

   ['id', 'name', 'host_id', 'host_name', 'neighbourhood_group', 'neighbourhood', 'latitude', 'longitude',
   'room_type', 'price', 'minimum_nights', 'number_of_reviews', 'last_review', 'reviews_per_month',
   'calculated_host_listings_count', 'availability_365', 'listing_url', 'scrape_id', 'last_scraped', 'summary',
   'space', 'description', 'neighborhood_overview', 'notes', 'transit', 'access', 'interaction', 'house_rules',
   'picture_url', 'host_url', 'host_since', 'host_location', 'host_about', 'host_response_time',
   'host_response_rate', 'host_is_superhost', 'host_thumbnail_url', 'host_picture_url', 'host_neighbourhood',
   'host_listings_count', 'host_total_listings_count', 'host_verifications', 'host_has_profile_pic',
   'host_identity_verified', 'street', 'neighbourhood_cleansed', 'neighbourhood_group_cleansed', 'city', 'state',
   'zipcode', 'market', 'smart_location', 'country_code', 'country', 'is_location_exact', 'property_type',
   'accommodates', 'bathrooms', 'bedrooms', 'beds', 'bed_type', 'amenities', 'square_feet', 'weekly_price',
   'monthly_price', 'security_deposit', 'cleaning_fee', 'guests_included', 'extra_people', 'maximum_nights',
   'minimum_minimum_nights', 'maximum_minimum_nights', 'minimum_maximum_nights', 'maximum_maximum_nights',
   'minimum_nights_avg_ntm', 'maximum_nights_avg_ntm', 'calendar_updated', 'has_availability', 'availability_30',
   'availability_60', 'availability_90', 'calendar_last_scraped', 'number_of_reviews_ltm', 'first_review',
   'review_scores_rating', 'review_scores_accuracy', 'review_scores_cleanliness', 'review_scores_checkin',
   'review_scores_communication', 'review_scores_location', 'review_scores_value', 'requires_license', 'license',
   'instant_bookable', 'is_business_travel_ready', 'cancellation_policy', 'require_guest_profile_picture',
   'require_guest_phone_verification', 'calculated_host_listings_count_entire_homes',
   'calculated_host_listings_count_private_rooms', 'calculated_host_listings_count_shared_rooms']

"""
import ast
from pathlib import Path

import pandas as pd
from pandas import DataFrame
from sklearn.preprocessing import LabelEncoder

from src.service.DataService import DataService


def load_listings(parent, verbose=True):
    df1 = pd.read_csv(
        Path.cwd().parents[parent].joinpath("data/raw", "listings.csv"))
    if verbose:
        print(df1.head())
        print(df1.info())

    # Convert object type columns to string
    df1 = DataService.convert_object_to_string(df1)
    # convert the 'last_review' column to a date data type handling missing values
    df1['last_review'] = pd.to_datetime(
        df1['last_review'], errors='coerce')
    if verbose:
        print(df1.info())

    # Impute missing values
    if verbose:
        print(f"Nullable: {df1.isna().sum()}")
    # Fill NaN values with 0.0 make sense where number_of_reviews == 0
    df1['reviews_per_month'] = df1['reviews_per_month'].fillna(0.0)
    # Interpolate missing date values
    if verbose:
        print(df1['last_review'].describe())
    df1['last_review'] = df1['last_review'].interpolate(method='linear')
    # Fill NA values with constant NOBODY
    df1['host_name'] = df1['host_name'].fillna('Nobody')
    # Fill NA values with value from host_id as string
    df1['name'] = df1['name'].fillna(str(df1['host_id']))
    if verbose:
        print(f"Nullable: {df1.isna().sum()}")
        print(f"Duplicated: {print(df1.duplicated().sum())}")

    return df1


def load_listings_desc(parent, verbose=True):
    df2 = pd.read_csv(
        Path.cwd().parents[parent].joinpath("data/raw", "listings.csv.gz"))
    if verbose:
        print(df2.head())

    # Convert object type columns to string
    df2 = DataService.convert_object_to_string(df2)
    if verbose:
        print(df2.info())

    return df2


def get_full_listings(parent, verbose=True):
    m_df = DataService.merge_dataset(
        load_listings(parent, verbose), load_listings_desc(parent, verbose), on='id')

    # Save the merged DataFrame to a new dataframe with the same name
    m_df = DataFrame(m_df)

    return m_df


def get_dataset_for_training():
    df = process_full_listings(parent=1)

    # drop
    df.drop(columns=[
        'id', 'host_id', 'name', 'host_name', 'last_review',
        'listing_url', 'neighborhood_overview', 'notes', 'transit',
        'scrape_id', 'last_scraped', 'summary', 'description',
        'access', 'interaction', 'house_rules', 'picture_url',
        'host_url', 'host_since', 'host_about', 'host_thumbnail_url',
        'host_picture_url', 'host_verifications', 'amenities',
        'calendar_updated', 'calendar_last_scraped', 'first_review'
    ], inplace=True)

    # Initialize LabelEncoder
    label_encoder = LabelEncoder()
    # Apply LabelEncoder to the categorical column
    for column in df.columns:
        if not pd.api.types.is_numeric_dtype(df[column]):
            try:
                df[column] = (
                    label_encoder.fit_transform(df[column]))
                print(f"{column} ==> Added\n")
            except:
                # drop meaningless column cause all the values are null
                df.drop(column, axis=1, inplace=True)
                print(f"{column} ==> Dropped\n")

    if df.isna().any().any():
        print(df.columns[df.isna().any()].tolist())
        raise ValueError()

    return df


def process_full_listings(parent=0, verbose=True):
    # Load the data files into Python using libraries like pandas
    merged_df = get_full_listings(parent, verbose)

    # for each column in the DataFrame that is null
    for column in merged_df.columns:
        if verbose:
            print(f"\n{column.upper()}\n{'=' * 20}\n{merged_df[column].describe()}")
            print(f"Number of null values: {merged_df[column].isna().sum()}\n")
            print(f"Duplicated: {print(merged_df[column].duplicated().sum())}")

        # if the column has null values
        if (merged_df[column].isna().any() or
                merged_df[column].isnull().any() or
                merged_df[column].eq('<NA>').any()):
            # fill the column with an empty string
            if column in ['host_about', 'host_response_time', 'summary', 'space', 'description',
                          'host_neighbourhood', 'neighborhood_overview', 'notes', 'transit', 'access',
                          'interaction', 'house_rules', 'license', 'city', 'zipcode', 'first_review']:
                merged_df[column] = merged_df[column].fillna('')
                if verbose:
                    print("==> Filled with empty string\n")
            # fill the column with mean value
            elif column in ['state']:
                merged_df[column] = merged_df[column].ffill()
                if verbose:
                    print("==> Filled with mean value\n")
            # fill the column with 0%
            elif column in ['host_response_rate']:
                merged_df[column] = merged_df[column].fillna('0%')
                if verbose:
                    print("==> Filled with '0%'\n")
            # fill the column with 0.0
            elif column in ['square_feet', 'review_scores_accuracy', 'review_scores_cleanliness', 'review_scores_value',
                            'review_scores_communication', 'review_scores_location', 'review_scores_checkin',
                            'review_scores_rating', 'host_listings_count', 'host_total_listings_count',
                            'bathrooms', 'bedrooms', 'beds']:
                merged_df[column] = merged_df[column].fillna('0.0')
                if verbose:
                    print("==> Filled with '0.0'\n")
            # fill the column with $0.0
            elif column in ['weekly_price', 'monthly_price', 'security_deposit',
                            'cleaning_fee', 'extra_people']:
                merged_df[column] = merged_df[column].fillna('$0.0')
                if verbose:
                    print("==> Filled with '$0.0'\n")
            # fill the column with an empty list
            elif column in ['host_verifications']:
                merged_df[column] = merged_df[column].fillna('[]')
                if verbose:
                    print("==> Filled with empty list\n")

    # Drop meaningless columns
    for column in ['host_acceptance_rate', 'experiences_offered', 'thumbnail_url',
                   'medium_url', 'xl_picture_url', 'jurisdiction_names',
                   'calculated_host_listings_count_entire_homes', 'calculated_host_listings_count_private_rooms',
                   'calculated_host_listings_count_shared_rooms', 'country_code', 'country']:
        # drop meaningless column cause all the values are null
        merged_df.drop(columns=[column], inplace=True)
        if verbose:
            print(f"{column} ==> Dropped\n")

    # Convert objects columns to category if necessary
    # The categorical features have to be converted internally to numerical features for efficient modeling
    for column in ['host_response_time', 'host_neighbourhood', 'neighbourhood_group',
                   'neighbourhood', 'room_type', 'amenities']:
        merged_df[column] = merged_df[column].astype("category")
        if verbose:
            print(merged_df[column].cat.codes)
            print(f"{column} ==> Converted to category type\n")

    # Convert string to datetime if necessary
    for column in ['host_since', 'first_review', 'last_review', 'last_scraped',
                   'calendar_last_scraped']:
        merged_df[column] = pd.to_datetime(merged_df[column], errors='coerce')
        # Fill missing values with desired method (e.g., fill forward)
        merged_df[column] = merged_df[column].bfill()
        if verbose:
            print(f"{column} ==> Converted to datetime and filling missing values\n")

    # Convert 't' and 'f' to boolean
    for column in ['host_is_superhost', 'host_has_profile_pic', 'host_identity_verified', 'is_location_exact',
                   'has_availability', 'instant_bookable', 'is_business_travel_ready', 'require_guest_profile_picture',
                   'require_guest_phone_verification', 'requires_license']:
        merged_df[column] = merged_df[column].map({'t': True, 'f': False})
        if verbose:
            print(f"{column} ==> Converted 't' and 'f' to boolean\n")

    # Convert the 'host_verifications' column to a list
    merged_df['host_verifications'] = merged_df['host_verifications'].apply(ast.literal_eval)
    if verbose:
        print("host_verifications ==> Converted to list\n")

    # Convert object type columns to string
    merged_df = DataService.convert_object_to_string(merged_df)
    if verbose:
        print(merged_df.info())
        for column in merged_df.columns:
            print(f"\n{column.upper()}\n{'=' * 20}\n{merged_df[column].describe()}")
            print(f"Number of null values: {merged_df[column].isna().sum()}\n")
            print(f"Duplicated: {print(merged_df[column].duplicated().sum())}")

    merged_df.to_csv(
        Path.cwd().parents[parent].joinpath(
            "data/processed", "listings.csv.gz"), index=False, compression="gzip")

    return merged_df


if __name__ == '__main__':
    process_full_listings()
