import re
from pathlib import Path

import pandas as pd
from pandas import DataFrame
from sklearn.preprocessing import LabelEncoder

from src.service.DataService import DataService
from src.service.RentalLogger import logger


def _load_listings(data_path: str) -> DataFrame:
    """
    Load listings

    :param data_path: source data path
    :return: dataframe of listings
    """
    listings_path = Path.cwd().parents[0].joinpath(data_path, 'listings.csv')
    if not listings_path.exists():
        raise ValueError("Processed listings not found.")
    logger.info(f"Load processed listings from: {listings_path}")
    df1 = pd.read_csv(listings_path, low_memory=False)
    logger.info(f"{df1.head()}")

    # Convert object type columns to string
    logger.info("Converting listings object type columns to string")
    df1 = DataService.convert_object_to_string(df1)

    return df1


def _load_listings_desc(data_path: str) -> DataFrame:
    """
    Load listing descriptions

    :param data_path: source data path
    :return: dataframe of listing descriptions
    """
    listing_desc_path = Path.cwd().parents[0].joinpath(data_path, 'listings.csv.gz')
    if not listing_desc_path.exists():
        raise ValueError("Processed listing descriptions not found.")
    logger.info(f"Load processed listing descriptions from: {listing_desc_path}")
    df2 = pd.read_csv(listing_desc_path, low_memory=False)
    logger.info(f"{df2.head()}")

    # Convert object type columns to string
    logger.info("Converting listings_desc object type columns to string")
    df2 = DataService.convert_object_to_string(df2)

    return df2


def get_full_listings(data_path: str = 'data/raw') -> DataFrame:
    """
    Get full listings linked with listing descriptions

    :param data_path: source data path
    :return: merged dataframe of listings and listing descriptions
    """
    m_df = DataService.merge_dataset(
        _load_listings(data_path),
        _load_listings_desc(data_path), on='id')

    # Save the merged DataFrame to a new dataframe with the same name
    m_df = DataFrame(m_df)

    return m_df


def process_full_listings(
        data_path='data/interim',
        store_path='data/processed',
        verbose: bool = False,
        cached: bool = True,
        parent: int = 0) -> DataFrame:
    """
    Process full listings linked with listing descriptions

    Args:
        data_path: source data path
        store_path: store data path
        verbose: print logs
        cached: use cached data
        parent: parent directory

    Returns:
        DataFrame: processed listings
    """
    if cached:
        # if the file exists, load the file
        if Path.cwd().parents[parent].joinpath(store_path, "listings.csv.gz").exists():
            logger.info(f"Load processed listings from: "
                        f"{Path.cwd().parents[parent].joinpath(store_path, 'listings.csv.gz')}")
            return pd.read_csv(
                Path.cwd().parents[parent].joinpath(
                    store_path, "listings.csv.gz"), low_memory=False)
        else:
            raise ValueError(f"Processed listings not found at "
                             f"{Path.cwd().parents[parent].joinpath(store_path, 'listings.csv.gz')}")

    logger.info("Load full listings...")
    if not Path.cwd().parents[parent].joinpath(data_path, "listings.csv.gz").exists():
        raise ValueError(
            f"Full listing file missing: "
            f"{Path.cwd().parents[parent].joinpath(data_path, 'listings.csv.gz')}")
    else:
        merged_df = pd.read_csv(
            Path.cwd().parents[parent].joinpath(
                data_path, "listings.csv.gz"), low_memory=False)

    logger.info("Processing listings...")

    # Drop meaningless columns
    merged_df.drop(columns=['host_id', 'name', 'host_name', 'last_review',
                            'listing_url', 'neighborhood_overview', 'notes', 'transit',
                            'scrape_id', 'last_scraped', 'summary', 'description',
                            'access', 'interaction', 'house_rules', 'picture_url',
                            'host_url', 'host_since', 'host_about', 'host_thumbnail_url',
                            'host_picture_url', 'host_verifications', 'amenities',
                            'calendar_updated', 'calendar_last_scraped', 'first_review',
                            'host_acceptance_rate', 'experiences_offered', 'thumbnail_url',
                            'medium_url', 'xl_picture_url', 'jurisdiction_names',
                            'country_code', 'country', 'smart_location',
                            'require_guest_profile_picture', 'requires_license',
                            'require_guest_phone_verification', 'market', 'cancellation_policy',
                            'calculated_host_listings_count_entire_homes',
                            'calculated_host_listings_count_private_rooms',
                            'calculated_host_listings_count_shared_rooms', 'license',
                            'instant_bookable', 'is_business_travel_ready', 'space',
                            'is_location_exact'
                            ],
                   inplace=True)

    if verbose:
        logger.info(f"{merged_df.info()}")
        logger.info(f"Nullable: {merged_df.isna().sum()}")
        logger.info(f"Duplicated: {merged_df.duplicated().sum()}")

    # Impute missing values
    logger.info("Imputing missing values")
    if verbose:
        logger.info(f"Nullable: {merged_df.isna().sum()}")
    # Fill NaN values with 0.0 make sense where number_of_reviews == 0
    merged_df['reviews_per_month'] = merged_df['reviews_per_month'].fillna(0.0)

    # for each column in the DataFrame that is null
    for column in merged_df.columns:
        # if the column has null values
        if (merged_df[column].isna().any() or
                merged_df[column].isnull().any() or
                merged_df[column].eq('<NA>').any()):
            # fill the column with an empty string
            if column in ['space', 'host_neighbourhood', 'city', 'zipcode']:
                merged_df[column] = merged_df[column].fillna('')
                if verbose:
                    logger.info(f"{column} ==> Filled with empty string")
            # fill the column with forward value
            elif column in ['state', 'host_location', 'host_is_superhost',
                            'host_has_profile_pic', 'host_identity_verified']:
                merged_df[column] = merged_df[column].ffill()
                if verbose:
                    logger.info(f"{column} ==> Filled with forward value")
            # fill the column with 0%
            elif column in ['host_response_rate']:
                merged_df[column] = merged_df[column].fillna('0%')
                if verbose:
                    logger.info(f"{column} ==> Filled with '0%'")
            # fill the column with 0.0
            elif column in ['square_feet', 'review_scores_accuracy', 'review_scores_cleanliness', 'review_scores_value',
                            'review_scores_communication', 'review_scores_location', 'review_scores_checkin',
                            'review_scores_rating', 'host_listings_count', 'host_total_listings_count',
                            'bathrooms', 'bedrooms', 'beds']:
                merged_df[column] = merged_df[column].fillna('0.0')
                if verbose:
                    logger.info(f"{column} ==> Filled with '0.0'")
            # fill the column with $0.0
            elif column in ['weekly_price', 'monthly_price', 'security_deposit',
                            'cleaning_fee', 'extra_people']:
                merged_df[column] = merged_df[column].fillna('$0.0')
                if verbose:
                    logger.info(f"{column} ==> Filled with '$0.0'")
            elif column in ['host_is_superhost', 'host_has_profile_pic', 'host_identity_verified']:
                merged_df[column] = merged_df[column].ffill()

    if merged_df['price'].dtypes != 'float64':
        merged_df['price'] = merged_df['price'].astype(float)

    # Convert to number
    def convert_price(price_str: str) -> float:
        """
        Convert price string to number

        Args:
            price_str: price string

        Returns:
            float: price number
        """
        # Remove the currency symbol and convert the string to float
        return float(re.search(r'\d+(\.\d+)?', price_str).group())

    merged_df['weekly_price'] = merged_df['weekly_price'].apply(convert_price)
    merged_df['monthly_price'] = merged_df['monthly_price'].apply(convert_price)
    merged_df['security_deposit'] = merged_df['security_deposit'].apply(convert_price)
    merged_df['cleaning_fee'] = merged_df['cleaning_fee'].apply(convert_price)
    merged_df['extra_people'] = merged_df['extra_people'].apply(convert_price)
    merged_df['host_response_rate'] = merged_df['host_response_rate'].apply(
        lambda x: float(re.search(r'\d+(\.\d+)?', x).group())
    )

    # Convert 't' and 'f' to boolean
    for column in ['host_is_superhost', 'host_has_profile_pic',
                   'host_identity_verified', 'has_availability']:
        merged_df[column] = merged_df[column].map({'t': True, 'f': False})
        if verbose:
            logger.info(f"{column} ==> Converted 't' and 'f' to boolean")

    # Convert objects columns to category if necessary
    # The categorical features have to be converted internally to numerical features for efficient modeling
    for column in ['host_response_time', 'host_neighbourhood', 'neighbourhood_group',
                   'neighbourhood', 'room_type']:
        merged_df[column] = merged_df[column].astype("category")
        if verbose:
            logger.info(merged_df[column].cat.codes)
            logger.info(f"{column} ==> Converted to category type")

    # Apply LabelEncoder to the categorical column
    label_encoder = LabelEncoder()
    for column in ['host_response_time']:
        encoded_categories = label_encoder.fit_transform(merged_df[column])
        category_mapping = dict(zip(merged_df[column], encoded_categories))
        merged_df[column] = encoded_categories
        if verbose:
            logger.info(f"{column} ==> "
                        f"Apply label encoder with label "
                        f"{category_mapping}")

    # Remove outliers
    # Calculate the IQR
    q1 = merged_df['price'].quantile(0.25)
    q3 = merged_df['price'].quantile(0.75)
    iqr = q3 - q1
    outliers = (merged_df['price'] <
                (q1 - 1.5 * iqr)) | (merged_df['price'] > (q3 + 1.5 * iqr))
    merged_df = merged_df[~outliers]

    if verbose:
        logger.info(f"{merged_df.info()}")
        for column in merged_df.columns:
            logger.info(f"\n{column.upper()}\n{'=' * 20}\n{merged_df[column].describe()}")
            logger.info(f"Number of null values: {merged_df[column].isna().sum()}")
            logger.info(f"Duplicated: {merged_df[column].duplicated().sum()}")

        # Check if there are any null values
        if merged_df.isna().any().any():
            logger.info(merged_df.columns[merged_df.isna().any()].tolist())
            raise ValueError()

    # save processed listings
    merged_df.to_csv(
        Path.cwd().parents[parent].joinpath(
            store_path, "listings.csv.gz"), index=False, compression="gzip")
    logger.info(f"Saved to: {Path.cwd().parents[parent].joinpath(store_path, 'listings.csv.gz')}")

    return merged_df


if __name__ == '__main__':
    parent = 0

    # create linked listing
    listing_df = get_full_listings(data_path='data/raw')
    listing_path = Path.cwd().parents[parent].joinpath('data/interim', 'listings.csv.gz')
    listing_df.to_csv(listing_path, index=False, compression="gzip")

    # process full listing
    process_full_listings(
        data_path='data/interim', cached=False, parent=parent)
