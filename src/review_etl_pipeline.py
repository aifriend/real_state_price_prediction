from pathlib import Path

import pandas as pd
from pandas import DataFrame

from src.listing_etl_pipeline import process_full_listings
from src.service.DataService import DataService
from src.service.NlpService import NlpService
from src.service.RentalLogger import logger


def _load_reviews(data_path: str) -> DataFrame:
    """
    Load reviews

    :param data_path: source data path
    :return: dataframe of reviews
    """
    reviews_path = Path.cwd().parents[0].joinpath(data_path, "reviews.csv")
    if not reviews_path.exists():
        raise ValueError("Processed reviews not found.")
    logger.info(f"Load processed reviews from: {reviews_path}")
    df1 = pd.read_csv(reviews_path, low_memory=False)
    logger.info(f"{df1.head()}")

    # Convert object type columns to string
    logger.info("Converting listings object type columns to string")
    df1 = DataService.convert_object_to_string(df1)

    # convert the 'last_review' column to a date data type handling missing values
    df1['date'] = pd.to_datetime(df1['date'], errors='coerce')

    # Check missing values
    logger.info(df1.isna().sum())

    return df1


def _load_reviews_desc(data_path: str) -> DataFrame:
    review_desc_path = Path.cwd().parents[0].joinpath(data_path, 'reviews.csv.gz')
    if not review_desc_path.exists():
        raise ValueError("Processed review descriptions not found.")
    logger.info(f"Load processed review descriptions from: {review_desc_path}")
    df2 = pd.read_csv(review_desc_path, low_memory=False)
    logger.info(f"{df2.head()}")

    # Convert object type columns to string
    logger.info("Converting listings_desc object type columns to string")
    df2 = DataService.convert_object_to_string(df2)

    # convert the 'last_review' column to a date data type handling missing values
    df2['date'] = pd.to_datetime(df2['date'], errors='coerce')

    # Impute missing values
    logger.info(df2.isna().sum())
    # fill reviewer_name with 'None'
    df2['reviewer_name'] = df2['reviewer_name'].fillna('None')
    # fill comments with empty string
    df2['comments'] = df2['comments'].fillna('')
    # remove duplicated
    df2 = df2.drop_duplicates()
    logger.info(df2.isna().sum())

    return df2


def get_full_reviews(data_path: str) -> DataFrame:
    """
    Get full reviews linked with review descriptions

    :param data_path: source data path
    :return: merged dataframe of reviews and review descriptions
    """
    # m_df = DataService.merge_dataset(
    #     _load_reviews(data_path),
    #     _load_reviews_desc(data_path), on='listing_id')
    #
    # # Save the merged DataFrame to a new dataframe with the same name
    # m_df = DataFrame(m_df)

    return _load_reviews_desc(data_path)


def process_full_reviews(
        data_path: str = 'data/interim',
        store_path: str = 'data/processed',
        verbose: bool = False,
        cached: bool = True,
        parent: int = 0):
    """
    Process full reviews linked with review descriptions

    :param data_path: source data path
    :param store_path: destination data path
    :param verbose: print logs
    :param cached: set to True to use cached data already processed
    :param parent: parent directory
    :return: processed reviews
    """
    if cached:
        # if file exists, load it
        if Path(Path.cwd().parents[parent].joinpath(store_path, "reviews.csv.gz")).exists():
            logger.info(f"Load processed reviews from: "
                        f"{Path.cwd().parents[parent].joinpath(store_path, 'reviews.csv.gz')}")
            return pd.read_csv(
                Path.cwd().parents[parent].joinpath(
                    store_path, "reviews.csv.gz"), low_memory=False)
        else:
            raise ValueError(f"Processed reviews not found at "
                             f"{Path.cwd().parents[parent].joinpath(store_path, 'reviews.csv.gz')}")

    # get full listing
    listings_df = process_full_listings(
        data_path=data_path, store_path=store_path, parent=parent)

    # get full reviews
    if not Path.cwd().parents[parent].joinpath(data_path, "reviews.csv.gz").exists():
        raise ValueError(
            f"Full reviews file missing: "
            f"{Path.cwd().parents[parent].joinpath(data_path, 'reviews.csv.gz')}")
    else:
        reviews_df = pd.read_csv(
            Path.cwd().parents[parent].joinpath(
                data_path, "reviews.csv.gz"), low_memory=False)

    # merge listings and reviews
    reviews_listings_df = pd.merge(
        listings_df, reviews_df, left_on='id', right_on='listing_id', how='inner')

    # random sampling
    reviews_listings_df = reviews_listings_df[(reviews_listings_df['comments'].str.len() > 100)]
    reviews_listings_df = reviews_listings_df[(reviews_listings_df['comments'].str.len() < 200)]

    # sampling
    reviews_listings_df = reviews_listings_df.sample(50000)  # TODO: optimization

    logger.info(f"Process {len(reviews_listings_df)} reviews...")
    merged_df = NlpService.process_reviews(reviews_listings_df)  # feature extraction

    # fill comments with empty string
    merged_df.loc[:, 'comments'] = merged_df['comments'].bfill()

    # remove duplicated
    merged_df = merged_df.drop_duplicates()

    # remove empty and too big
    merged_df = merged_df[(merged_df['comments'].str.len() > 100)]
    merged_df = merged_df[(merged_df['comments'].str.len() < 200)]

    # remove non string
    merged_df = merged_df[
        merged_df['comments'].apply(lambda x: isinstance(x, str))]

    # save processed reviews
    merged_df.to_csv(
        Path.cwd().parents[parent].joinpath(
            store_path, "reviews.csv.gz"), index=False, compression="gzip")
    logger.info(f"Saved to: {Path.cwd().parents[parent].joinpath(store_path, 'reviews.csv.gz')}")

    return merged_df


if __name__ == '__main__':
    # create linked reviews
    review_df = get_full_reviews(data_path='data/raw')
    review_path = Path.cwd().parents[0].joinpath('data/interim', 'reviews.csv.gz')
    review_df.to_csv(review_path, index=False, compression="gzip")

    # process full listing
    process_full_reviews(
        data_path='data/interim', verbose=True, cached=False)
