# -*- coding: utf-8 -*-
import logging
from pathlib import Path

import click
from dotenv import find_dotenv, load_dotenv

from src.listing_etl_pipeline import get_full_listings
from src.review_etl_pipeline import get_full_reviews


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    # load listings
    logger.info(f"Load processed listings from: {input_filepath}")
    listing_df = get_full_listings(input_filepath)
    listing_path = Path.cwd().parents[0].joinpath(output_filepath, 'listings.csv.gz')
    listing_df.to_csv(listing_path, index=False, compression="gzip")
    logger.info(f"Saved to: {listing_path}")

    # load reviews
    logger.info(f"Load processed reviews from: {input_filepath}")
    reviews_df = get_full_reviews(input_filepath)
    reviews_path = Path.cwd().parents[0].joinpath(output_filepath, 'reviews.csv.gz')
    reviews_df.to_csv(reviews_path, index=False, compression="gzip")
    logger.info(f"Saved to: {reviews_path}")


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
