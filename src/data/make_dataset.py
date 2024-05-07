# -*- coding: utf-8 -*-
from pathlib import Path

import click
from dotenv import find_dotenv, load_dotenv

from src.listing_etl_pipeline import get_full_listings
from src.review_etl_pipeline import get_full_reviews
from src.service.RentalLogger import logger


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    # load listings
    logger.info(f"Load processed listings from: {input_filepath}")
    listing_df = get_full_listings(input_filepath)
    listing_path = Path.cwd().parents[0].joinpath(output_filepath, 'listings.csv.gz')
    logger.info(f"Saving to {listing_path}...")
    listing_df.to_csv(listing_path, index=False, compression="gzip")

    # load reviews
    logger.info(f"Load processed reviews from: {input_filepath}")
    reviews_df = get_full_reviews(input_filepath)
    reviews_path = Path.cwd().parents[0].joinpath(output_filepath, 'reviews.csv.gz')
    logger.info(f"Saving to {reviews_path}...")
    reviews_df.to_csv(reviews_path, index=False, compression="gzip")


if __name__ == '__main__':
    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
