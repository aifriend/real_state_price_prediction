# -*- coding: utf-8 -*-
from pathlib import Path

import click
from dotenv import find_dotenv, load_dotenv

from src.listing_etl_pipeline import process_full_listings
from src.review_etl_pipeline import process_full_reviews


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    _ = process_full_listings(
        data_path=input_filepath,
        store_path=output_filepath,
        verbose=True,
        cached=False)
    _ = process_full_reviews(
        data_path=input_filepath,
        store_path=output_filepath,
        verbose=True,
        cached=False)


if __name__ == '__main__':
    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
