# -*- coding: utf-8 -*-
import logging
from pathlib import Path

import click
from dotenv import find_dotenv, load_dotenv

from src.review_etl_pipeline import process_full_reviews
from src.visualization.visualize import EDA


@click.command()
@click.argument('output_filepath', type=click.Path())
def main(output_filepath):
    """ Runs data analysis scripts to turn features data from (saved in ../processed)
    into meaningful data ready to be shown.
    """
    logger = logging.getLogger(__name__)
    logger.info('I explored the data to check if there are trends between '
                'the explanatory variables and the target variable.')

    dea_df = process_full_reviews(
        store_path=output_filepath, cached=True, verbose=False)

    EDA.feature_correlation_analysis(dea_df)
    EDA.feature_exploratory_analysis(dea_df, output_filepath)
    EDA.neighborhood_analysis(dea_df)
    EDA.review_analysis(dea_df)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
