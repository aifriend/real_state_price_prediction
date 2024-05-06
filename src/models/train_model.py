# -*- coding: utf-8 -*-
import logging
from pathlib import Path

import click
from dotenv import find_dotenv, load_dotenv

from src.listing_etl_pipeline import process_full_listings
from src.review_etl_pipeline import process_full_reviews
from src.service.RentalLogger import logger
from src.service.TrainService import TrainService


def train_price_model(output_filepath: str, root_dir: Path, parent: int = 0, optimization=False) -> None:
    """
    Train price model

    Args:
        output_filepath: str
        root_dir: Path
        parent: int
        optimization: bool

    Returns:
        None
    """
    # load data
    listing_df = process_full_listings(
        store_path=output_filepath, cached=True, parent=parent)

    # pre-processing for training
    Xtr, Xts, ytr, yts = TrainService.pre_process_for(listing_df)

    # Train
    xgb, resc_x_tr = TrainService.train(Xtr, Xts, ytr, yts, root_dir)

    # Tuning
    b_model = None
    if optimization:
        b_model, b_score, b_params = TrainService.xgboost_tuning(resc_x_tr, ytr)
        logger.info("Grid search completed.")
        logger.info(f"Best hyperparameters: {b_params}")
        logger.info(f"Best score: {b_score}")

    return xgb if xgb else b_model


@click.command()
@click.argument('output_filepath', type=click.Path())
def main(output_filepath):
    logger = logging.getLogger(__name__)
    logger.info('Training models...')

    train_price_model(output_filepath, project_dir, parent=1)

    review_df = process_full_reviews(
        store_path=output_filepath, cached=True, parent=1)

    TrainService.train_reviews_by_score(review_df, project_dir, parent=1)
    TrainService.train_review_sentiment_analysis(review_df, project_dir, parent=1)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
