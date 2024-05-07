# -*- coding: utf-8 -*-
from pathlib import Path
from typing import Any, List

import click
import numpy as np
from dotenv import find_dotenv, load_dotenv
from pandas import DataFrame
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

from src.listing_etl_pipeline import process_full_listings
from src.review_etl_pipeline import process_full_reviews
from src.service.DataService import DataService
from src.service.EmbeddingService import EmbeddingService
from src.service.RentalLogger import logger
from src.service.TrainService import TrainService


def predict_with_tuned_model(model: Any, X_test: List) -> np.array:
    """
    Make predictions using the tuned model

    Args:
        model: The trained model
        X_test: The test data

    Returns:
        y_pred: The predicted values
    """
    # Make predictions using the tuned model
    y_pred = model.predict(X_test)
    return y_pred


def create_xgb_regressor(
        root_dir: Path,
        colsample_bytree: float,
        min_child_weight: int,
        subsample: float,
        n_estimators: int,
        max_depth: int,
        learning_rate: float,
        gamma: float = 0,
) -> Any:
    """
    Create the best model using the best hyperparameters

    Args:
        root_dir: The project directory
        colsample_bytree: The column subsample
        min_child_weight: The minimum child weight
        subsample: The subsample
        n_estimators: The number of estimators
        max_depth: The maximum depth
        learning_rate: The learning rate
        gamma: The gamma

    Returns:
        model: The best model
    """
    model = XGBRegressor(
        colsample_bytree=colsample_bytree,
        min_child_weight=min_child_weight,
        subsample=subsample,
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        gamma=gamma
    )

    model = DataService.load_model(model, root_dir, parent=1, apex='price_train')

    return model


def predict_price_model(output_filepath: str, parent: int, root_dir: Path) -> None:
    """
    The chosen model was an XGBoost regression model with the following hyperparameters:
    n_estimators=400,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.9,
    colsample_bytree=1.0,
    min_child_weight=3,
    gamma=0

    Args:
        output_filepath: The output filepath
        parent: The parent process ID
        root_dir: The project directory

    Returns:
        None
    """

    # load data
    listing_df = process_full_listings(
        store_path=output_filepath, cached=True, parent=parent)

    # pre-processing for training
    Xtr, Xts, ytr, yts = TrainService.pre_process_for(listing_df)

    # Create the best model using the best hyperparameters
    params = {
        'colsample_bytree': 1.0,
        'gamma': 0,
        'learning_rate': 0.1,
        'max_depth': 5,
        'min_child_weight': 3,
        'n_estimators': 400,
        'subsample': 0.9
    }
    best_model = create_xgb_regressor(root_dir, **params)

    # Make predictions using the tuned model
    y_pred = predict_with_tuned_model(best_model, Xts)

    logger.info("Evaluate the model's metrics")
    TrainService.rmse(best_model, yts, y_pred)
    TrainService.mape(best_model, yts, y_pred)
    TrainService.r2(best_model, yts, y_pred)


def predict_review_by_score_model(review_df: DataFrame, root_dir: Path, parent: int = 0) -> Any:
    """
    The chosen model was an Isolation Forest model

    Args:
        review_df: The review dataframe
        root_dir: The project directory
        parent: The parent process ID

    Returns:
        model: The best model
    """

    # Load the trained model
    classifier = LogisticRegression()
    classifier = DataService.load_model(
        classifier, root_dir, parent, apex='review_train_score_str')

    # embedding reviews
    emb_service = EmbeddingService()
    listings_reviews_df = emb_service.get_embeddings(review_df, 'comments')

    # Split the data into training and testing sets
    embeddings = listings_reviews_df['comments_emb'].tolist()
    labels = listings_reviews_df['review_scores_value'].tolist()
    _, X_test, _, y_test = train_test_split(
        embeddings, labels, test_size=0.2, random_state=42)

    # Make predictions using the trained model
    y_pred = classifier.predict(X_test)

    # Evaluate the model's accuracy
    accuracy = accuracy_score(y_test, y_pred)
    logger.info(f"Accuracy: {accuracy:.2f}")


@click.command()
@click.argument('output_filepath', type=click.Path())
def main(output_filepath):
    review_df = process_full_reviews(
        store_path=output_filepath, cached=True, parent=1)

    predict_price_model(output_filepath, parent=1, root_dir=project_dir)
    predict_review_by_score_model(review_df, project_dir, parent=1)


if __name__ == '__main__':
    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
