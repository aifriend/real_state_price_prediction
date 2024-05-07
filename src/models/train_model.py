# -*- coding: utf-8 -*-
from pathlib import Path

import click
import numpy as np
import torch
from dotenv import find_dotenv, load_dotenv
from pandas import DataFrame
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoModel

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


def predict_review_outliers(review_df: DataFrame) -> None:
    """
    use the Isolation Forest algorithm to detect outlier reviews based on their embeddings.
    The Isolation Forest is an unsupervised learning algorithm that isolates anomalies by
    randomly selecting features and splitting the data recursively. It works well for
    high-dimensional data like text embeddings.

    Args:
        review_df: The reviews dataframe

    Returns:
        None
    """

    def get_outliers_tfidf_vectorizer(rev_df: DataFrame) -> np.ndarray:
        """
        Create a TF-IDF vectorizer to generate embeddings

        Args:
            rev_df (DataFrame): reviews dataframe

        Returns:
            review_embeddings (np.ndarray): embeddings for each review
        """
        logger.info("Create a TF-IDF vectorizer to generate embeddings")
        # Create a TF-IDF vectorizer to generate embeddings
        vectorizer = TfidfVectorizer()

        # Generate embeddings for each review
        review_texts = [review for review in rev_df['comments']]
        review_embeddings = vectorizer.fit_transform(review_texts).toarray()

        return review_embeddings

    def get_outliers_bert_vectorizer(rev_df: DataFrame) -> np.ndarray:
        """
        Create a BERT vectorizer to generate embeddings

        Args:
            rev_df (DataFrame): reviews dataframe

        Returns:
            review_embeddings (np.ndarray): embeddings for each review
        """
        logger.info("Create a BERT vectorizer to generate embeddings")

        # Load the BERT tokenizer and model
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        bert_model = BertModel.from_pretrained('bert-base-uncased')

        # Generate BERT embeddings for each review
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        bert_model.to(device)
        batch_size = 64
        num_reviews = len(rev_df['comments'])
        review_embeddings = np.zeros((num_reviews, bert_model.config.hidden_size))
        for i in tqdm(range(0, num_reviews, batch_size)):
            batch_reviews = rev_df['comments'][i:i + batch_size]
            encoded_input = tokenizer.batch_encode_plus(
                batch_reviews, return_tensors='pt', padding=True, truncation=True)
            encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
            with torch.no_grad():
                outputs = bert_model(**encoded_input)
                embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            review_embeddings[i:i + len(batch_reviews)] = embeddings

        # Convert the embeddings to a numpy array
        review_embeddings = np.array(review_embeddings)
        return review_embeddings

    def get_outliers_hf_vectorizer(rev_df: DataFrame) -> np.ndarray:
        """
        Create a Hugging Face vectorizer to generate embeddings

        Args:
            rev_df (DataFrame): reviews dataframe

        Returns:
            review_embeddings (np.ndarray): embeddings for each review
        """
        logger.info(f"Create a Hugging Face vectorizer to generate embeddings")

        # Load a pre-trained tokenizer and model from Hugging Face
        tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
        hf_model = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2')

        # Generate embeddings for each review
        review_embeddings = np.zeros((len(rev_df['comments']), hf_model.config.hidden_size))
        batch_size = 64
        num_reviews = len(rev_df['comments'])
        for i in tqdm(range(0, num_reviews, batch_size)):
            batch_reviews = rev_df['comments'][i:i + batch_size].tolist()
            inputs = tokenizer(batch_reviews, padding=True, truncation=True, return_tensors='pt')
            outputs = hf_model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1).detach().numpy()
            review_embeddings[i:i + batch_size] = embeddings

        review_embeddings = np.vstack(review_embeddings)
        return review_embeddings

    def isolate(model_name: str, review_list: np.ndarray, rev_df: DataFrame, xgb_model: IsolationForest) -> None:
        """
        Train an Isolation Forest model

        Args:
            model_name (str): Model name
            xgb_model (IsolationForest): Isolation Forest model
            review_list (np.ndarray): embeddings for each review
            rev_df (DataFrame): reviews dataframe

        Returns:
            None
        """
        logger.info(f"Train an Isolation Forest model for {model_name}")

        # Train an Isolation Forest model
        xgb_model.fit(review_list)

        # Get outlier scores for each review
        outlier_scores = model.decision_function(review_list)

        # Identify outliers based on the scores
        outlier_indices = np.where(outlier_scores < 0)[0]
        outlier_reviews = rev_df.iloc[outlier_indices]

        # Print the outlier reviews
        logger.info(f"Outlier Reviews from {model_name}:")
        for index, review in outlier_reviews.iterrows():
            logger.info(f"Listing ID: {review['listing_id']}")
            logger.info(f"Review Text: {review['comments']}")

    model = IsolationForest(contamination=0.1, random_state=42)

    # # Create a TF-IDF vectorizer to generate embeddings
    # review_tfidf_embeddings = get_outliers_tfidf_vectorizer(review_df)
    # isolate('TF-IDF', review_tfidf_embeddings, review_df, model)
    #
    # # Create a BERT vectorizer to generate embeddings
    # review_bert_embeddings = get_outliers_bert_vectorizer(review_df)
    # isolate('BERT', review_bert_embeddings, review_df, model)

    # Create a vectorizer from HuggingFace to generate embeddings
    review_glove_embeddings = get_outliers_hf_vectorizer(review_df)
    isolate('HuggingFace', review_glove_embeddings, review_df, model)


@click.command()
@click.argument('output_filepath', type=click.Path())
def main(output_filepath):
    review_df = process_full_reviews(
        store_path=output_filepath, cached=True, parent=1)

    train_price_model(output_filepath, project_dir, parent=1)
    TrainService.train_reviews_by_score(review_df, project_dir, parent=1)
    TrainService.train_review_sentiment_analysis(review_df, project_dir, parent=1)
    predict_review_outliers(review_df)


if __name__ == '__main__':
    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
