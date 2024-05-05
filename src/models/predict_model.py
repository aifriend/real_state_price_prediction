from typing import Any, List

import numpy as np
import torch
from pandas import DataFrame
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoModel
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
        colsample_bytree: float,
        min_child_weight: int,
        subsample: float,
        n_estimators: int,
        max_depth: int,
        learning_rate: float,
        gamma: float = 0
) -> Any:
    """
    Create the best model using the best hyperparameters

    Args:
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
    return model


def predict_price_model(parent: int = 1) -> None:
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
        parent: The parent process ID

    Returns:
        None
    """
    listing_df = process_full_listings(parent=parent)

    # X and y dataset preparation
    _, Xts, _, yts = TrainService.TrainSampleModel(listing_df)

    # X = listing_df.drop(['price'], axis=1).values
    # y = listing_df['price'].values
    # logger.info(X.shape)
    # logger.info(y.shape)

    # _, Xts, _, yts = train_test_split(
    #     X, y, test_size=0.3, random_state=42)

    params = {
        'colsample_bytree': 1.0,
        'gamma': 0,
        'learning_rate': 0.1,
        'max_depth': 5,
        'min_child_weight': 3,
        'n_estimators': 400,
        'subsample': 0.9
    }

    # Create the best model using the best hyperparameters
    best_model = create_xgb_regressor(**params)

    # Make predictions using the tuned model
    y_pred = predict_with_tuned_model(best_model, Xts)

    logger.info("Evaluate the model's accuracy")
    accuracy = accuracy_score(yts, y_pred)
    logger.info(f"Accuracy: {accuracy:.2f}")


def predict_review_by_score_model(parent: int = 0) -> Any:
    """
    The chosen model was an Isolation Forest model

    Args:
        parent: The parent process ID

    Returns:
        model: The best model
    """
    # Load listings reviews
    reviews_df = process_full_reviews()
    emb_service = EmbeddingService()
    listings_reviews_df = emb_service.get_embeddings(reviews_df, 'comments')

    # Split the data into training and testing sets
    embeddings = listings_reviews_df['comments_emb'].tolist()
    labels = listings_reviews_df['review_scores_value'].tolist()
    _, X_test, _, y_test = train_test_split(
        embeddings, labels, test_size=0.2, random_state=42)

    # Make predictions using the trained model
    classifier = DataService.load_model(
        'classifier', parent, apex='train_reviews_by_score')
    y_pred = classifier.predict(X_test)

    # Evaluate the model's accuracy
    accuracy = accuracy_score(y_test, y_pred)
    logger.info(f"Accuracy: {accuracy:.2f}")

    # Use the trained model to predict house rental based on new text reviews
    new_reviews = ["This house is amazing!", "The rental experience was terrible."]
    new_embeddings = emb_service.model.encode(new_reviews)
    new_predictions = classifier.predict(new_embeddings)
    logger.info(new_predictions)


def predict_review_outliers():
    """
    use the Isolation Forest algorithm to detect outlier reviews based on their embeddings.
    The Isolation Forest is an unsupervised learning algorithm that isolates anomalies by
    randomly selecting features and splitting the data recursively. It works well for
    high-dimensional data like text embeddings.
    """

    def get_outliers_tfidf_vectorizer(rev_df: DataFrame) -> List:
        """
        Create a TF-IDF vectorizer to generate embeddings

        Args:
            rev_df (DataFrame): reviews dataframe

        Returns:
            review_embeddings (np.ndarray): embeddings for each review
        """
        # Create a TF-IDF vectorizer to generate embeddings
        vectorizer = TfidfVectorizer()

        # Generate embeddings for each review
        review_texts = [review for review in rev_df['comments']]
        review_embeddings = vectorizer.fit_transform(review_texts).toarray()

        return review_embeddings

    def get_outliers_bert_vectorizer(rev_df: DataFrame) -> List:
        """
        Create a BERT vectorizer to generate embeddings

        Args:
            rev_df (DataFrame): reviews dataframe

        Returns:
            review_embeddings (np.ndarray): embeddings for each review
        """

        # Load the BERT tokenizer and model
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        bert_model = BertModel.from_pretrained('bert-base-uncased')

        # Generate BERT embeddings for each review
        review_embeddings = []
        for review in rev_df['comments']:
            encoded_input = tokenizer(review, return_tensors='pt', padding=True, truncation=True)
            with torch.no_grad():
                outputs = bert_model(**encoded_input)
                embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
            review_embeddings.append(embedding)

        # Convert the embeddings to a numpy array
        review_embeddings = np.array(review_embeddings)
        return review_embeddings

    def get_outliers_hf_vectorizer(rev_df: DataFrame) -> List:
        """
        Create a Hugging Face vectorizer to generate embeddings

        Args:
            rev_df (DataFrame): reviews dataframe

        Returns:
            review_embeddings (np.ndarray): embeddings for each review
        """

        # Load a pre-trained tokenizer and model from Hugging Face
        tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
        hf_model = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2')

        # Generate embeddings for each review
        review_embeddings = []
        for review in rev_df['comments']:
            inputs = tokenizer(review, padding=True, truncation=True, return_tensors='pt')
            outputs = hf_model(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1).detach().numpy()
            review_embeddings.append(embedding)

        review_embeddings = np.vstack(review_embeddings)
        return review_embeddings

    def isolate(review_list: List, rev_df: DataFrame, xgb_model: IsolationForest) -> None:
        """
        Train an Isolation Forest model

        Args:
            xgb_model (IsolationForest): Isolation Forest model
            review_list (np.ndarray): embeddings for each review
            rev_df (DataFrame): reviews dataframe

        Returns:
            None
        """
        # Train an Isolation Forest model
        xgb_model.fit(review_list)

        # Get outlier scores for each review
        outlier_scores = model.decision_function(review_list)

        # Identify outliers based on the scores
        outlier_indices = np.where(outlier_scores < 0)[0]
        outlier_reviews = rev_df.iloc[outlier_indices]

        # Print the outlier reviews
        logger.info("Outlier Reviews from TF-IDF:")
        for index, review in outlier_reviews.iterrows():
            logger.info(f"Listing ID: {review['listing_id']}")
            logger.info(f"Review Text: {review['comments']}")
            logger.info()

    # Load listings reviews
    reviews_df = process_full_reviews()
    model = IsolationForest(contamination=0.1, random_state=42)

    # Create a TF-IDF vectorizer to generate embeddings
    review_tfidf_embeddings = get_outliers_tfidf_vectorizer(reviews_df)
    isolate(review_tfidf_embeddings, reviews_df, model)

    # Create a BERT vectorizer to generate embeddings
    review_bert_embeddings = get_outliers_bert_vectorizer(reviews_df)
    isolate(review_bert_embeddings, reviews_df, model)

    # Create a vectorizer from HuggingFace to generate embeddings
    review_glove_embeddings = get_outliers_hf_vectorizer(reviews_df)
    isolate(review_glove_embeddings, reviews_df, model)


if __name__ == '__main__':
    predict_price_model()
    predict_review_by_score_model()
    predict_review_outliers()
