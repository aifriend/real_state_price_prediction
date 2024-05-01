import pandas as pd
from sklearn.model_selection import train_test_split

from src.listing_pipeline import get_dataset_for_training
from src.listing_pipeline import process_full_listings
from src.review_pipeline import get_reviews_desc
from src.service.NlpService import NlpService
from src.service.TrainService import TrainService


def train_price_model():
    df = get_dataset_for_training()

    # X and y
    X = df.drop(['price'], axis=1).values
    y = df['price'].values
    print(X.shape)
    print(y.shape)

    # Split the data into training and testing sets
    Xtr, Xts, ytr, yts = train_test_split(
        X, y, test_size=0.3, random_state=42)

    # Train
    xgb, resc_x_tr = TrainService.train(Xtr, Xts, ytr, yts)

    # Tuning
    b_model, b_score, b_params = TrainService.xgboost_tuning(resc_x_tr, ytr)
    print("Grid search completed.")
    print("Best hyperparameters: ", b_params)
    print("Best score: ", b_score)

    return xgb if xgb else b_model


def train_reviews_by_score(parent=1):
    print("Load listings reviews")
    reviews_df = get_reviews_desc(parent=1, cached=True)
    listings_df = process_full_listings(parent=1, verbose=False)
    reviews_listings_df = pd.merge(
        listings_df, reviews_df, left_on='id', right_on='listing_id', how='inner')

    # Preprocess reviews
    rev_df = NlpService.process_reviews(reviews_listings_df, parent)

    # Train
    TrainService.train_reviews_by_score(rev_df, parent)


def train_review_sentiment_analysis(parent=1):
    print("Load listings reviews")
    reviews_df = get_reviews_desc(parent=1, cached=True)
    listings_df = process_full_listings(parent=1, verbose=False)
    reviews_listings_df = pd.merge(
        listings_df, reviews_df, left_on='id', right_on='listing_id', how='inner')

    # Preprocess reviews
    rev_df = NlpService.process_reviews(reviews_listings_df, parent)

    # Train
    TrainService.train_review_sentiment_analysis(rev_df, parent)


if __name__ == '__main__':
    train_price_model()
    train_reviews_by_score()
    train_review_sentiment_analysis()
