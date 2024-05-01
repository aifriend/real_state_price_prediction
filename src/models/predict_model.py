import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

from src.listing_pipeline import get_dataset_for_training, process_full_listings
from src.review_pipeline import get_reviews_desc
from src.service.DataService import DataService
from src.service.EmbeddingService import EmbeddingService
from src.service.NlpService import NlpService


def predict_with_tuned_model(model, X_test):
    # Make predictions using the tuned model
    y_pred = model.predict(X_test)
    return y_pred


def create_xgb_regressor(n_estimators, max_depth, learning_rate):
    model = XGBRegressor(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate)
    return model


def predict_price_model():
    df = get_dataset_for_training()

    X = df.drop(['price'], axis=1).values
    y = df['price'].values
    print(X.shape)
    print(y.shape)

    _, Xts, _, yts = train_test_split(
        X, y, test_size=0.3, random_state=42)

    params = {
        'colsample_bytree': 0.9,
        'gamma': 0,
        'learning_rate': 0.1,
        'max_depth': 5,
        'min_child_weight': 2,
        'n_estimators': 500,
        'subsample': 0.8
    }

    # Create the best model using the best hyperparameters
    best_model = create_xgb_regressor(**params)

    # Make predictions using the tuned model
    y_pred = predict_with_tuned_model(best_model, Xts)

    print("Evaluate the model's accuracy")
    accuracy = accuracy_score(yts, y_pred)
    print(f"Accuracy: {accuracy:.2f}")


def predict_review_by_score_model(parent=0):
    print("Load listings reviews")
    reviews_df = get_reviews_desc(parent=1, cached=True)
    listings_df = process_full_listings(parent=1, verbose=False)
    reviews_listings_df = pd.merge(
        listings_df, reviews_df, left_on='id', right_on='listing_id', how='inner')

    # Preprocess reviews
    rev_df = NlpService.process_reviews(reviews_listings_df, parent)
    print("Transform text to an embedding vector space")
    emb_service = EmbeddingService()
    listings_reviews_df = emb_service.get_embeddings(rev_df, 'comments')

    # Split the data into training and testing sets
    print("Split the data into training and testing sets")
    embeddings = listings_reviews_df['comments_emb'].tolist()
    labels = listings_reviews_df['review_scores_value'].tolist()
    _, X_test, _, y_test = train_test_split(
        embeddings, labels, test_size=0.2, random_state=42)

    print("Make price predictions on new text reviews")
    classifier = DataService.save_model(
        'classifier', parent, apex='train_reviews_by_score')
    y_pred = classifier.predict(X_test)

    print("Evaluate the model's accuracy")
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")

    print("Use the trained model to predict house rental based on new text reviews")
    new_reviews = ["This house is amazing!", "The rental experience was terrible."]
    new_embeddings = emb_service.model.encode(new_reviews)
    new_predictions = classifier.predict(new_embeddings)
    print(new_predictions)


if __name__ == '__main__':
    predict_price_model()
    predict_review_by_score_model()
