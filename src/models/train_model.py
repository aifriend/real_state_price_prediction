from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.listing_etl_pipeline import process_full_listings
from src.review_etl_pipeline import process_full_reviews
from src.service.RentalLogger import logger
from src.service.TrainService import TrainService


def pre_process_for_training(df):
    # Drop meaningless columns
    df.drop(columns=['street', 'neighbourhood_cleansed',
                     'neighbourhood_group_cleansed', 'city',
                     ],
            inplace=True)

    # Convert objects columns to category if necessary
    # The categorical features have to be converted internally to numerical features for efficient modeling
    label_encoder = LabelEncoder()
    column_list = df.select_dtypes(exclude=['int', 'float']).columns
    for column in column_list:
        df[column] = df[column].astype("category")
        # Apply LabelEncoder to the categorical column
        encoded_categories = label_encoder.fit_transform(df[column])
        category_mapping = dict(zip(df[column], encoded_categories))
        df[column] = encoded_categories
        logger.info(f"{column} ==> "
                    f"Apply label encoder with label "
                    f"{category_mapping}")

    return df


def train_price_model(optimization=False):
    listing_df = process_full_listings(parent=1)

    train_df = pre_process_for_training(listing_df)

    # drop less significance features
    train_df.drop(
        columns=train_df.columns[
            [1, 2, 7, 11, 12, 13, 15, 17, 18, 19, 20,
             23, 27, 28, 29, 30, 31, 34, 36, 37, 38,
             39, 40, 41, 42, 43, 47, 49, 50, 51]], inplace=True)

    # X and y dataset preparation
    Xtr, Xts, ytr, yts = TrainService.TrainSampleModel(train_df)

    # X = train_df.drop(['price'], axis=1).values
    # y = train_df['price'].values
    # logger.info(X.shape)
    # logger.info(y.shape)
    #
    # # Split the data into training and testing sets
    # Xtr, Xts, ytr, yts = train_test_split(
    #     X, y, test_size=0.3, random_state=42)

    # Train
    xgb, resc_x_tr = TrainService.train(Xtr, Xts, ytr, yts)

    # Tuning
    b_model = None
    if optimization:
        b_model, b_score, b_params = TrainService.xgboost_tuning(resc_x_tr, ytr)
        logger.info("Grid search completed.")
        logger.info(f"Best hyperparameters: {b_params}")
        logger.info(f"Best score: {b_score}")

    return xgb if xgb else b_model


def train_reviews_by_score(parent=1):
    # Load listings reviews
    reviews_df = process_full_reviews(parent=1)

    # Train
    TrainService.train_reviews_by_score(reviews_df, parent)


def train_review_sentiment_analysis():
    # Load listings reviews
    reviews_df = process_full_reviews(parent=1)

    # Train
    TrainService.train_review_sentiment_analysis(reviews_df, parent=1)


if __name__ == '__main__':
    # train_price_model()
    # train_reviews_by_score()
    train_review_sentiment_analysis()
