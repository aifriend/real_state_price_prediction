"""Here's a step-by-step approach to solving the task using Python:

1. Data Loading and Preprocessing:
   - Load the data files into Python using libraries like pandas.
   - Perform data cleaning and preprocessing:
     - Handle missing values
     - Convert data types if necessary
     - Merge relevant datasets based on common columns
   - Explore the data to gain initial insights
"""
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from src.service.DataService import DataService
from src.service.EmbeddingService import EmbeddingService


def load_reviews():
    df1 = pd.read_csv(
        Path.cwd().parents[0].joinpath("data/raw", "reviews.csv"))
    print(df1.head())

    # Convert object type columns to string
    df1 = DataService.convert_object_to_string(df1)
    # convert the 'last_review' column to a date data type handling missing values
    df1['date'] = pd.to_datetime(df1['date'], errors='coerce')

    # Check missing values
    print(df1.isna().sum())

    return df1


def load_reviews_desc():
    df2 = pd.read_csv(
        Path.cwd().parents[0].joinpath("data/raw", "reviews.csv.gz"))
    print(df2.head())

    # Convert object type columns to string
    df2 = DataService.convert_object_to_string(df2)
    # convert the 'last_review' column to a date data type handling missing values
    df2['date'] = pd.to_datetime(df2['date'], errors='coerce')

    # Impute missing values
    print(df2.isna().sum())
    # fill reviewer_name with 'None'
    df2['reviewer_name'] = df2['reviewer_name'].fillna('None')
    # fill comments with empty string
    df2['comments'] = df2['comments'].fillna('')
    print(df2.isna().sum())

    return df2


def get_merged_reviews():
    return load_reviews(), load_reviews_desc()


def preprocess_merged_reviews():
    # Load reviews
    rev_df, rev_desc_df = get_merged_reviews()

    for column in rev_desc_df.columns:
        print(f"\n{column.upper()}\n{'=' * 20}\n{rev_desc_df[column].describe()}")
        print(f"Number of null values: {rev_desc_df[column].isna().sum()}\n")

    return rev_df, rev_desc_df


if __name__ == '__main__':
    """
    The following code is used for text clustering and documents embedding. We want to represent reviews 
    as vectors representation to be able to apply clustering algorithms to detect topics. Reviews are 
    usually short sentences, thus, we should look for a suitable embedding approach for this situation
    like Sentence Transformers with BERT or Glove.
    
    Use the sentence embedding model BERT to convert text reviews to a vector space, and 
    train a logistic regression model to predict price based on these embeddings, and use the trained model 
    to make price predictions on new text reviews.
    -------------------------------------------------------------------------------------------------
    We analyzed the relationship between the description of each listing and its price, 
    and proposed a text-based price recommendation system called TAPE to recommend a reasonable price 
    for newly added listings
    """
    print("Load reviews")
    # _, reviews_desc_df = preprocess_merged_reviews()
    # reviews_df = NlpService.process_reviews(reviews_desc_df)
    reviews_df = pd.read_csv(
        Path.cwd().parents[0].joinpath(
            "data/processed", "neighborhood_reviews.csv.gz"))

    print("Transform text to an embedding vector space")
    emb_service = EmbeddingService()
    reviews_df = emb_service.get_embeddings(reviews_df, 'comments')

    print("Split the data into training and testing sets")
    embeddings = reviews_df['comments'].tolist()
    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, reviews_df['price'], test_size=0.2, random_state=42)

    print("Train a logistic regression model using embeddings")
    classifier = LogisticRegression()
    classifier.fit(X_train, y_train)

    print("Make price predictions on new text reviews")
    y_pred = classifier.predict(X_test)

    print("Evaluate the model's accuracy")
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")

    print("Use the trained model to predict house rental based on new text reviews")
    new_reviews = ["This house is amazing!", "The rental experience was terrible."]
    new_embeddings = emb_service.model.encode(new_reviews)
    new_predictions = classifier.predict(new_embeddings)
    print(new_predictions)

