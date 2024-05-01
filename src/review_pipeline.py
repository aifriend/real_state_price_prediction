"""
Data Loading and Preprocessing:
   - Load the data files into Python using libraries like pandas.
   - Perform data cleaning and preprocessing:
     - Handle missing values
     - Convert data types if necessary
     - Merge relevant datasets based on common columns
   - Explore the data to gain initial insights
"""
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import BertTokenizer, BertModel

from src.listing_pipeline import process_full_listings
from src.service.DataService import DataService
from src.service.NlpService import NlpService


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


def get_reviews_desc(parent=0, cached=False):
    if not cached:
        _, reviews_desc_df = _preprocess_merged_reviews()
        reviews_desc_df = NlpService.process_reviews(
            reviews_desc_df, parent, save=True)
    else:
        print("Load processed reviews")
        reviews_desc_df = pd.read_csv(
            Path.cwd().parents[parent].joinpath(
                "data/processed", "neighborhood_reviews.csv.gz"))

    return reviews_desc_df


def _get_merged_reviews():
    return load_reviews(), load_reviews_desc()


def _preprocess_merged_reviews():
    # Load reviews
    rev_df, rev_desc_df = _get_merged_reviews()

    for column in rev_desc_df.columns:
        print(f"\n{column.upper()}\n{'=' * 20}\n{rev_desc_df[column].describe()}")
        print(f"Number of null values: {rev_desc_df[column].isna().sum()}\n")

    return rev_df, rev_desc_df


def process_reviews_by_clustering():
    # Load pre-trained BERT model and tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    def get_bert_embedding(text, model, tokenizer):
        """Generate BERT embedding for given text"""
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        outputs = model(**inputs)
        last_hidden_states = outputs.last_hidden_state
        return last_hidden_states[:, 0, :].detach().numpy()

    def cluster_listings(reviews_df, num_clusters=5):
        """Cluster rental listings based on review embeddings"""
        print("Load listings reviews")
        listings_df = process_full_listings(verbose=False)
        reviews_listings_df = pd.merge(
            listings_df, reviews_df, left_on='id', right_on='listing_id', how='inner')

        # all listings
        # reviews_listings_df = reviews_listings_df.loc[:1000, :]
        grouped_df = reviews_listings_df.groupby("neighbourhood")['comments'].apply(list).reset_index()
        grouped_df = grouped_df[grouped_df['comments'].apply(lambda x: len(x) > 0)]

        # Generate BERT embeddings for each listing's reviews
        listing_embeddings = []
        listing_keys = []
        listing_reviews = dict()
        for idx, (index, row) in enumerate(grouped_df.iterrows()):
            listing_neighbourhood = row['neighbourhood']
            listing_keys.append(listing_neighbourhood)
            reviews = row['comments'][:30]  # Limit the number of reviews
            listing_reviews[listing_neighbourhood] = reviews
            print(f"Embedding for [{idx + 1}/{len(grouped_df)}] "
                  f"neighbourhood: {listing_neighbourhood} - "
                  f"comments: {len(reviews)}")
            review_embeddings = [get_bert_embedding(review, model, tokenizer) for review in reviews]
            if review_embeddings:
                listing_embedding = np.mean(review_embeddings, axis=0)
                listing_embeddings.append(listing_embedding[0])

        # Perform K-means clustering on listing embeddings
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        kmeans.fit(listing_embeddings)
        cluster_labels = kmeans.labels_

        # Print cluster details
        cluster_reviews = {i: [] for i in range(num_clusters)}
        for listing_id, label in zip(listing_keys, cluster_labels):
            cluster_reviews[label].extend(listing_reviews[listing_id])

        for i, reviews in cluster_reviews.items():
            print(f"Cluster {i}:")
            print("Top terms:", get_top_terms(reviews))
            print("Example reviews:", reviews[:1])
            print()

        return cluster_labels

    def get_top_terms(reviews, n=7):
        """Get top N terms based on TF-IDF scores"""
        vectorizer = TfidfVectorizer()
        tfidf = vectorizer.fit_transform(reviews)
        feature_names = vectorizer.get_feature_names_out()
        top_terms = Counter()
        for i in range(tfidf.shape[0]):
            top_indices = tfidf[i].toarray().argsort()[0][-n:]
            top_terms.update(feature_names[top_indices])

        top_terms = top_terms.most_common(n)
        return top_terms

    reviews_df = get_reviews_desc(cached=True)

    """
    Cluster 0:
    Top terms: [('definitely', 1), ('cool', 1), ('bathroom', 1), ('hot', 1), ('way', 1), ('great', 1), ('family', 1)]
    Example reviews: ['great family nice go way accommodate room spacious bathroom great , hot shower , patio super cool definitely recommend stay family']
    
    Cluster 1:
    Top terms: [('good', 41), ('flat', 34), ('great', 29), ('place', 29), ('check', 29), ('nice', 24), ('jose', 24)]
    Example reviews: ['tsvetana friendly hospitable , speak common language , able need . room apartment nice clean . beautiful park near apartment worth visit .']
    
    Cluster 2:
    Top terms: [('enjoy', 2), ('single', 1), ('minute', 1), ('elizabeth', 1), ('solo', 1), ('family', 1), ('10', 1)]
    Example reviews: ['elizabeth place clean nice . family ( especially granddaughter ) friendly helpful . 10 minute walk metro , single solo traveler feel safe nice neighborhood . thank great stay']
    """
    cluster_listings(reviews_df, num_clusters=3)

    """
    Cluster 0:
    Top terms: [('definitely', 1), ('cool', 1), ('bathroom', 1), ('hot', 1), ('way', 1), ('great', 1), ('family', 1)]
    Example reviews: ['great family nice go way accommodate room spacious bathroom great , hot shower , patio super cool definitely recommend stay family']
    
    Cluster 1:
    Top terms: [('good', 11), ('travel', 10), ('nice', 9), ('expect', 8), ('great', 8), ('taxi', 7), ('tip', 7)]
    Example reviews: ['marga house exactly picture lovely host . difficulty communicate manage . kind accommodate friend separate room despite pay . highly recommend']
    
    Cluster 2:
    Top terms: [('enjoy', 2), ('especially', 1), ('family', 1), ('helpful', 1), ('granddaughter', 1), ('single', 1), ('10', 1)]
    Example reviews: ['elizabeth place clean nice . family ( especially granddaughter ) friendly helpful . 10 minute walk metro , single solo traveler feel safe nice neighborhood . thank great stay']
    
    Cluster 3:
    Top terms: [('flat', 31), ('good', 29), ('check', 24), ('great', 23), ('place', 22), ('nice', 19), ('maria', 19)]
    Example reviews: ['tsvetana friendly hospitable , speak common language , able need . room apartment nice clean . beautiful park near apartment worth visit .']
    
    Cluster 4:
    Top terms: [('exactly', 1), ('evening', 1), ('couple', 1), ('consider', 1), ('boy', 1), ('late', 1), ('subway', 1)]
    Example reviews: ['need place lay couple night exactly . quiet neighborhood 50 m local subway stop . marcos boy kind consider friend , late evening .']
    """
    cluster_listings(reviews_df, num_clusters=5)

    """
    """
    cluster_listings(reviews_df, num_clusters=10)
