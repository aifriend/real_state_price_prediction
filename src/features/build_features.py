from collections import Counter

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import BertTokenizer, BertModel

from src.listing_pipeline import process_full_listings
from src.review_pipeline import get_reviews_desc
from src.service.DataService import DataService


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
    Cluster 0:
    Top terms: [('definitely', 1), ('cool', 1), ('bathroom', 1), ('hot', 1), ('way', 1), ('great', 1), ('family', 1)]
    Example reviews: ['great family nice go way accommodate room spacious bathroom great , hot shower , patio super cool definitely recommend stay family']
    
    Cluster 1:
    Top terms: [('place', 4), ('room', 2), ('stay', 2), ('miguel', 2), ('good', 2), ('muy', 2), ('sure', 1)]
    Example reviews: ['esther friendly , helpful , super person , house clean quiet like star hotel , area like restaurant , supermarket . \n highly recommend room traveler , stay room sure .']
    
    Cluster 2:
    Top terms: [('family', 1), ('especially', 1), ('elizabeth', 1), ('clean', 1), ('helpful', 1), ('walk', 1), ('nice', 1)]
    Example reviews: ['elizabeth place clean nice . family ( especially granddaughter ) friendly helpful . 10 minute walk metro , single solo traveler feel safe nice neighborhood . thank great stay']
    
    Cluster 3:
    Top terms: [('good', 34), ('flat', 32), ('great', 25), ('check', 25), ('nice', 23), ('place', 21), ('eat', 19)]
    Example reviews: ['marga house exactly picture lovely host . difficulty communicate manage . kind accommodate friend separate room despite pay . highly recommend']
    
    Cluster 4:
    Top terms: [('communication', 1), ('arrive', 1), ('apartment', 1), ('little', 1), ('host', 1), ('lovely', 1), ('jose', 1)]
    Example reviews: ['jose kind considerate host great communication . help arrive airport flight little late regular transfer . apartment lovely clean , good facility . shower lovely . thank jose .']
    
    Cluster 5:
    Top terms: [('communication', 1), ('city', 1), ('bit', 1), ('bike', 1), ('trouble', 1), ('walk', 1), ('enjoy', 1)]
    Example reviews: ['bit trouble communication host speak english room nice far metro . park nearby enjoy bike ride feel like take walk enjoy atmosphere outside city .']
    
    Cluster 6:
    Top terms: [('issue', 1), ('find', 1), ('cause', 1), ('barrier', 1), ('language', 1), ('way', 1), ('great', 1)]
    Example reviews: ['place great amazing key slightly stiff cause major issue number occasion stay perfect substantial language barrier jesus find way great']
    
    Cluster 7:
    Top terms: [('apartment', 9), ('good', 8), ('place', 8), ('close', 6), ('maria', 6), ('house', 6), ('minute', 6)]
    Example reviews: ['tsvetana friendly hospitable , speak common language , able need . room apartment nice clean . beautiful park near apartment worth visit .']
    
    Cluster 8:
    Top terms: [('bruna', 1), ('bit', 1), ('away', 1), ('20', 1), ('19euro', 1), ('helpful', 1), ('want', 1)]
    Example reviews: ['bruna nice helpful short stay madrid . location bit far tourist want uber ride 20 minute & ( 17 - 19euro ) away el centro / plaza mayor . , great .']
    
    Cluster 9:
    Top terms: [('exactly', 1), ('evening', 1), ('couple', 1), ('consider', 1), ('boy', 1), ('late', 1), ('subway', 1)]
    Example reviews: ['need place lay couple night exactly . quiet neighborhood 50 m local subway stop . marcos boy kind consider friend , late evening .']
    """
    cluster_listings(reviews_df, num_clusters=10)


def main_features():
    forest = DataService.load_model(
        'forest', parent=1, apex='feature_selection')

    # Feature importance
    importance = forest.feature_importances_
    for i, v in enumerate(importance):
        print('feature: %d, score: %.5f' % (i, v))
    sns.set()
    plt.bar([x for x in range(len(importance))], importance)
    plt.title("A barplot showing the significance of each feature")
    plt.show()
