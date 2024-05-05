import math
from collections import Counter
from pathlib import Path

import folium
import geopandas as gpd
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from pandas import DataFrame
from scipy import stats
from scipy.stats import f_oneway
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer, BertModel

from src.review_etl_pipeline import process_full_reviews
from src.service.EmbeddingService import EmbeddingService
from src.service.RentalLogger import logger


class EDA:
    """
    Exploratory data analysis: Visualized property locations on an interactive map,
    generated a word cloud to extract insights from property agent descriptions,
    and examined descriptive statistics, distributions, and correlations.

    The following methods are available:
    - cluster_review_features
    - feature_correlation_analysis
    - neighborhood_analysis
    - review_analysis
    - feature_exploratory_analysis
    """

    @staticmethod
    def cluster_review_features() -> None:
        """
        Cluster review features based on TF-IDF scores

        Returns:
        """
        # Load pre-trained BERT model and tokenizer
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased')

        def get_bert_embedding(text, b_model, b_tokenizer):
            """
            Generate BERT embedding for given text

            Args:
                text (str): text to be embedded
                b_model (BertModel): pre-trained BERT model
                b_tokenizer (BertTokenizer): pre-trained BERT tokenizer
            """
            inputs = b_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            outputs = b_model(**inputs)
            last_hidden_states = outputs.last_hidden_state
            return last_hidden_states[:, 0, :].detach().numpy()

        def contains_nan(my_list):
            """
            Check if a list contains NaN values

            Args:
                my_list (list): list to be checked
            """
            for item in my_list:
                if isinstance(item, float) and math.isnan(item):
                    return True
            return False

        def cluster_listings(num_clusters=5):
            """
            Cluster rental listings based on review embeddings

            Args:
                num_clusters (int): number of clusters to be generated
            """
            logger.info("Load listings reviews")
            reviews_listings_df = process_full_reviews(parent=1)

            # all listings
            # reviews_listings_df = reviews_listings_df.loc[:1000, :]
            grouped_df = reviews_listings_df.groupby("neighbourhood")['comments'].apply(list).reset_index()
            grouped_df = grouped_df[grouped_df['comments'].apply(lambda x: len(x) > 0)]

            # Generate BERT embeddings for each listing's reviews
            listing_embeddings = []
            listing_keys = []
            listing_reviews = dict()
            for idx, (index, row) in enumerate(grouped_df.iterrows()):
                reviews = row['comments'][:30]  # Limit the number of reviews
                if contains_nan(reviews):
                    continue
                listing_neighbourhood = row['neighbourhood']
                listing_keys.append(listing_neighbourhood)
                listing_reviews[listing_neighbourhood] = reviews
                logger.info(f"Embedding for [{idx + 1}/{len(grouped_df)}] "
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
                logger.info(f"Cluster {i}:")
                logger.info(f"Top terms: {get_top_terms(reviews)}")
                logger.info(f"Example reviews: {reviews[:1]}")

            return cluster_labels

        def get_top_terms(reviews, n=7):
            """
            Get top N terms based on TF-IDF scores

            Args:
                reviews (list): list of reviews
                n (int): number of top terms to be returned
            """
            vectorizer = TfidfVectorizer()
            tfidf = vectorizer.fit_transform(reviews)
            feature_names = vectorizer.get_feature_names_out()
            top_terms = Counter()
            for i in range(tfidf.shape[0]):
                top_indices = tfidf[i].toarray().argsort()[0][-n:]
                top_terms.update(feature_names[top_indices])

            top_terms = top_terms.most_common(n)
            return top_terms

        cluster_listings(num_clusters=5)

    @staticmethod
    def feature_correlation_analysis(df: DataFrame) -> None:
        """
        visualizing the correlation of the features

        Shows relations between these features:
        -bathrooms
        -bedrooms
        -beds
        -guests_included
        -maximum_nights
        -minimum_minimum_nights
        -maximum_minimum_nights

        Also, relation between host_listing_count and availabilities

        Args:
            df (pd.DataFrame): dataframe containing the features
        """
        # Select numerical columns
        numerical_columns = df.select_dtypes(include=['int', 'float'])

        # Select categorical columns
        categorical_columns = df.select_dtypes(include=['category'])
        # Apply LabelEncoder to the categorical column
        label_encoder = LabelEncoder()
        for column in categorical_columns.columns:
            categorical_columns[column] = (
                label_encoder.fit_transform(categorical_columns[column]))

        c = pd.concat([numerical_columns], axis=1)
        plt.figure(figsize=(30, 12))
        plt.title("A heatmap showing correlation between the features")
        sns.heatmap(c.corr(), annot=True)
        plt.savefig(
            Path.cwd().parents[1].joinpath(
                "reports/figures", "heatmap.png"))
        plt.show()

    @staticmethod
    def neighborhood_analysis(df: DataFrame) -> None:
        """
        Exploratory analysis of the neighborhood

        Args:
            df (pd.DataFrame): dataframe containing the features
        """

        # Analyze the relationship between price and neighborhood
        plt.figure(figsize=(12, 8))
        plt.title('Price Distribution by Neighborhood')
        plt.xlabel('Neighborhood')
        plt.ylabel('Price')
        plt.xticks(rotation=45)
        sns.boxplot(data=df, x='neighbourhood_group', y='price')
        plt.savefig(
            Path.cwd().parents[1].joinpath(
                "reports/figures", "neighborhood_by_price_box_plot.png"))
        plt.show()

        # It provides a more detailed view of the price ranges and the density of listings at different price points
        plt.figure(figsize=(12, 8))
        plt.title('Price Distribution by Neighborhood')
        plt.xlabel('Neighborhood')
        plt.ylabel('Price')
        sns.violinplot(data=df, x='neighbourhood_group', y='price')
        plt.xticks(rotation=45)
        plt.savefig(
            Path.cwd().parents[1].joinpath(
                "reports/figures", "neighborhood_by_price_violin.png"))
        plt.show()

        # This will provide an overview of the average, median, minimum, and maximum prices for each neighborhood
        neighborhood_stats = df.groupby('neighbourhood_group')['price'].agg(['mean', 'median', 'min', 'max'])
        logger.info(neighborhood_stats)

        # Statistical tests to compare prices across neighborhoods
        neighborhoods = df['neighbourhood_group'].unique()
        price_data = [df[df['neighbourhood_group'] == neighborhood]['price'] for neighborhood in neighborhoods]
        # test to determine if there are significant differences in prices across neighborhoods.
        # A low p-value (typically < 0.05) indicates that there are statistically significant differences
        # in prices between at least two neighborhoods
        f_statistic, p_value = f_oneway(*price_data)
        logger.info(f"One-way ANOVA: F-statistic = {f_statistic:.2f}, p-value = {p_value:.4f}")

        # Create a heatmap to visualize the correlation between price and other relevant features
        # This heatmap visualizes the correlation between price and other relevant features such as minimum nights,
        # number of reviews, reviews per month, host listings count, and availability. It helps identify any potential
        # relationships or patterns between these features and the price like: number_of_reviews (0.69)
        plt.figure(figsize=(10, 8))
        plt.title('Correlation Heatmap')
        sns.heatmap(df[['price', 'minimum_nights', 'number_of_reviews',
                        'reviews_per_month', 'availability_365']].corr(), annot=True, cmap='coolwarm')
        plt.savefig(
            Path.cwd().parents[1].joinpath(
                "reports/figures", "heatmap_reduced.png"))
        plt.show()

        # Analyze the top neighborhoods by average price
        # This code identifies the top neighborhoods based on the average price,
        # providing insights into the most expensive areas in the city.
        top_neighborhoods = neighborhood_stats.sort_values(by='mean', ascending=False).head(10)
        logger.info("Top Neighborhoods by Average Price:")
        logger.info(top_neighborhoods)

        """
        Visualize the distribution of listings by neighborhood

        - The distribution of listings across neighborhoods is uneven, with some neighborhoods having 
        a higher concentration of listings compared to others.
        - This could be influenced by factors such as tourist attractions, accessibility, or local regulations.
        """
        plt.figure(figsize=(12, 8))
        plt.title('Distribution of Listings by Neighborhood')
        plt.xlabel('Neighborhood')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        sns.countplot(data=df, x='neighbourhood_cleansed', order=df['neighbourhood_cleansed'].value_counts().index)
        plt.savefig(
            Path.cwd().parents[1].joinpath(
                "reports/figures", "listings_by_neighborhood.png"))
        plt.show()

        """
        Read the property data into a GeoDataFrame        
        """
        property_gdf = gpd.read_file(
            Path.cwd().parents[1].joinpath(
                "data/raw", "neighbourhoods.geojson"))
        # Get the center coordinates of the property locations
        center_lat = property_gdf.geometry.centroid.y.mean()
        center_lon = property_gdf.geometry.centroid.x.mean()
        # Create a Folium map centered on the mean coordinates
        map_center = [center_lat, center_lon]
        map_zoom = 12
        folium_map = folium.Map(location=map_center, zoom_start=map_zoom)
        # Add property locations as markers to the map
        for idx, row in property_gdf.iterrows():
            location = [row.geometry.centroid.y, row.geometry.centroid.x]
            popup_text = (f"Property Neighbourhood Location: {row['neighbourhood']}<br>"
                          f"Property Neighbourhood Area: {row['neighbourhood_group']}")
            folium.Marker(location=location, popup=popup_text).add_to(folium_map)
        # Save the map to an HTML file
        folium_map.save(
            Path.cwd().parents[1].joinpath(
                "reports/figures", "property_map.html"))

    @staticmethod
    def review_analysis(listings_df: DataFrame) -> None:
        """
        Analyze the relationship between price and number of reviews

        - There is a weak positive correlation between the price of a listing and the number
        of reviews it has received.
        - This suggests that higher-priced listings tend to receive more reviews, possibly due to
        their popularity or quality.

        - The correlation coefficient ranges from -1 to 1, where values close to 1 indicate a strong positive
        correlation, values close to -1 indicate a strong negative correlation, and values close to 0 indicate
        a weak or no correlation.

        Args:
            listings_df (DataFrame): The DataFrame containing the listings data.
        """
        plt.figure(figsize=(10, 6))
        plt.title('Price vs. Number of Reviews (scatter)')
        plt.xlabel('Number of Reviews')
        plt.ylabel('Price')
        sns.scatterplot(data=listings_df, x='number_of_reviews', y='price')
        plt.savefig(
            Path.cwd().parents[1].joinpath(
                "reports/figures", "reviews_by_price.png"))
        plt.show()

        # Correlation analysis
        # Correlation coefficient between price and number of reviews. The correlation coefficient ranges from -1 to 1,
        # where values close to 1 indicate a strong positive correlation, values close to -1 indicate a strong negative
        # correlation, and values close to 0 indicate a weak or no correlation.
        correlation = listings_df['price'].corr(listings_df['number_of_reviews'])
        logger.info(f"Correlation between price and number of reviews: {correlation:.2f}")

        # Price distribution by review categories
        # Categorize the number of reviews into different bins (e.g., Low, Medium, High, Very High) and
        # create a box plot to visualize the price distribution for each review category. This will help
        # identify if there are any significant differences in price based on the number of reviews.
        listings_df['review_category'] = pd.cut(
            listings_df['number_of_reviews'], bins=[0, 10, 50, 100, float('inf')],
            labels=['Low', 'Medium', 'High', 'Very High'])

        plt.figure(figsize=(10, 6))
        plt.title('Price Distribution by Review Categories')
        plt.xlabel('Review Category')
        plt.ylabel('Price')
        sns.boxplot(data=listings_df, x='review_category', y='price')
        plt.savefig(
            Path.cwd().parents[1].joinpath(
                "reports/figures", "review_category_by_price.png"))
        plt.show()

        # Statistical tests
        # Perform ANOVA test to determine if there are significant differences in price among the different review
        # categories. The ANOVA test compares the means of the price across the review categories. It indicates
        # that there are significant differences in price among the review categories.
        f_statistic, p_value = stats.f_oneway(
            listings_df[listings_df['review_category'] == 'Low']['price'],
            listings_df[listings_df['review_category'] == 'Medium']['price'],
            listings_df[listings_df['review_category'] == 'High']['price'],
            listings_df[listings_df['review_category'] == 'Very High']['price']
        )
        logger.info(f"ANOVA test results:")
        logger.info(f"F-statistic: {f_statistic:.2f}")
        logger.info(f"p-value: {p_value:.4f}")

        """
        Relationship between room type and number of reviews

        This relationship help us understand if certain room types (e.g., entire home/apt, private room, shared room) 
        tend to receive more reviews than others. It can provide insights into the popularity and guest satisfaction 
        associated with different room types.
        """
        plt.figure(figsize=(10, 6))
        plt.title('Number of Reviews by Room Type')
        plt.xlabel('Room Type')
        plt.ylabel('Number of Reviews')
        sns.boxplot(data=listings_df, x='room_type', y='number_of_reviews')
        plt.savefig(
            Path.cwd().parents[1].joinpath(
                "reports/figures", "reviews_by_room_type.png"))
        plt.show()

        """
        Relationship between host response rate and review scores

        This analysis explores the potential correlation between a host's responsiveness 
        (measured by the host response rate) and the overall review scores received by their listings. 
        It can help identify if hosts who are more responsive tend to receive higher review scores, 
        indicating better guest satisfaction
        """
        plt.figure(figsize=(10, 6))
        plt.title('Review Scores vs. Host Response Rate')
        plt.xlabel('Host Response Rate')
        plt.ylabel('Review Scores Rating')
        sns.scatterplot(data=listings_df, x='host_response_rate', y='review_scores_rating')
        plt.savefig(
            Path.cwd().parents[1].joinpath(
                "reports/figures", "review_scores_vs_host_response_rate.png"))
        plt.show()

        """
        Relationship between review length and number of reviews
        """
        listings_df['review_length'] = listings_df['comments'].apply(len)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_title('Distribution of Review Lengths')
        ax.set_xlabel('Review Length')
        ax.set_ylabel('Count')
        sns.histplot(data=listings_df, x='review_length', ax=ax, kde=True)
        plt.savefig(
            Path.cwd().parents[1].joinpath(
                "reports/figures", "reviews_length.png"))
        plt.show()

        """
        Reviews per day
        """
        listings_df['date'] = pd.to_datetime(listings_df['date'])
        daily_reviews = listings_df.groupby(
            listings_df['date'].dt.date).size().reset_index(name='count')
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.set_title('Daily Number of Reviews')
        ax.set_xlabel('Date')
        ax.set_ylabel('Number of Reviews')
        plt.xticks(rotation=45)
        sns.lineplot(data=daily_reviews, x='date', y='count', ax=ax)
        plt.savefig(
            Path.cwd().parents[1].joinpath(
                "reports/figures", "reviews_per_day.png"))
        plt.show()

        """
        Gaussian Mixture Model (GMM) algorithm. GMM is a probabilistic model that assumes
        the data is generated from a mixture of a finite number of Gaussian distributions.
        It can automatically determine the optimal number of clusters based on the data.
        """
        logger.info("Transform text to an embedding vector space")
        emb_service = EmbeddingService()
        emb_reviews_df = emb_service.get_embeddings(listings_df, 'comments')

        logger.info("Split the data into training and testing sets")
        embeddings = np.array(emb_reviews_df['comments_emb'].tolist())

        # Flatten the embeddings
        flattened_embeddings = embeddings.reshape(embeddings.shape[0], -1)

        # Perform dimensionality reduction using t-SNE
        tsne = TSNE(n_components=2, perplexity=30, random_state=42)
        reduced_embeddings = tsne.fit_transform(flattened_embeddings)

        # Perform clustering using Gaussian Mixture Model
        n_components_range = range(2, 11)  # Range of number of clusters to try
        best_gmm = None
        best_bic = np.inf
        best_n_components = 0

        for n_components in n_components_range:
            gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)
            gmm.fit(reduced_embeddings)
            bic = gmm.bic(reduced_embeddings)
            if bic < best_bic:
                best_bic = bic
                best_gmm = gmm
                best_n_components = n_components

        # Assign cluster labels to each data point
        cluster_labels = best_gmm.predict(reduced_embeddings)

        # Visualize the clustering results in a 2D figure
        plt.figure(figsize=(8, 6))
        plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=cluster_labels, cmap='viridis')
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.title(f'Clustering Results (Number of Clusters: {best_n_components})')
        plt.colorbar(label='Cluster')
        plt.savefig(
            Path.cwd().parents[1].joinpath(
                "reports/figures", "reviews_cluster.png"))
        plt.show()

    @staticmethod
    def feature_exploratory_analysis(df: DataFrame) -> None:
        """
        Visualize the distribution of rental prices

        Args:
            df (DataFrame): DataFrame containing rental data
        """

        """
        Analyze the distribution of rental prices

        - The distribution of rental prices shows a right-skewed pattern, with a majority of listings having 
        lower prices and a few high-priced outliers.
        - The median price varies across different neighborhoods, indicating that location plays a significant role 
        in determining rental prices.
        """
        plt.figure(figsize=(10, 6))
        plt.title('Distribution of Rental Prices')
        plt.xlabel('Price')
        plt.ylabel('Count')
        sns.histplot(data=df, x='price', kde=True)
        plt.savefig(
            Path.cwd().parents[1].joinpath(
                "reports/figures", "rental_prices_distribution.png"))
        plt.show()

        """
        Visualize the distribution of room types

        - The most common room types in the temporary rental market are entire homes/apartments, followed by 
        private rooms and shared rooms.
        - This suggests that a significant portion of the market caters to travelers seeking private accommodations.
        """
        plt.figure(figsize=(8, 6))
        plt.title('Distribution of Room Types')
        plt.xlabel('Room Type')
        plt.ylabel('Count')
        sns.countplot(data=df, x='room_type')
        plt.savefig(
            Path.cwd().parents[1].joinpath(
                "reports/figures", "room_types_distribution.png"))
        plt.show()

        """
        Visualize the availability of listings throughout the year

        - The availability of listings throughout the year varies, with some listings being available 
        for a significant portion of the year, while others have limited availability.
        - This could be influenced by factors such as seasonality, host preferences, or local regulations.
        """
        plt.figure(figsize=(10, 6))
        plt.title('Distribution of Listing Availability')
        plt.xlabel('Number of Available Days')
        plt.ylabel('Count')
        sns.histplot(data=df, x='availability_365', kde=True)
        plt.savefig(
            Path.cwd().parents[1].joinpath(
                "reports/figures", "listings_availability.png"))
        plt.show()

        """
        Analyze the relationship between price and property type

        - The box plot will show a box for each property type, representing the distribution of prices 
        for that property type.
        - The box extends from the first quartile (25th percentile) to the third quartile (75th percentile) 
        of the price distribution, with a line inside the box representing the median price.
        - The whiskers extend from the box to show the range of prices, excluding outliers.
        - Outliers, if any, will be represented as individual points beyond the whiskers.
        - By comparing the boxes and their positions, we can identify which property types have higher or lower 
        median prices and how the price ranges differ across property types.
        - Property types with higher median prices will be positioned towards the top of the plot, while those 
        with lower median prices will be positioned towards the bottom.
        """
        plt.figure(figsize=(12, 8))
        plt.title('Price Distribution by Property Type')
        plt.xlabel('Property Type')
        plt.ylabel('Price')
        plt.xticks(rotation=45)
        sns.boxplot(data=df, x='property_type', y='price',
                    order=df.groupby('property_type')['price'].median().sort_values(ascending=False).index)
        plt.savefig(
            Path.cwd().parents[1].joinpath(
                "reports/figures", "property_type_by_price.png"))
        plt.show()

        """
        Analyze the relationship between price and host response time

        Observations:
        - Listings with faster host response times tend to have higher median prices. The median price is highest for 
        listings where hosts respond within an hour, followed by those who respond within a few hours, within a day, 
        and a few days or more.
        - The price range (inter-quartile range) is also wider for listings with faster host response times. 
        This suggests that there is more variability in prices for listings with responsive hosts.
        - There are outliers present in all response time categories, indicating that there are some listings with 
        exceptionally high prices regardless of the host's response time.

        Interpretations:
        - Hosts who respond quickly to inquiries and bookings may be more attentive and provide better service, 
        which could justify higher prices for their listings.
        - Responsive hosts may be more experienced and professional in managing their listings, leading to higher 
        quality accommodations and thus higher prices.
        - Listings with faster host response times may be in higher demand, as guests prefer hosts who are responsive 
        and communicative. This increased demand could drive up the prices for these listings.
        """
        plt.figure(figsize=(10, 6))
        plt.title('Price Distribution by Host Response Time')
        plt.xlabel('Host Response Time')
        plt.ylabel('Price')
        sns.boxplot(data=df, x='host_response_time', y='price',
                    order=[3, 4, 1, 0])
        plt.savefig(
            Path.cwd().parents[1].joinpath(
                "reports/figures", "host_response_time_by_price.png"))
        plt.show()

        """
        Distribution of listings by property type

        This visualization shows the distribution of listings across different property types 
        (e.g., apartment, house, condominium). It helps understand the composition of the rental market 
        in terms of property types and identifies the most common types of properties available for rent.
        """
        plt.figure(figsize=(12, 8))
        plt.title('Distribution of Listings by Property Type')
        plt.xlabel('Property Type')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        sns.countplot(data=df, x='property_type', order=df['property_type'].value_counts().index)
        plt.savefig(
            Path.cwd().parents[1].joinpath(
                "reports/figures", "property_type_distribution.png"))
        plt.show()

        """
        Cluster rental listings based on review embeddings
        """
        EDA.cluster_review_features()


if __name__ == '__main__':
    """
    I explored the data to check if there are trends between
    the explanatory variables and the target variable.
    """
    dea_df = process_full_reviews(
        data_path='../data/interim', store_path='../data/processed')

    # EDA.feature_correlation_analysis(dea_df)
    # EDA.feature_exploratory_analysis(dea_df)
    # EDA.neighborhood_analysis(dea_df)
    EDA.review_analysis(dea_df)
