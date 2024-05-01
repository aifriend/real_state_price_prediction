from pathlib import Path

import folium
import geopandas as gpd
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy import stats
from scipy.stats import f_oneway
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import LabelEncoder

from src.listing_pipeline import process_full_listings
from src.review_pipeline import get_reviews_desc
from src.service.EmbeddingService import EmbeddingService


class EDA:
    """
    Exploratory data analysis: Visualized property locations on an interactive map,
    generated a word cloud to extract insights from property agent descriptions,
    and examined descriptive statistics, distributions, and correlations.
    """

    @staticmethod
    def feature_correlation_analysis(df):
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

        Also relation between host_listing_count and availabilities
        """
        corr_df = pd.DataFrame(df)
        corr_df.drop(columns=[
            'id', 'host_id', 'name', 'host_name', 'last_review',
            'listing_url', 'neighborhood_overview', 'notes', 'transit',
            'scrape_id', 'last_scraped', 'summary', 'description',
            'access', 'interaction', 'house_rules', 'picture_url',
            'host_url', 'host_since', 'host_about', 'host_thumbnail_url',
            'host_picture_url', 'host_verifications', 'amenities',
            'calendar_updated', 'calendar_last_scraped', 'first_review'
        ], inplace=True)
        # Select numerical columns
        numerical_columns = corr_df.select_dtypes(include=['int', 'float'])
        numerical_columns.drop(columns=[
            'latitude', 'longitude'
        ], inplace=True)
        # Select categorical columns
        categorical_columns = corr_df.select_dtypes(include=['category'])
        categorical_columns.drop(columns=[
            'host_neighbourhood'
        ], inplace=True)
        # Initialize LabelEncoder
        label_encoder = LabelEncoder()
        # Apply LabelEncoder to the categorical column
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
    def neighborhood_analysis(df):
        """
        Analyze the relationship between price and neighborhood
        """
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
        print(neighborhood_stats)

        # Statistical tests to compare prices across neighborhoods
        neighborhoods = df['neighbourhood_group'].unique()
        price_data = [df[df['neighbourhood_group'] == neighborhood]['price'] for neighborhood in neighborhoods]
        # test to determine if there are significant differences in prices across neighborhoods.
        # A low p-value (typically < 0.05) indicates that there are statistically significant differences
        # in prices between at least two neighborhoods
        f_statistic, p_value = f_oneway(*price_data)
        print(f"One-way ANOVA: F-statistic = {f_statistic:.2f}, p-value = {p_value:.4f}")

        # Create a heatmap to visualize the correlation between price and other relevant features
        # This heatmap visualizes the correlation between price and other relevant features such as minimum nights,
        # number of reviews, reviews per month, host listings count, and availability. It helps identify any potential
        # relationships or patterns between these features and the price like: number_of_reviews (0.69) and
        # calculated_host_listings_count (0.17)
        plt.figure(figsize=(10, 8))
        plt.title('Correlation Heatmap')
        sns.heatmap(df[['price', 'minimum_nights', 'number_of_reviews', 'reviews_per_month',
                        'calculated_host_listings_count', 'availability_365']].corr(), annot=True, cmap='coolwarm')
        plt.savefig(
            Path.cwd().parents[1].joinpath(
                "reports/figures", "heatmap_reduced.png"))
        plt.show()

        # Analyze the top neighborhoods by average price
        # This code identifies the top neighborhoods based on the average price,
        # providing insights into the most expensive areas in the city.
        top_neighborhoods = neighborhood_stats.sort_values(by='mean', ascending=False).head(10)
        print("Top Neighborhoods by Average Price:")
        print(top_neighborhoods)

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
    def review_analysis(listings_df, reviews_df):
        """
        Analyze the relationship between price and number of reviews

        - There is a weak positive correlation between the price of a listing and the number
        of reviews it has received.
        - This suggests that higher-priced listings tend to receive more reviews, possibly due to
        their popularity or quality.
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
        print(f"Correlation between price and number of reviews: {correlation:.2f}")

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
        print(f"ANOVA test results:")
        print(f"F-statistic: {f_statistic:.2f}")
        print(f"p-value: {p_value:.4f}")

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
        reviews_df['review_length'] = reviews_df['comments'].apply(len)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_title('Distribution of Review Lengths')
        ax.set_xlabel('Review Length')
        ax.set_ylabel('Count')
        sns.histplot(data=reviews_df, x='review_length', ax=ax, kde=True)
        plt.savefig(
            Path.cwd().parents[1].joinpath(
                "reports/figures", "reviews_length.png"))
        plt.show()

        """
        Reviews per day
        """
        reviews_df['date'] = pd.to_datetime(reviews_df['date'])
        daily_reviews = reviews_df.groupby(
            reviews_df['date'].dt.date).size().reset_index(name='count')
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
        print("Transform text to an embedding vector space")
        emb_service = EmbeddingService()
        emb_reviews_df = emb_service.get_embeddings(reviews_df, 'comments')

        print("Split the data into training and testing sets")
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
    def exploratory_analysis(df):
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
        Visualize the distribution of listings by host

        - The distribution of listings by host reveals that a small number of hosts have a large number 
        of listings, while the majority of hosts have only a few listings due to professional hosts or 
        property management companies present in the market.
        - Similar results can be drawn from the figure of distribution on rental prices.
        """
        plt.figure(figsize=(10, 6))
        plt.title('Distribution of Listings by Host')
        plt.xlabel('Number of Listings')
        plt.ylabel('Count')
        sns.histplot(data=df, x='calculated_host_listings_count', kde=True)
        plt.savefig(
            Path.cwd().parents[1].joinpath(
                "reports/figures", "listings_distribution.png"))
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
                    order=['within an hour', 'within a few hours', 'within a day', 'a few days or more'])
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


if __name__ == '__main__':
    """
    I explored the data to check if there are trends between
    the explanatory variables and the target variable.
    """
    dea_df = process_full_listings(parent=1, verbose=False)
    review_df = get_reviews_desc(parent=1)

    EDA.feature_correlation_analysis(dea_df)
    EDA.exploratory_analysis(dea_df)
    EDA.neighborhood_analysis(dea_df)
    EDA.review_analysis(dea_df, review_df)
