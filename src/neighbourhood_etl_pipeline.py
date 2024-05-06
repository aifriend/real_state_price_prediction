from pathlib import Path

import folium
import seaborn as sns
from matplotlib import pyplot as plt
from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler

from src.listing_etl_pipeline import process_full_listings


def normalize(df: DataFrame, column: str) -> DataFrame:
    """
    Normalize a column in a dataframe

    :param df: dataframe
    :param column: column to normalize
    :return: normalized dataframe
    """
    scaler = MinMaxScaler()
    df[column] = scaler.fit_transform(df[[column]])
    return df


def avg_price_per_sqft(rentals_df: DataFrame, price_col: str, sqft_col: str) -> DataFrame:
    """
    Calculate average rental price per square foot for different neighborhoods or zip codes.
    This will allow you to map out which areas are more or less expensive.

    :param rentals_df: dataframe
    :param price_col: price column
    :param sqft_col: square feet column
    :return: aggregated dataframe
    """
    # Normalize price and square footage
    rentals_df = normalize(rentals_df, price_col)
    rentals_df = normalize(rentals_df, sqft_col)
    rentals_df['price_per_sqft'] = rentals_df[price_col] / rentals_df[sqft_col]
    price_by_neighborhood = rentals_df.groupby('neighbourhood').agg({'price_per_sqft': 'mean'})

    plt.figure(figsize=(12, 8))
    plt.title('Price Distribution by Neighborhood square foots')
    plt.xlabel('Neighborhood')
    plt.ylabel('price_per_sqft')
    plt.xticks(rotation=45)
    sns.boxplot(data=price_by_neighborhood, x='neighbourhood', y='price_per_sqft')
    plt.savefig(
        Path.cwd().parents[1].joinpath(
            "reports/figures", "neighborhood_by_price_per_sqft_box_plot.png"))
    plt.show()

    return rentals_df


def get_geodata_map(df) -> None:
    """
    Read the property data into a GeoDataFrame

    :param df: dataframe
    :return: None
    """
    # Get the center coordinates of the property locations
    center_lat = df.geometry.centroid.y.mean()
    center_lon = df.geometry.centroid.x.mean()

    # Create a Folium map centered on the mean coordinates
    map_center = [center_lat, center_lon]
    map_zoom = 12
    folium_map = folium.Map(location=map_center, zoom_start=map_zoom)

    # Add property locations as markers to the map
    for idx, row in df.iterrows():
        location = [row.geometry.centroid.y, row.geometry.centroid.x]
        popup_text = (f"Property Neighbourhood Location: {row['neighbourhood']}<br>"
                      f"Property Neighbourhood Area: {row['neighbourhood_group']}")
        folium.Marker(location=location, popup=popup_text).add_to(folium_map)

    # Save the map to an HTML file
    folium_map.save(
        Path.cwd().parents[1].joinpath(
            "reports/figures", "property_map.html"))


if __name__ == '__main__':
    parent = 0

    # geometry_col = gpd.read_file(
    #     Path.cwd().parents[0].joinpath(
    #         "data/raw", "neighbourhoods.geojson"))
    # get_geodata_map(geometry_col)

    # process full listing
    listing_df = process_full_listings(
        store_path='data/processed', cached=True, parent=parent)

    listing_df = avg_price_per_sqft(
        listing_df, 'price', 'square_feet')
