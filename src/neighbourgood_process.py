from pathlib import Path

import folium
import geopandas as gpd


def visualize_property_locations(property_geo_data, output_interactive_file):
    # Read the property data into a GeoDataFrame
    property_gdf = gpd.read_file(
        Path.cwd().parents[0].joinpath("data/raw", property_geo_data))

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
        Path.cwd().parents[0].joinpath("src/visualization", output_interactive_file))


if __name__ == '__main__':
    property_data = "neighbourhoods.geojson"  # Path to the property data file
    output_file = "property_map.html"  # Output file name for the interactive map

    visualize_property_locations(property_data, output_file)
