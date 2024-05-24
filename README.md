# Technical Challenge

## Documentation

[index.html](docs%2Fhtml)

## Environment

Setup:

    make requirements 

## Loading datasets

Loading and preparation of data:

    make data /PROJECT_DIR/data/raw /PROJECT_DIR/data/interim

## Feature extraction

    make features /PROJECT_DIR/data/interim /PROJECT_DIR/data/processed

## Analysis

### Exploratory analysis (EDA)

* Describe the situation of the temporary rental market on Airbnb in the city of Madrid at a general level.
* Visual descriptions and exploitation of the richness of the data will be valued.
  * Making use of geospatial data and/or natural language.

#### Features
![features_significance.png](reports/figures/features_significance.png)
![feature_corr_heatmap.png](reports/figures/feature_corr_heatmap.png)
![corr_heatmap.png](reports/figures/corr_heatmap.png)

#### Listings
![host_response_time_by_price.png](reports/figures/host_response_time_by_price.png)
![listing_property_type_distro.png](reports/figures/listing_property_type_distro.png)
![listings_availability.png](reports/figures/listings_availability.png)
![room_types_distribution.png](reports/figures/room_types_distribution.png)

#### Neighbourhood
![listing_by_neigbourhood_distro.png](reports/figures/listing_by_neigbourhood_distro.png)
##### Geolocations are shown in this map:
![neighbourhood_madrid.png](reports/figures/neighbourhood_madrid.png)
[property_map.html](reports%2Fproperty_map.html)

#### Reviews
![review_by_daily.png](reports/figures/review_by_daily.png)
![review_by_response_rate.png](reports/figures/review_by_response_rate.png)
![review_by_room_type.png](reports/figures/review_by_room_type.png)
![review_category_distro.png](reports/figures/review_category_distro.png)
![review_clusters.png](reports/figures/review_clusters.png)
![review_length_distro.png](reports/figures/review_length_distro.png)

### Analyze the prices of published properties.
![price_by_neighbourhood.png](reports/figures/price_by_neighbourhood.png)
![price_by_neighbourhood_distro.png](reports/figures/price_by_neighbourhood_distro.png)
![price_by_property_type.png](reports/figures/price_by_property_type.png)
![price_by_reviews.png](reports/figures/price_by_reviews.png)
![rental_prices_distribution.png](reports/figures/rental_prices_distribution.png)

## Prediction

* Train a model capable of predicting the daily rental price of a property on Airbnb.
* In addition to the metrics obtained, the justification of the model construction process will be assessed. Creativity in the construction of new variables (using geolocation data and/or unstructured text) and the use of different predictive techniques.
  * variables used/discarded
  * evaluation metric/s, model selection/s


    make train /PROJECT_DIR/data/processed
    make predict /PROJECT_DIR/data/processed


## Deployment

[Deployment solution](references%2FCHALLENGE.md)

## Report

[Report of results](references%2FREPORT.md)
