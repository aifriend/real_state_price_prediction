# Airbnb Rental Price Prediction in Madrid

## Introduction
This report analyzes Airbnb rental data for the city of Madrid, with the goal of predicting rental prices for a client's properties located in various neighborhoods. The data comes from the Inside Airbnb project. 

## Data Exploration
- Key features include: neighborhood, property type, number of bedrooms, number of bathrooms, amenities, review scores, etc.
- The target variable is the rental price per night
- Larger properties with more bedrooms and bathrooms tend to have higher rental prices
- Listings with higher review scores also correlate with higher prices
- The correlation heatmap shows high correlations between maximum and minimum number of days, and between the different review scoring values like communication, location, check-in, etc. 
- Other values like cleaning fees, guests included, and extra people fees correlate with room characteristics like number of bedrooms, beds, bathrooms, etc.
- There are also correlations between number of reviews and total reviews per month, which seems logical. More significantly, there is a relationship between price and year-round availability of the rental.
- Shorter response times are associated with lower price ranges, while longer response times allow the price ranges to increase. There is more supply of apartments with very high availability, either year-round or for a few days or months.
- The vast majority of listings are concentrated in a small set of accommodation types: apartments, lofts, houses, condominiums, and a few other types. Other types have very few listings.
- Sol, Embajadores, Universidad, Justicia, Argüelles are some of the areas of Madrid with the most rental listings. Reviews have been increasing almost exponentially over the years, especially from 2016 to 2020. The longer the response time, the higher the positive review value. 
- Entire flats or apartments have a similar distribution of reviews as private rooms. Hotel rooms and shared rooms are distributed to a lesser extent. 
- The review clusters are grouped homogeneously, with little dispersion, although there are some groups that deviate significantly from the mean.
- The average length of reviews ranges from a few characters to 180 characters (the sample has been trimmed from 100 characters). 
- There are neighborhoods like Moratalaz where prices are very dispersed and other neighborhoods like Villaverde where prices are very limited in a very short price range.

There are more reviews on low-priced accommodations than on high-priced accommodations. The average price per dwelling per day is $60.

## Modeling
- The data was split into train and test sets 
- Several regression models were evaluated, including linear regression, decision tree, random forest, and gradient boosting
- Hyperparameter tuning was performed using grid search cross validation
- The XGBoost regression model achieved the following metrics:
  - Root Mean Squared Error (RMSE): 54.85 
  - Mean Absolute Percentage Error (MAPE): 113.49%
  - R-squared (R²): -1.04
- The Isolation Forest model predicted review scores with an accuracy of 0.56.
- The model's feature importances indicate that location (neighborhood), size (bedrooms/bathrooms), and review scores are the biggest drivers of price

## Interpretation and Recommendations
1. The XGBoost model's high RMSE and low R-squared values indicate that it is struggling to accurately predict prices based on the provided features. The high MAPE also shows the predictions are off by a large percentage on average. More feature engineering and model tuning is likely needed to improve the price prediction performance.
2. The Isolation Forest model's accuracy of 0.56 for predicting review scores is only slightly better than random chance. This suggests the model is not able to reliably distinguish between different review score levels.
3. Focus on acquiring properties in high-value neighborhoods like Sol, Embajadores and Universidad, where predicted rental prices are highest. Avoid low-price areas like El Plantio.
4. Properties with more bedrooms and bathrooms command significantly higher rents. If possible, look for larger properties or consider renovating to add additional rooms.
5. Aim to achieve high review scores by providing great customer service, amenities, and ensuring the property is well-maintained. Higher reviewed properties can charge a premium.
6. Utilize the price prediction model to optimize your pricing. Input the specific characteristics of each property to get a customized price prediction. You may want to charge slightly lower than the model price to stay competitive.
7. Consider additional amenities that tend to result in higher prices, such as air conditioning, wiffi, parking, etc.
8. Monitor competitor prices in each neighborhood to ensure your listings remain competitive. Adjust pricing dynamically based on seasonality and demand.

## For the client looking to rent out properties on Airbnb, I would recommend
1. Focus listings in the most popular neighborhoods like Sol, Embajadores, Universidad, Justicia and Argüelles to maximize visibility.
2. Aim to have very high availability year-round or at least for full days/months at a time, as this correlates with higher prices.
3. Provide amenities that correlate with higher prices like more bedrooms, beds, bathrooms, and allow for extra guests and charge cleaning fees.
4. Maintain short response times to inquiries, as this is associated with being able to charge higher prices.
5. Solicit reviews from guests, as listings with more reviews can command higher prices. Aim for longer, more detailed reviews.
6. Set competitive prices in the $50-70 per night range to start, as this is the average. Avoid neighborhoods with very dispersed prices.
7. Consider pricing on the lower end of the range to attract more bookings and reviews initially, then raise prices over time.
