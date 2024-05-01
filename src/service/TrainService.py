import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import silhouette_score
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.service.DataService import DataService
from src.service.EmbeddingService import EmbeddingService


class TrainService:

    @staticmethod
    def train_review_sentiment_analysis(rev_df, parent=0):
        """
        In this method, we perform sentiment analysis on the reviews with these steps:
        1) We have a list of review texts without any sentiment labels.
        2) We tokenize the reviews by splitting them into words.
        3) We train a Word2Vec model on the tokenized reviews to learn word embeddings.
        4) We generate review embeddings by averaging the word embeddings for each review.
        5) We perform clustering using the KMeans algorithm on the review embeddings. Here, we specify the number of clusters as 2, assuming we want to group the reviews into two sentiment clusters (positive and negative).
        6) We evaluate the clustering performance using the silhouette score, which measures how well the reviews are clustered.
        7) We print the reviews along with their assigned cluster labels.
        8) We predict the cluster for a new review by tokenizing it, generating its embedding, and using the trained KMeans model.

        In this unsupervised approach, we don't have explicit sentiment labels. Instead, we rely on the clustering algorithm to group similar reviews together based on their embeddings. The assumption is that reviews with similar sentiments will have similar embeddings and will be clustered together.
        Note that the interpretation of the clusters as positive or negative sentiment may require manual inspection of the reviews within each cluster. Additionally, the optimal number of clusters may vary depending on the data and the desired granularity of sentiment analysis.
        """
        reviews = rev_df['comments']

        # Tokenize the reviews
        tokenized_reviews = [review.lower().split() for review in reviews]

        # Train a Word2Vec model
        model = Word2Vec(
            tokenized_reviews, vector_size=10000, window=5, min_count=1, workers=4)
        DataService.save_model(
            model, parent=parent, apex='train_review_sentiment_analysis')

        # Generate review embeddings by averaging word embeddings
        review_embeddings = []
        for review in tokenized_reviews:
            review_embedding = np.mean(
                [model.wv[word] for word in review if word in model.wv], axis=0)
            review_embeddings.append(review_embedding)

        # Perform clustering using KMeans
        num_clusters = 2
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(review_embeddings)

        # Evaluate clustering performance using silhouette score
        silhouette_avg = silhouette_score(review_embeddings, cluster_labels)
        print(f"Silhouette Score: {silhouette_avg:.2f}")

        # Print the reviews and their assigned cluster labels
        for review, label in zip(reviews, cluster_labels):
            print(f"Review: {review}")
            print(f"Cluster Label: {label}")
            print()

        # Predict the cluster for a new review
        new_review = "The room was spacious and clean, but the staff was rude."
        new_review_tokens = new_review.lower().split()
        new_review_embedding = np.mean(
            [model.wv[word] for word in new_review_tokens if word in model.wv], axis=0)
        new_review_cluster = kmeans.predict([new_review_embedding])[0]
        print(f"New Review: {new_review}")
        print(f"Predicted Cluster: {new_review_cluster}")

    @staticmethod
    def train_reviews_by_score(rev_df, parent):
        """
        The following code is used for text clustering and documents embedding. We want to represent reviews
        as vectors representation to be able to apply clustering algorithms to detect topics. Reviews are
        usually short sentences, thus, we should look for a suitable embedding approach for this situation
        like Sentence Transformers with BERT or Glove.

        We analyzed the relationship between the description of each listing and its price,
        and proposed a text-based price recommendation system called TAPE to recommend a reasonable price
        for newly added listings
        """
        print("Transform text to an embedding vector space")
        emb_service = EmbeddingService()
        listings_reviews_df = emb_service.get_embeddings(rev_df, 'comments')

        print("Split the data into training and testing sets")
        embeddings = listings_reviews_df['comments_emb'].tolist()
        labels = listings_reviews_df['review_scores_value'].tolist()
        X_train, X_test, y_train, y_test = train_test_split(
            embeddings, labels, test_size=0.2, random_state=42)

        print("Train a logistic regression model using embeddings")
        classifier = LogisticRegression()
        classifier.fit(X_train, y_train)
        DataService.save_model(
            classifier, parent, apex='train_reviews_by_score')

        return classifier, emb_service, X_test, y_test

    @staticmethod
    def hyperparameter_grid_search(X, y):
        """
        Hyperparameter Tuning

        Identified XGBoost and Gradient Boosting as the top two performers for hyperparameter tuning.
        Employed grid search with 5-fold cross-validation to find the best hyperparameter combinations.
        Selected the model that demonstrated the best performance on the validation data.
        XGBoost Hyperparameter Tuning:
        Hyperparameter      Explanation                                                 Values
        n_estimators	    Number of trees	                                            [100, 200, 300, 400, 500]
        max_depth	        Maximum depth of each tree	                                [3, 4, 5]
        subsample	        Fraction of samples used for fitting each tree	            [0.8, 0.9, 1.0]
        colsample_bytree	Fraction of features used for fitting each tree	            [0.8, 0.9, 1.0]
        learning_rate	    Rate at which the model adapts during training	            [0.1, 0.01, 0.001]
        min_child_weight	Minimum sum of instance weight (hessian) needed in a child	[1, 2, 3]
        gamma	            Minimum loss reduction required when splitting a node	    [0, 0.1, 0.2]
        """

        # Create the XGBRegressor model
        model = XGBRegressor()

        # Define the cross-validation strategy
        cv = KFold(n_splits=5, shuffle=True, random_state=42)

        param_grid = {
            'max_depth': [3, 4, 5],
            'learning_rate': [0.1, 0.01],
            'n_estimators': [100, 200, 300, 400, 500],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'min_child_weight': [1, 2, 3],
            'gamma': [0, 0.1, 0.2]
        }

        print("Performing grid search...")
        grid_search = GridSearchCV(
            estimator=model, param_grid=param_grid, cv=cv,
            scoring='neg_mean_squared_error', n_jobs=-1, verbose=2)
        grid_search.fit(X, y)

        best_model = grid_search.best_estimator_
        return best_model, -grid_search.best_score_, grid_search.best_params_

    @staticmethod
    def xgboost_tuning(X_train, y_train):
        """
        The chosen model was an XGBoost regression model with the following hyperparameters:
        Best hyperparameters:  {
            'colsample_bytree': 0.9,
            'gamma': 0,
            'learning_rate': 0.1,
            'max_depth': 5,
            'min_child_weight': 2,
            'n_estimators': 500,
            'subsample': 0.8
        }
        Best score:  47826.55584619551
        """
        best_model, best_score, best_param = (
            TrainService.hyperparameter_grid_search(X_train, y_train))

        # Train the best model on the entire dataset
        best_model.fit(X_train, y_train)

        return best_model, best_score, best_param

    @staticmethod
    def train(X_train, X_test, y_train, y_test):
        def rmse(model, y_true, y_pred):
            """
            Calculate the Root Mean Squared Error (RMSE) between the true and predicted values.

            Args:
                model: object
                y_true (array-like): True values.
                y_pred (array-like): Predicted values.

            Returns:
                float: RMSE value.
            """
            result = np.sqrt(mean_squared_error(y_true, y_pred))
            print(f"{model.__class__.__name__} Root Mean Squared Error (RMSE): {result}")
            return result

        def mape(model, y_true, y_pred):
            """
            Calculate the Mean Absolute Percentage Error (MAPE) between the true and predicted values.

            Args:
                y_true (array-like): True values.
                y_pred (array-like): Predicted values.

            Returns:
                float: MAPE value.
            """
            y_true, y_pred = np.array(y_true), np.array(y_pred)
            result = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
            print(f"{model.__class__.__name__} Mean Absolute Percentage Error (MAPE): {result}")
            return result

        def r2(model, y_true, y_pred):
            """
            Calculate the R-squared (R²) value between the true and predicted values.

            Args:
                y_true (array-like): True values.
                y_pred (array-like): Predicted values.

            Returns:
                float: R-squared value.
            """
            result = r2_score(y_true, y_pred)
            print(f"{model.__class__.__name__} R-squared (R²): {result}")
            return result

        # feature Scaling
        scaler = StandardScaler()
        rescaled_x_train = scaler.fit_transform(X_train)
        rescaled_x_test = scaler.transform(X_test)

        """
        Baseline Models
    
        Implemented five machine learning models with baseline parameter configurations and evaluated model performance 
        on the validation data based on root mean squared error (RMSE), mean absolute percentage error (MAPE), 
        and R-squared (R²).
    
        Model	                RMSE	MAPE	R²   
        Linear Regression       407     151     0.05
        Decision Tree           388     77      0.13
        Xgboost                 331     110     0.36
        Gradient Boosting       341     105     0.33    
        Random Forest           353     80      0.28
        """

        # linear regression
        lr = LinearRegression()
        lr.fit(rescaled_x_train, y_train)
        y_pred_lr = lr.predict(rescaled_x_test)
        rmse(lr, y_test, y_pred_lr)
        mape(lr, y_test, y_pred_lr)
        r2(lr, y_test, y_pred_lr)
        DataService.save_model(lr, parent=1, apex='price_prediction')

        # Decision Tree
        tree = DecisionTreeRegressor(min_samples_split=30, max_depth=10)
        tree.fit(rescaled_x_train, y_train)
        y_pred_tree = tree.predict(rescaled_x_test)
        rmse(tree, y_test, y_pred_tree)
        mape(tree, y_test, y_pred_tree)
        r2(tree, y_test, y_pred_tree)
        DataService.save_model(tree, parent=1, apex='price_prediction')

        # Xgboost
        xgb = XGBRegressor(
            colsample_bytree=0.9, gamma=0,
            learning_rate=0.1, min_child_weight=2,
            n_estimators=500, subsample=0.8,
            max_depth=5, n_jobs=-1)
        xgb.fit(rescaled_x_train, y_train)
        y_pred_xgb = xgb.predict(rescaled_x_test)
        rmse(xgb, y_test, y_pred_xgb)
        mape(xgb, y_test, y_pred_xgb)
        r2(xgb, y_test, y_pred_xgb)
        DataService.save_model(xgb, parent=1, apex='price_prediction')

        # Gradient Boosting
        boost = GradientBoostingRegressor(
            n_estimators=300, min_samples_split=20)
        boost.fit(rescaled_x_train, y_train)
        y_pred_boost = boost.predict(rescaled_x_test)
        rmse(boost, y_test, y_pred_boost)
        mape(boost, y_test, y_pred_boost)
        r2(boost, y_test, y_pred_boost)
        DataService.save_model(boost, parent=1, apex='price_prediction')

        # Random Forest
        forest = RandomForestRegressor(
            n_estimators=300, max_depth=10, min_samples_split=30, n_jobs=-1, random_state=0)
        forest.fit(rescaled_x_train, y_train)
        y_pred_forest = forest.predict(rescaled_x_test)
        rmse(forest, y_test, y_pred_forest)
        mape(forest, y_test, y_pred_forest)
        r2(forest, y_test, y_pred_forest)
        DataService.save_model(forest, parent=1, apex='price_prediction')

        return xgb, rescaled_x_train
