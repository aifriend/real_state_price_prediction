from pathlib import Path
from typing import List, Any

import numpy as np
import seaborn as sns
from gensim.models import Word2Vec
from matplotlib import pyplot as plt
from pandas import DataFrame
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import silhouette_score
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.service.DataService import DataService
from src.service.EmbeddingService import EmbeddingService
from src.service.RentalLogger import logger


class TrainService:

    @staticmethod
    def rmse(model: object, y_true: List, y_pred: List) -> float:
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
        logger.info(f"{model.__class__.__name__} Root Mean Squared Error (RMSE): {result}")
        return result

    @staticmethod
    def mape(model: object, y_true: List, y_pred: List) -> float:
        """
        Calculate the Mean Absolute Percentage Error (MAPE) between the true and predicted values.

        Args:
            model: object
            y_true (array-like): True values.
            y_pred (array-like): Predicted values.

        Returns:
            float: MAPE value.
        """
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        result = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        logger.info(f"{model.__class__.__name__} Mean Absolute Percentage Error (MAPE): {result}")
        return result

    @staticmethod
    def r2(model: object, y_true: List, y_pred: List) -> float:
        """
        Calculate the R-squared (R²) value between the true and predicted values.

        Args:
            model: object
            y_true (array-like): True values.
            y_pred (array-like): Predicted values.

        Returns:
            float: R-squared value.
        """
        result = r2_score(y_true, y_pred)
        logger.info(f"{model.__class__.__name__} R-squared (R²): {result}")
        return result

    @staticmethod
    def pre_process_for(df: DataFrame) -> Any:
        """
        Pre-process

        Args:
            df: DataFrame

        Returns:
            Any
        """
        # Drop meaningless columns
        df.drop(columns=['street', 'neighbourhood_cleansed',
                         'neighbourhood_group_cleansed', 'city',
                         ],
                inplace=True)

        # Convert objects columns to category if necessary
        # The categorical features have to be converted internally to
        # numerical features for efficient modeling
        label_encoder = LabelEncoder()
        column_list = df.select_dtypes(exclude=['int', 'float']).columns
        for column in column_list:
            df[column] = df[column].astype("category")
            # Apply LabelEncoder to the categorical column
            encoded_categories = label_encoder.fit_transform(df[column])
            category_mapping = dict(zip(df[column], encoded_categories))
            df[column] = encoded_categories
            logger.info(f"{column} ==> "
                        f"Apply label encoder with label "
                        f"{category_mapping}")

        # drop less significance features from feature selection with embedding method
        df.drop(
            columns=df.columns[
                [1, 7, 11, 12, 13, 15, 17, 18, 19, 20,
                 23, 27, 28, 29, 30, 31, 34, 36, 37, 38,
                 39, 40, 41, 42, 43, 47, 49, 50, 51]], inplace=True)

        # X and y
        X = df.drop(['price'], axis=1).values
        y = df['price'].values
        logger.info(X.shape)
        logger.info(y.shape)

        # Split the data into training and testing sets
        Xtr, Xts, ytr, yts = train_test_split(
            X, y, test_size=0.3, random_state=42)

        return Xtr, Xts, ytr, yts

    # Train
    @staticmethod
    def train_review_sentiment_analysis(rev_df: DataFrame, project_dir: Path, parent: int = 0) -> None:
        """
        In this method, I perform sentiment analysis on the reviews with these steps:
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

        Args:
            rev_df: reviews dataframe
            project_dir: project directory
            parent: parent id

        Returns:
            None
        """
        reviews = rev_df['comments']

        # Tokenize the reviews
        tokenized_reviews = [review.lower().split() for review in reviews]

        # Train a Word2Vec model
        model = Word2Vec(
            tokenized_reviews, vector_size=10000, window=5, min_count=1, workers=4)
        DataService.save_model(
            model, project_dir, parent=parent, apex='review_train_sentiment_analysis')

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
        logger.info(f"Silhouette Score: {silhouette_avg:.2f}")

        # Print the reviews and their assigned cluster labels
        for review, label in zip(reviews, cluster_labels):
            logger.info(f"Review: {review}")
            logger.info(f"Cluster Label: {label}")
            break

        # Predict the cluster for a new review
        new_review = "The room was spacious and clean, but the staff was rude."
        new_review_tokens = new_review.lower().split()
        new_review_embedding = np.mean(
            [model.wv[word] for word in new_review_tokens if word in model.wv], axis=0)
        new_review_cluster = kmeans.predict([new_review_embedding])[0]
        logger.info(f"New Review: {new_review}")
        logger.info(f"Predicted Cluster: {new_review_cluster}")

    @staticmethod
    def train_reviews_by_score(rev_df: DataFrame, project_dir: Path, parent: int = 0) -> (
            LogisticRegression, EmbeddingService, List, List):
        """
        The following code is used for text clustering and documents embedding. We want to represent reviews
        as vectors representation to be able to apply clustering algorithms to detect topics. Reviews are
        usually short sentences, thus, we should look for a suitable embedding approach for this situation
        like Sentence Transformers with BERT or Glove.

        We analyzed the relationship between the description of each listing and its price,
        and proposed a text-based price recommendation system called TAPE to recommend a reasonable price
        for newly added listings

        Args:
            rev_df: reviews dataframe
            project_dir: project directory
            parent: parent id

        Returns:
            classifier: classifier
            emb_service: embedding service
            X_test: test data
            y_test: test labels
        """
        logger.info("Transform text to an embedding vector space")
        emb_service = EmbeddingService()
        listings_reviews_df = emb_service.get_embeddings(rev_df, 'comments')

        logger.info("Split the data into training and testing sets")
        embeddings = listings_reviews_df['comments_emb'].tolist()
        labels = listings_reviews_df['review_scores_value'].tolist()
        X_train, X_test, y_train, y_test = train_test_split(
            embeddings, labels, test_size=0.2, random_state=42)

        logger.info("Train a logistic regression model using embeddings")
        classifier = LogisticRegression()
        classifier.fit(X_train, y_train)
        DataService.save_model(
            classifier, project_dir, parent, apex='review_train_score_str')

        return classifier, emb_service, X_test, y_test

    @staticmethod
    def xgboost_tuning(X_train, y_train) -> (XGBRegressor, float, float):
        """
        The chosen model was an XGBoost regression model with the following hyperparameters:
        Best hyperparameters:
        colsample_bytree: 1.0
        gamma: 0
        learning_rate: 0.1
        max_depth: 5
        min_child_weight: 3
        n_estimators: 400
        subsample: 0.9

        Args:
            X_train: training data
            y_train: training labels

        Returns:
            None
        """

        def hyperparameter_grid_search(X, y) -> (XGBRegressor, float, float):
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

            Args:
                X: training data
                y: training labels

            Returns:
                best_model_: best XGBRegressor model
                best_score_: best score
                best_params_: best dictionary of hyperparameters
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

            logger.info("Performing grid search...")
            grid_search = GridSearchCV(
                estimator=model, param_grid=param_grid, cv=cv,
                scoring='neg_mean_squared_error', n_jobs=-1, verbose=2)
            grid_search.fit(X, y)

            best_model_ = grid_search.best_estimator_
            return best_model_, -grid_search.best_score_, grid_search.best_params_

        best_model, best_score, best_param = hyperparameter_grid_search(X_train, y_train)

        # Train the best model on the entire dataset
        best_model.fit(X_train, y_train)

        return best_model, best_score, best_param

    @staticmethod
    def train(X_train, X_test, y_train, y_test, project_dir: Path) -> (XGBRegressor, List):
        """
        Train the model

        Baseline Models
        Implemented five machine learning models with baseline parameter configurations and
        evaluated model performance on the validation data based on root mean squared error (RMSE),
        mean absolute percentage error (MAPE), and R-squared (R²).

        Model	                RMSE	MAPE	R²
        Linear Regression       407     151     0.05
        Decision Tree           388     77      0.13
        Xgboost                 331     110     0.36
        Gradient Boosting       341     105     0.33
        Random Forest           353     80      0.28

        Args:
            X_train: training data
            X_test: test data
            y_train: training labels
            y_test: test labels
            project_dir: project directory

        Returns:
            XGBRegressor: trained model
            List: list of X values for training
        """

        # feature Scaling
        scaler = StandardScaler()
        rescaled_x_train = scaler.fit_transform(X_train)
        rescaled_x_test = scaler.transform(X_test)

        """
        Baseline Models
        """

        # linear regression
        lr = LinearRegression()
        lr.fit(rescaled_x_train, y_train)
        y_pred_lr = lr.predict(rescaled_x_test)
        TrainService.rmse(lr, y_test, y_pred_lr)
        TrainService.mape(lr, y_test, y_pred_lr)
        TrainService.r2(lr, y_test, y_pred_lr)
        DataService.save_model(lr, project_dir, parent=1, apex='price_train')

        # Decision Tree
        tree = DecisionTreeRegressor(min_samples_split=30, max_depth=10)
        tree.fit(rescaled_x_train, y_train)
        y_pred_tree = tree.predict(rescaled_x_test)
        TrainService.rmse(tree, y_test, y_pred_tree)
        TrainService.mape(tree, y_test, y_pred_tree)
        TrainService.r2(tree, y_test, y_pred_tree)
        DataService.save_model(tree, project_dir, parent=1, apex='price_train')

        # Xgboost
        xgb = XGBRegressor(
            colsample_bytree=1.0,
            gamma=0,
            learning_rate=0.1,
            max_depth=5,
            min_child_weight=3,
            n_estimators=400,
            subsample=0.9,
            n_jobs=-1)
        xgb.fit(rescaled_x_train, y_train)
        y_pred_xgb = xgb.predict(rescaled_x_test)
        TrainService.rmse(xgb, y_test, y_pred_xgb)
        TrainService.mape(xgb, y_test, y_pred_xgb)
        TrainService.r2(xgb, y_test, y_pred_xgb)
        DataService.save_model(xgb, project_dir, parent=1, apex='price_train')

        # Feature importance
        # Plot feature importance
        importance = xgb.feature_importances_
        # for i, v in enumerate(importance):
        #     logger.info('feature: %d, score: %.5f' % (i, v))
        sns.set()
        plt.bar([x for x in range(len(importance))], importance)
        plt.title("A barplot showing the significance of each feature from XGBOOST")
        plt.show()

        # Gradient Boosting
        boost = GradientBoostingRegressor(
            n_estimators=300, min_samples_split=20)
        boost.fit(rescaled_x_train, y_train)
        y_pred_boost = boost.predict(rescaled_x_test)
        TrainService.rmse(boost, y_test, y_pred_boost)
        TrainService.mape(boost, y_test, y_pred_boost)
        TrainService.r2(boost, y_test, y_pred_boost)
        DataService.save_model(boost, project_dir, parent=1, apex='price_train')

        # Random Forest
        forest = RandomForestRegressor(
            n_estimators=300, max_depth=10, min_samples_split=30, n_jobs=-1, random_state=0)
        forest.fit(rescaled_x_train, y_train)
        y_pred_forest = forest.predict(rescaled_x_test)
        TrainService.rmse(forest, y_test, y_pred_forest)
        TrainService.mape(forest, y_test, y_pred_forest)
        TrainService.r2(forest, y_test, y_pred_forest)
        DataService.save_model(forest, project_dir, parent=1, apex='price_train')

        # Feature importance
        # Plot feature importance
        importance = forest.feature_importances_
        # for i, v in enumerate(importance):
        #     logger.info('feature: %d, score: %.5f' % (i, v))
        sns.set()
        plt.bar([x for x in range(len(importance))], importance)
        plt.title("A barplot showing the significance of each feature from FOREST")
        plt.show()

        return xgb, rescaled_x_train
