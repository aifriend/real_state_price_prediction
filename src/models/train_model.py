import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.listing_process import get_dataset_for_training


class TrainService:
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

        Hyperparameter tuning improved the XGBoost model compared to its baseline configuration,
        resulting in the following performance metrics:
        Data	    RMSE	MAPE	R²
        Training	279	    0.05	0.99
        Validation	1040	0.14	0.90
        Test	    995	    0.13	0.90
        """
        best_model, best_score, best_param = (
            TrainService.hyperparameter_grid_search(X_train, y_train))

        # Train the best model on the entire dataset
        best_model.fit(X, y)

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

        # Decision Tree
        tree = DecisionTreeRegressor(min_samples_split=30, max_depth=10)
        tree.fit(rescaled_x_train, y_train)
        y_pred_tree = tree.predict(rescaled_x_test)
        rmse(tree, y_test, y_pred_tree)
        mape(tree, y_test, y_pred_tree)
        r2(tree, y_test, y_pred_tree)

        # Xgboost
        xgb = XGBRegressor(max_depth=3, n_jobs=-1)
        xgb.fit(rescaled_x_train, y_train)
        y_pred_xgb = xgb.predict(rescaled_x_test)
        rmse(xgb, y_test, y_pred_xgb)
        mape(xgb, y_test, y_pred_xgb)
        r2(xgb, y_test, y_pred_xgb)

        # Gradient Boosting
        boost = GradientBoostingRegressor(
            n_estimators=300, min_samples_split=20)
        boost.fit(rescaled_x_train, y_train)
        y_pred_boost = boost.predict(rescaled_x_test)
        rmse(boost, y_test, y_pred_boost)
        mape(boost, y_test, y_pred_boost)
        r2(boost, y_test, y_pred_boost)

        # Random Forest
        forest = RandomForestRegressor(
            n_estimators=300, max_depth=10, min_samples_split=30, n_jobs=-1, random_state=0)
        forest.fit(rescaled_x_train, y_train)
        y_pred_forest = forest.predict(rescaled_x_test)
        rmse(forest, y_test, y_pred_forest)
        mape(forest, y_test, y_pred_forest)
        r2(forest, y_test, y_pred_forest)

        # Feature importance
        importance = forest.feature_importances_
        for i, v in enumerate(importance):
            print('feature: %d, score: %.5f' % (i, v))
        sns.set()
        plt.bar([x for x in range(len(importance))], importance)
        plt.title("A barplot showing the significance of each feature")
        plt.show()

        return rescaled_x_train


if __name__ == '__main__':
    df = get_dataset_for_training()

    if df.isna().any().any():
        print(df.columns[df.isna().any()].tolist())
        raise ValueError()

    X = df.drop(['price'], axis=1).values
    y = df['price'].values
    print(X.shape)
    print(y.shape)

    Xtr, Xts, ytr, yts = train_test_split(
        X, y, test_size=0.3, random_state=42)

    resc_x_tr = TrainService.train(Xtr, Xts, ytr, yts)
    b_model, b_score, b_params = TrainService.xgboost_tuning(resc_x_tr, ytr)
    print("Grid search completed.")
    print("Best hyperparameters: ", b_params)
    print("Best score: ", b_score)

    """
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
