from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

from src.listing_process import get_dataset_for_training


class PredictService:

    @staticmethod
    def predict_with_tuned_model(model, X_test):
        # Make predictions using the tuned model
        y_pred = model.predict(X_test)
        return y_pred

    @staticmethod
    def create_xgb_regressor(n_estimators, max_depth, learning_rate):
        model = XGBRegressor(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate)
        return model

    @staticmethod
    def predict(X, y):
        params = {
            'colsample_bytree': 0.9,
            'gamma': 0,
            'learning_rate': 0.1,
            'max_depth': 5,
            'min_child_weight': 2,
            'n_estimators': 500,
            'subsample': 0.8
        }

        # Create the best model using the best hyperparameters
        best_model = PredictService.create_xgb_regressor(**params)

        # Train the best model on the entire dataset
        best_model.fit(X, y)

        # Make predictions using the tuned model
        y_pred = PredictService.predict_with_tuned_model(best_model, X)

        # Print the predictions
        print("Predictions:")
        print(y_pred)


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

    PredictService.predict(Xts, yts)
