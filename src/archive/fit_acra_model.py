"""
Train a Random Forest model to predict company incorporation dates (in months since year 2000)
based on features from Google POI and matched ACRA data.

- Filters NA values from feature columns
- Converts incorporation date to numerical target
- Splits dataset into training and test sets
- Trains a RandomForestRegressor
- Prints MAE and R² score
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score


def fit_acra_model(df_merged):
    """
    Trains and evaluates a Random Forest model to predict incorporation date from POI features.
    """
    x_columns = [
        'user_ratings_total', 'rating', 'delta_rating_count',
        'lat', 'lng', 'google_live', 'acra_live']
    df_merged = df_merged.dropna(subset=x_columns)

    X = df_merged[x_columns]
    date = pd.to_datetime(df_merged["registration_incorporation_date"])

    y = (date.dt.year - 2000) * 12 + date.dt.month

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=1)

    model = RandomForestRegressor(
        n_estimators=50, random_state=1, max_depth=10)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Absolute Error: {mae:.2f}")
    print(f"R² Score: {r2:.4f}")

    return model, mae, r2
