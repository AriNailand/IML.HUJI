import sklearn

from IMLearn import BaseEstimator
from challenge.agoda_cancellation_estimator import AgodaCancellationEstimator
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd


def load_data(filename: str):
    """
    Load Agoda booking cancellation dataset
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector in either of the following formats:
    1) Single dataframe with last column representing the response
    2) Tuple of pandas.DataFrame and Series
    3) Tuple of ndarray of shape (n_samples, n_features) and ndarray of shape (n_samples,)
    """
    # TODO - replace below code with any desired preprocessing
    # todo what to do if data isnt full
    # clean data for unrealistic shit
    full_data = pd.read_csv(filename).drop_duplicates()

    features, labels = preprocessor(full_data)

    return features, labels

# todo remove test_y
def evaluate_and_export(estimator: BaseEstimator, X: np.ndarray, filename: str, test_y):
    """
    Export to specified file the prediction results of given estimator on given testset.

    File saved is in csv format with a single column named 'predicted_values' and n_samples rows containing
    predicted values.

    Parameters
    ----------
    estimator: BaseEstimator or any object implementing predict() method as in BaseEstimator (for example sklearn)
        Fitted estimator to use for prediction

    X: ndarray of shape (n_samples, n_features)
        Test design matrix to predict its responses

    filename:
        path to store file at

    """
    y_pred = estimator.predict(X)
    pd.DataFrame(y_pred, columns=["predicted_values"]).to_csv(filename, index=False)
    print(sklearn.metrics.roc_auc_score(test_y, y_pred))


def preprocessor(full_data: np.ndarray):
    full_data.loc[full_data["cancellation_datetime"].isnull(), "cancellation_datetime"] = full_data["checkin_date"]

    # starting with the numerical columns
    features = full_data[["hotel_star_rating",
                          "guest_is_not_the_customer",
                          "original_selling_amount",
                          "is_user_logged_in",
                          "is_first_booking",
                          "original_payment_type",
                          "charge_option",
                          "original_payment_currency",
                          ]].fillna(0)

    for f in ["is_user_logged_in", "is_first_booking"]:
        features[f] = features[f].astype(int)

    # todo check how to edit the date to get it into days
    full_data['cancellation_datetime'] = pd.to_datetime(full_data["cancellation_datetime"])
    full_data['booking_datetime'] = pd.to_datetime(full_data['booking_datetime'])
    full_data['checkin_date'] = pd.to_datetime(full_data['checkin_date'])
    full_data['checkout_date'] = pd.to_datetime(full_data['checkout_date'])

    features = pd.get_dummies(features, columns=["original_payment_type",
                                                 "charge_option",
                                                 "original_payment_currency"])

    features["days_to_checkin"] = (full_data["checkin_date"] - full_data["booking_datetime"]).dt.days
    features["length_of_stay"] = (full_data['checkout_date'] - full_data['checkin_date']).dt.days
    features["cancel_warning_days"] = (full_data['checkin_date'] - full_data['cancellation_datetime']).dt.days
    features["days_cancelled_after_booking"] = (full_data["cancellation_datetime"] - full_data["booking_datetime"]).dt.days
    # defining binary label based on weather the person:
    # 1. reservation at least 15 days before checkin
    # 2. cancelled 2 to 7 days before booking

    labels = (7 <= features["days_cancelled_after_booking"]) & (features["days_cancelled_after_booking"] <= 43) & (features["cancel_warning_days"] > 2)

    return features, labels


if __name__ == '__main__':
    np.random.seed(0)

    # Load data
    df, cancellation_labels = load_data("../datasets/agoda_cancellation_train.csv")

    train_X, test_X, train_y, test_y = sklearn.model_selection.train_test_split(df, cancellation_labels, test_size=0.25)

    # Fit model over data
    estimator = AgodaCancellationEstimator().fit(train_X, train_y)

    # Store model predictions over test set
    evaluate_and_export(estimator, test_X, "342473642_206200552_316457340.csv", test_y)
