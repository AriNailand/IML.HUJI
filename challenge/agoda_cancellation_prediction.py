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

    full_data["cancellation_datetime"].fillna(value=0)

    # starting with the numerical columns
    features = full_data[["hotel_id",
                          "hotel_star_rating",
                          "guest_is_not_the_customer",
                          "no_of_adults",
                          "no_of_children",
                          "no_of_extra_bed",
                          "no_of_room",
                          "original_selling_amount",
                          "is_user_logged_in",
                          "is_first_booking",
                          "request_nonesmoke",
                          "request_highfloor",
                          "request_largebed",
                          "request_twinbeds",
                          "request_airport",
                          "request_earlycheckin",
                          "hotel_area_code",
                          "hotel_brand_code",
                          "hotel_chain_code",
                          "hotel_city_code"
                          ]]

    for f in ["is_user_logged_in", "is_first_booking"]:
        features[f] = features[f].astype(int)

    # todo check how to edit the date to get it into days
    full_data['cancellation_datetime'] = pd.to_datetime(full_data["cancellation_datetime"]).date()
    full_data['booking_datetime'] = pd.to_datetime(full_data['booking_datetime']).date()
    full_data['checkin_date'] = pd.to_datetime(full_data['checkin_date']).date()
    full_data['checkout_date'] = pd.to_datetime(full_data['checkout_date']).date()

    features["days_to_checkin"] = (full_data["checkin_date"] - full_data["booking_datetime"]).days
    features["length_of_stay"] = (full_data['checkout_date'] - full_data['checkin_date']).days
    if full_data['cancellation_datetime'] == 0:
        features["cancel_warning_days"] = 0
    else:
        features["cancel_warning_days"] = (full_data['checkin_date'] - full_data['cancellation_datetime']).days

    # defining binary label based on weather the person:
    # 1. reservation at least 15 days before checkin
    # 2. cancelled 2 to 7 days before booking
    labels = full_data["cancellation_datetime"]


    return features, labels


def evaluate_and_export(estimator: BaseEstimator, X: np.ndarray, filename: str):
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
    pd.DataFrame(estimator.predict(X), columns=["predicted_values"]).to_csv(filename, index=False)


if __name__ == '__main__':
    np.random.seed(0)

    # Load data
    df, cancellation_labels = load_data("../datasets/agoda_cancellation_train.csv")
    train_X, train_y, test_X, test_y = split_train_test(df, cancellation_labels)

    # Fit model over data
    estimator = AgodaCancellationEstimator().fit(train_X, train_y)

    # Store model predictions over test set
    evaluate_and_export(estimator, test_X, "id1_id2_id3.csv")
