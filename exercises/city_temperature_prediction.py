from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    # opening file
    df = pd.read_csv(filename, parse_dates=True).dropna().drop_duplicates()

    # adding days of year
    df["DayOfYear"] = pd.to_datetime(df["Date"]).dt.dayofyear
    df.drop(["Date"], axis=1)

    # removing invalid data
    df = df[df["Month"] <= 12]
    df = df[df["Month"] > 0]
    df = df[df["Day"] <= 31]
    df = df[df["Day"] > 0]
    df = df[df["Temp"] > -50]

    return df


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    full_data = load_data("/Users/aryehnailand/Desktop/HUJICSdegree/Sem5/IML/IML.HUJI/datasets/City_Temperature.csv")

    # # Question 2 - Exploring data for specific country (israel)
    israel_data = full_data[full_data["Country"] == "Israel"]
    # day of year vs temp plot
    israel_data["Year"] = israel_data["Year"].astype(str)
    fig1 = px.scatter(israel_data, x="DayOfYear", y="Temp", color="Year", title="Temperature By The Day of Year")
    fig1.show()
    # bar plot of std of each month
    std_of_months = israel_data.groupby(['Month']).Temp.agg(['std'])
    fig2 = px.bar(std_of_months, y='std',
                  labels={'std': 'SD', 'x': 'Month'},
                  title="SD of Temperature By Month")
    fig2.show()

    # Question 3 - Exploring differences between countries
    mean_std_country_month = full_data.groupby(['Month', 'Country']).Temp.agg(['std', 'mean']).reset_index()
    fig3 = px.line(mean_std_country_month, x=['Month'], y='mean', error_y='std', color='Country',
                   labels={'x': "Month", "y": "Temp Mean"},
                   title="Average Temperature by Month with Confidence Interval")
    fig3.show()

    # Question 4 - Fitting model for different values of `k`
    israel_data = israel_data.drop(["City", "Country", "Year", "Month", "Year", "Date", "Day"], axis=1)
    train_X, train_y, test_X, test_y = split_train_test(israel_data["DayOfYear"], israel_data["Temp"])
    test_loss = []
    for k in range(1, 11):
        poly_fit = PolynomialFitting(k).fit(train_X, train_y)
        loss = np.round(poly_fit.loss(test_X, test_y), decimals=2)
        test_loss.append(loss)
        print(f'Test Loss for k = {k}: {loss}')
    fig4 = px.bar(x=range(1, 11), y=test_loss,
                  labels={'y': 'test_loss', 'x': 'degree k'},
                  title="Test Loss by The Degree K")
    fig4.show()

    # Question 5 - Evaluating fitted model on different countries
    K = 5  # lowest loss on the samples
    full_data = full_data.drop(["City", "Year", "Month", "Year", "Date", "Day"], axis=1)
    poly_fit = PolynomialFitting(K).fit(israel_data["DayOfYear"], israel_data["Temp"])
    countries = ["Israel", "South Africa", "Jordan", "The Netherlands"]
    test_lost_israel_fitted = []
    for country in countries:
        country_data = full_data[full_data["Country"] == country]
        test_lost_israel_fitted.append(poly_fit.loss(country_data["DayOfYear"], country_data["Temp"]))

    fig5 = px.bar(x=countries, y=test_lost_israel_fitted,
                  labels={"y": "test_loss", "x": "countries"},
                  title="Test Loss Per Country")
    fig5.show()
