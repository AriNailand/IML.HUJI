from IMLearn.learners.regressors import LinearRegression
from typing import NoReturn
from IMLearn.utils import split_train_test
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"


def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    df = pd.read_csv(filename).dropna().drop_duplicates()

    # these features dont help
    df = df.drop(["id", "lat", "long"], axis=1)

    # all of these must be greater or equal to zero
    for c in ["bathrooms", "sqft_basement", "price"]:
        df = df[df[c] >= 0]

    # all of these must be greater than 0
    for col in ["sqft_living", "sqft_lot", "sqft_above", "sqft_living15", "sqft_lot15", "floors"]:
        df = df[df[col] > 0]

    # we want zipcode as an int
    df["zipcode"] = df["zipcode"].astype(int)

    # only want the year of the date
    df["date"] = df["date"].str[:4]

    # seperate every five year grouping of houses built period
    df["5_yrs_built_group"] = (df["yr_built"].astype(int) / 5).astype(int)

    df["yr_renovated"] = np.where(df["yr_renovated"].astype(int) == 0, df["yr_built"], df["yr_renovated"])

    df = df.drop(["yr_built"], axis=1)

    # check only for logical range of these values
    df = df[df["waterfront"].isin([0, 1])]
    df = df[df["view"].isin(range(10))]
    df = df[df["condition"].isin(range(1, 10))]
    df = df[df["grade"].isin(range(1, 20))]

    # changing zip code and year built group to categorical variables
    df = pd.get_dummies(df, columns=['zipcode', '5_yrs_built_group'])

    df.insert(0, 'intercept', 1, True)

    return df.drop(["price"], axis=1).astype(int), df["price"].astype(int)


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    X = X.loc[:, ~(X.columns.str.contains('^zipcode_') | X.columns.str.contains('^5_yrs_built_group'))].drop("intercept", axis=1)

    for f in X:
        pearson_corr = np.cov(X[f], y)[0, 1] / (np.std(X[f]) * np.std(y))

        figure = px.scatter(pd.DataFrame({'x': X[f], 'y': y}), x="x", y="y", trendline="ols",
                            title=f'Correlation Between {f} Values and Response <br>Pearson Correlation {pearson_corr}',
                            labels={"x": f"{f} Values", "y": "Response Values"})

        figure.write_image(output_path % f)


if __name__ == '__main__':
    np.random.seed(0)

    # Question 1 - Load and preprocessing of housing prices dataset
    design_mat, response_vec = load_data("/Users/aryehnailand/Desktop/HUJICSdegree/Sem5/IML/IML.HUJI/datasets/house_prices.csv")

    # Question 2 - Feature evaluation with respect to response
    feature_evaluation(design_mat, response_vec, "Pearson_Correlation_%s.png")

    # Question 3 - Split samples into training- and testing sets.
    train_X, train_y, test_X, test_y = split_train_test(design_mat, response_vec)

    # Question 4 - Fit model over increasing percentages of the overall training data
    avg_loss, var_loss = [], []
    for p in range(10, 101):
        loss = []
        for time in range(10):
            indexes = np.random.randint(0, train_X.shape[0] - 1, int((p / 100) * train_X.shape[0]))
            train_sample_X = np.asarray(train_X)[indexes]
            train_sample_y = np.asarray(train_y)[indexes]
            fitted_model = LinearRegression().fit(train_sample_X, train_sample_y)
            loss.append(fitted_model.loss(test_X, test_y))
        avg_loss.append(np.sum(np.asarray(loss)) / 10)
        var_loss.append(np.std(np.asarray(loss)))

    mean_pred, var_pred = np.asarray(avg_loss), np.asarray(var_loss)
    x = np.linspace(10, 101)
    fig = go.Figure([go.Scatter(x=x, y=mean_pred, mode="markers+lines", name="Mean Prediction",
                                line=dict(dash="dash"), marker=dict(color="green", opacity=.7)),
                     go.Scatter(x=x, y=mean_pred + 2 * var_pred, name="Upper confidence bound (2SD's)",
                                fill=None, mode="lines", line=dict(color="lightgrey")),
                     go.Scatter(x=x, y=mean_pred - 2 * var_pred, name="Lower confidence bound (2SD's)",
                                fill='tonexty', mode="lines", line=dict(color="lightgrey"))])
    fig.update_layout(title_text="Mean loss as function of p%", xaxis_title="p% of training samples",
                      yaxis_title="Loss")
    fig.show()
