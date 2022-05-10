import numpy as np
from typing import Tuple
from IMLearn.metalearners.adaboost import AdaBoost
from IMLearn.learners.classifiers import DecisionStump
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from IMLearn.metrics import misclassification_error

COLOR_SCALE = ["yellow", "green"]


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, noise)

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    training_error, test_error, learners_arr = [], [], np.arange(1, n_learners + 1)
    adaboost = AdaBoost(DecisionStump, n_learners).fit(train_X, train_y)
    for T in learners_arr:
        training_error.append(adaboost.partial_loss(train_X, train_y, T))
        test_error.append(adaboost.partial_loss(test_X, test_y, T))
    fig = go.Figure([go.Scatter(x=learners_arr, y=training_error, mode='lines', name=r'$Training-Error$'),
                     go.Scatter(x=learners_arr, y=test_error, mode='lines', name=r'$Test-Error$')])
    fig.update_xaxes(title_text="Number of Learners")
    fig.update_yaxes(title_text="Error Value")
    fig.update_layout(title_text=rf"$\textbf{{Adaboost Error Per Number of Learners}}$")
    fig.show()

    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    # todo why is first boundary missing
    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])
    fig = make_subplots(rows=2, cols=2, subplot_titles=[rf"$\textbf{{{t}}}$" for t in T],
                        horizontal_spacing=0.05, vertical_spacing=.1)
    test_set_data_points_scattered = go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers", showlegend=False,
                                                marker=dict(color=test_y, symbol=class_symbols[test_y.astype(int)],
                                                            colorscale=COLOR_SCALE,
                                                            line=dict(color="black", width=1)))
    for i, t in enumerate(T):
        fig.add_traces(
            [decision_surface(lambda X: adaboost.partial_predict(X, t), lims[0], lims[1], showscale=False,colorscale=COLOR_SCALE),
             test_set_data_points_scattered],
            rows=(i // 2) + 1,
            cols=(i % 2) + 1
        )
        acc = np.round(1 - misclassification_error(test_y, adaboost.partial_predict(test_X, t)), 3)
        fig.layout.annotations[i].update(text=f'{t} Iterations, accuracy: {acc}')
    fig.update_layout(width=700, height=900,
                      title=rf"$\textbf{{AdaBoost Decision Boundary based on number of Learners: noise={noise}}}$",
                      margin=dict(t=100))
    fig.update_xaxes(matches='x', range=[-1, 1], constrain="domain")
    fig.update_yaxes(matches='y', range=[-1, 1], constrain="domain", scaleanchor="x", scaleratio=1)
    fig.show()

    # Question 3: Decision surface of best performing ensemble
    best_ensemble_idx = np.argmin(test_error)
    acc = np.round(1 - test_error[best_ensemble_idx], 3)
    best_ensemble_t = best_ensemble_idx + 1
    fig = go.Figure(
        [decision_surface(lambda x: adaboost.partial_predict(x, best_ensemble_t), lims[0], lims[1], showscale=False, colorscale=COLOR_SCALE),
         test_set_data_points_scattered])
    # todo check these
    fig.update_xaxes(matches='x', range=[-1, 1], constrain="domain")
    fig.update_yaxes(matches='y', range=[-1, 1], constrain="domain", scaleanchor="x", scaleratio=1)
    fig.update_layout(title_text=f"Best Ensemble Size: {best_ensemble_t}, Accuracy: {acc}")
    fig.show()

    # Question 4: Decision surface with weighted samples
    size_factor = 50 if noise == 0 else 5
    training_set_scatter = go.Scatter(
        x=train_X[:, 0],
        y=train_X[:, 1],
        mode="markers",
        showlegend=False,
        marker=dict(color=(train_y == 1).astype(int),
                    size=size_factor * adaboost.D_ / np.max(adaboost.D_),
                    symbol=class_symbols[train_y.astype(int)],
                    colorscale=COLOR_SCALE,
                    line=dict(color="black", width=1))
    )
    fig = go.Figure([decision_surface(adaboost.predict, lims[0], lims[1], showscale=False, colorscale=COLOR_SCALE), training_set_scatter])
    fig.update_xaxes(range=[-1, 1], constrain="domain")
    fig.update_yaxes(range=[-1, 1], constrain="domain", scaleanchor="x", scaleratio=1)
    fig.update_layout(dict1=dict(width=600,
                                 height=600,
                                 title=rf"$\textbf{{Adaboost Training Set sized by Last Iteration Weights, Noise:{noise}}}$"))

    fig.show()


if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(0)
    fit_and_evaluate_adaboost(noise=0.4)
