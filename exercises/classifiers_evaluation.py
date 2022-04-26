import math

from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
import numpy as np
from typing import Tuple
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from IMLearn.metrics import accuracy
from plotly.subplots import make_subplots

from utils import decision_surface

pio.templates.default = "simple_white"


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    dataset = np.load(f"..//datasets//{filename}")
    X, y = dataset[:, :2], dataset[:, -1]
    return X, y


def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix
    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse
    cov: ndarray of shape (2,2)
        Covariance of Gaussian
    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = math.atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * np.pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines", marker_color="black")


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """
    for n, f in [("Linearly Separable",
                  "/Users/aryehnailand/Desktop/HUJICSdegree/Sem5/IML/IML.HUJI/datasets/linearly_separable.npy"),
                 ("Linearly Inseparable",
                  "/Users/aryehnailand/Desktop/HUJICSdegree/Sem5/IML/IML.HUJI/datasets/linearly_inseparable.npy")]:
        # Load dataset
        loss = []
        dataset = np.load(f)
        # Fit Perceptron and record loss in each fit iteration
        X, y = dataset[:, :2], dataset[:, -1]
        perceptron_classifier = Perceptron(callback=lambda p: loss.append(p.loss(X, y)))
        perceptron_classifier.fit(X, y)
        # Plot figure
        fig = px.line(x=range(len(loss)),
                      y=loss,
                      title=f'Perception Training loss per Iteration: {n} data',
                      labels=dict(x="Iterations", y="Training Loss"))
        fig.show()


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    # for n, f in [("gaussian1", "/Users/aryehnailand/Desktop/HUJICSdegree/Sem5/IML/IML.HUJI/datasets/gaussian1.npy"),
    #              ("gaussian2", "/Users/aryehnailand/Desktop/HUJICSdegree/Sem5/IML/IML.HUJI/datasets/gaussian2.npy")]:
    #     # Load dataset
    #     X, y = load_dataset(f)
    #
    #     # Fit models and predict over training set
    #     lda = LDA().fit(X, y)
    #     lda_predict = lda.predict(X)
    #     lda_accuracy = accuracy(y, lda_predict)
    #
    #     gnb = GaussianNaiveBayes().fit(X, y)
    #     gnb_predict = gnb.predict(X)
    #     gnb_accuracy = accuracy(y, gnb_predict)
    #
    #     # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
    #     # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
    #     fig = make_subplots(rows=1, cols=2, start_cell="bottom-left",
    #                         subplot_titles=(f'LDA Prediction - Accuracy: {lda_accuracy}',
    #                                         f'Gaussian Naive Bayes Prediction - Accuracy: {gnb_accuracy}')
    #                         )
    #
    #     # add legends
    #     fig.add_trace(
    #         go.Scatter(mode='markers', marker_symbol=y, x=X[:, 0].flatten(), y=X[:, 1].flatten(),
    #                    marker=dict(color=lda_predict, size=10, line=dict(color='Black', width=2))),
    #         row=1, col=1)
    #
    #     fig.add_trace(
    #         go.Scatter(mode='markers', x=X[:, 0].flatten(), y=X[:, 1].flatten(),
    #                    marker=dict(color=gnb_predict, symbol=y, size=10, line=dict(color='Black', width=2))),
    #         row=1, col=2)
    #
    #     # 1. gaussian centers:
    #
    #     # 2. ellipsis:
    #     fig.add_trace(get_ellipse(np.mean(lda.mu_, axis=0), lda.cov_), row=1, col=1)
    #     # todo check the cov calculation
    #     fig.add_trace(get_ellipse(np.mean(gnb.mu_, axis=0), np.diag(np.mean(gnb.var_, axis=0))), row=1, col=2)
    #
    #     fig.update_layout(title=f'Comparing LDA and Gaussian Naive Bayes on {n} dataset')
    #     fig.show()
    for f in ["gaussian1.npy", "gaussian2.npy"]:
        # Load dataset
        X, y = load_dataset(f)

        # Fit models and predict over training set
        lda = LDA()
        lda.fit(X, y)

        gnb = GaussianNaiveBayes()
        gnb.fit(X, y)

        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
        # Create subplots
        from IMLearn.metrics import accuracy

        models = [lda, gnb]
        models_names = ["LDA", "Gaussian Naive Bayes"]

        symbols = np.array(["circle", "x", 'triangle-up'])
        fig = make_subplots(rows=2, cols=3, subplot_titles=[rf"$\textbf{{{m}}}$" for m in models_names],
                            horizontal_spacing=0.05, vertical_spacing=.1)
        lims = np.array([X.min(axis=0), X.max(axis=0)]).T + np.array([-.4, .4])

        for i, m in enumerate(models):
            acc = np.round(accuracy(y, m._predict(X)), 3)
            fig.add_traces(
                [decision_surface(m._predict, lims[0], lims[1], showscale=False),
                 go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers", showlegend=False,
                            marker=dict(color=y, symbol=symbols[y.astype(int)],
                                        colorscale=['green', 'yellow', 'red'],
                                        line=dict(color="black", width=1)),
                            text=f"accuracy: {acc}",
                            textposition='middle center'),
                 go.Scatter(x=m.mu_[:, 0], y=m.mu_[:, 1], mode='markers',
                            marker=dict(color="black", symbol="x"),
                            showlegend=False),

                 # Add ellipses depicting the covariances of the fitted Gaussians
                 get_ellipse(m.mu_[0], m.cov_ if i == 0 else np.diag(m.vars_[0])),
                 get_ellipse(m.mu_[1], m.cov_ if i == 0 else np.diag(m.vars_[1])),
                 get_ellipse(m.mu_[2], m.cov_ if i == 0 else np.diag(m.vars_[2]))
                 ],
                rows=(i // 2) + 1,
                cols=(i % 2) + 1
            )
            fig.layout.annotations[i].update(text=f'{models_names[i]} accuracy: {acc}')

        fig.update_layout(title=f"Analysis of {f} dataset.", title_x=0.2, height=600, width=1500, )
        fig.show()


if __name__ == '__main__':
    np.random.seed(0)
    # run_perceptron()
    compare_gaussian_classifiers()
