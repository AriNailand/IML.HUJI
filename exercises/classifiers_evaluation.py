from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
import numpy as np
from typing import Tuple
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from plotly.subplots import make_subplots
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
    raise NotImplementedError()


def perceptron_iteration_callback(perceptron: Perceptron, curr_sample: np.ndarray, curr_response: int):
    # todo
    pass


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """
    for n, f in [("Linearly Separable", "/Users/aryehnailand/Desktop/HUJICSdegree/Sem5/IML/IML.HUJI/datasets/linearly_separable.npy"),
                 ("Linearly Inseparable", "/Users/aryehnailand/Desktop/HUJICSdegree/Sem5/IML/IML.HUJI/datasets/linearly_inseparable.npy")]:
        # Load dataset
        loss = []
        data_set = np.load(f)
        # Fit Perceptron and record loss in each fit iteration
        X, y = data_set[:, :2], data_set[:, -1]
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
    for f in ["gaussian1.npy", "gaussian2.npy"]:
        # Load dataset
        raise NotImplementedError()

        # Fit models and predict over training set
        raise NotImplementedError()

        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
        from IMLearn.metrics import accuracy
        raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    run_perceptron()
    compare_gaussian_classifiers()
