from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
import numpy as np
from typing import Tuple
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from IMLearn.metrics import accuracy
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
    dataset = np.load(filename)
    X, y = dataset[:, :2], dataset[:, -1]
    return X, y


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
    for n, f in [("gaussian1", "/Users/aryehnailand/Desktop/HUJICSdegree/Sem5/IML/IML.HUJI/datasets/gaussian1.npy"),
                 ("gaussian2", "/Users/aryehnailand/Desktop/HUJICSdegree/Sem5/IML/IML.HUJI/datasets/gaussian2.npy")]:
        # Load dataset
        X, y = load_dataset(f)

        # Fit models and predict over training set
        lda = LDA().fit(X, y)
        lda_predict = lda.predict(X)
        lda_accuracy = accuracy(y, lda_predict)

        gnb = GaussianNaiveBayes().fit(X, y)
        gnb_predict = gnb.predict(X)
        gnb_accuracy = accuracy(y, gnb_predict)


        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
        fig = make_subplots(rows=1, cols=2, start_cell="bottom-left")

        # todo add markers colours and names
        fig.add_trace(go.Scatter(x=X[:,:1], y=X[:,-1]), row=1, col=1, color=gnb_predict)

        fig.add_trace(go.Scatter(x=X[:,:1], y=X[:,-1]), row=1, col=2, color=lda_predict)

        fig.show()



if __name__ == '__main__':
    np.random.seed(0)
    # run_perceptron()
    compare_gaussian_classifiers()
