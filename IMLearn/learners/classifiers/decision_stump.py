from __future__ import annotations
from typing import Tuple, NoReturn
from IMLearn.metrics import misclassification_error
from IMLearn.base import BaseEstimator
import numpy as np
from itertools import product


class DecisionStump(BaseEstimator):
    """
    A decision stump classifier for {-1,1} labels according to the CART algorithm

    Attributes
    ----------
    self.threshold_ : float
        The threshold by which the data is split

    self.j_ : int
        The index of the feature by which to split the data

    self.sign_: int
        The label to predict for samples where the value of the j'th feature is about the threshold
    """
    def __init__(self) -> DecisionStump:
        """
        Instantiate a Decision stump classifier
        """
        super().__init__()
        self.threshold_, self.j_, self.sign_ = None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a decision stump to the given data

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        d = X.shape[1]
        min_loss, self.j_, self.threshold_, self.sign_ = np.inf, 0, 0, 0
        for sign, j in product([-1, 1], range(d)):
            threshold, loss = self._find_threshold(X[:, j], y, sign)
            if loss < min_loss:
                self.j_ = j
                self.threshold_ = threshold
                self.sign_ = sign
                min_loss = loss

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples

        Notes
        -----
        Feature values strictly below threshold are predicted as `-sign` whereas values which equal
        to or above the threshold are predicted as `sign`
        """
        y_pred = np.where(X[:, self.j_] < self.threshold_, -self.sign_, self.sign_)
        return y_pred

    @staticmethod
    def _find_threshold(values: np.ndarray, labels: np.ndarray, sign: int) -> Tuple[float, float]:
        """
        Given a feature vector and labels, find a threshold by which to perform a split
        The threshold is found according to the value minimizing the misclassification
        error along this feature

        Parameters
        ----------
        values: ndarray of shape (n_samples,)
            A feature vector to find a splitting threshold for

        labels: ndarray of shape (n_samples,)
            The labels to compare against

        sign: int
            Predicted label assigned to values equal to or above threshold

        Returns
        -------
        thr: float
            Threshold by which to perform split

        thr_err: float between 0 and 1
            Misclassificaiton error of returned threshold

        Notes
        -----
        For every tested threshold, values strictly below threshold are predicted as `-sign` whereas values
        which equal to or above the threshold are predicted as `sign`
        """
        # sort the values
        sort_idx = np.argsort(values)
        values, labels = values[sort_idx], labels[sort_idx]
        # if we take everything to be sign -> start from left
        min_threshold_loss = np.sum(np.abs(labels[np.sign(labels) != sign]))
        # need -inf + inf for edge cases and we take in between each two values
        in_between_thresholds = 0.5 * (values[1:]+values[:-1])
        possible_thresholds = np.hstack(([[-np.inf], in_between_thresholds, [np.inf]]))
        # count left to right with cum sum
        cumulative_losses = np.cumsum(labels * sign) + min_threshold_loss
        losses = np.hstack((min_threshold_loss, cumulative_losses))
        # take index with the minimum loss
        min_loss_idx = np.argmin(losses)
        return possible_thresholds[min_loss_idx], losses[min_loss_idx]

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        return misclassification_error(self.predict(X), y)
