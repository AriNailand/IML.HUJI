from __future__ import annotations
from typing import NoReturn

from IMLearn.base import BaseEstimator
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA, LinearDiscriminantAnalysis as LDA
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB, CategoricalNB

class AgodaCancellationEstimator(BaseEstimator):
    """
    An estimator for solving the Agoda Cancellation challenge
    """

    def __init__(self) -> AgodaCancellationEstimator:
        """
        Instantiate an estimator for solving the Agoda Cancellation challenge

        Parameters
        ----------


        Attributes
        ----------

        """
        super().__init__()
        # self.adaboost = AdaBoostClassifier(n_estimators=10,
        #                                    learning_rate=1,
        #                                    base_estimator=BaggingClassifier(base_estimator=MultinomialNB(),
        #                                                                     max_samples=0.2),
        #                                    )

        # self.adaboost = AdaBoostClassifier(n_estimators=10,
        #                                     learning_rate=1,
        #                                     base_estimator=MultinomialNB())

        # self.adaboost = AdaBoostClassifier()
        self.adaboost = RandomForestClassifier()


    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit an estimator for given samples

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Notes
        -----

        """
        # self.logistic_regression.fit(X, y)
        self.adaboost.fit(X, y)

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
        """
        # THRESH = 0.47
        #
        # threshold_taker = lambda x: 1 if x > THRESH else 0
        #
        # vfunc = np.vectorize(threshold_taker)
        #
        # return vfunc(self.adaboost.predict_proba(X).T[1])

        return self.adaboost.predict(X)

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under loss function
        """
        pass
