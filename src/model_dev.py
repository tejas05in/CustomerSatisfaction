import logging
from abc import ABC, abstractmethod
from sklearn.linear_model import LinearRegression


class Model(ABC):
    """Abstract class for all models

    Args:
        ABC (_type_): _Abstract class for all models
    """

    @abstractmethod
    def train(self, X_train, y_train):
        """Train a model

        Args:
            X_train (_type_): Training data
            y_train (_type_): Training labels
        returns:
            None
        """
        pass


class LinearRegressionModel(Model):
    """Linear regression model
    """

    def train(self, X_train, y_train, **kwargs):
        """Train a model

        Args:
            X_train (_type_): Training data
            y_train (_type_): Training labels
        Returns:
            Model object
        """
        try:
            reg = LinearRegression(**kwargs)
            reg.fit(X_train, y_train)
            logging.info("Model training completed successfully")
            return reg
        except Exception as e:
            logging.error("Error in training model: {}".format(e))
            raise e
