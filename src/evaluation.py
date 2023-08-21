import logging
from abc import ABC , abstractmethod
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score


class Evaluation(ABC):
    """Abstract class defining strategies for evaluation of our model."""
    @abstractmethod
    def calculate_scores(self,y_true: np.ndarray,y_pred: np.ndarray):
        """Calculates the scores for the model

        Args:
            y_train (np.ndarray): True labels
            y_pred (np.ndarray): Prediction labels
        Returns:
            None
        """
        pass

class MSE(Evaluation):
    """Mean Squared Error evaluation strategy"""
    def calculate_scores(self,y_true: np.ndarray,y_pred: np.ndarray):
        """Calculates the Mean Squared error for the model

        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Prediction labels
        Returns: 
            MSE: Float
        """
        try:
            logging.info("Calculating mean squared error")
            mse = mean_squared_error(y_true, y_pred)
            logging.info("MSE: %f" % mse)
            return mse
        except Exception as e:
            logging.error("Error calculating MSE: %s" % e)
            raise e
        
class R2(Evaluation):
    """R2 score evaluation strategy"""
    def calculate_scores(self,y_true: np.ndarray,y_pred: np.ndarray):
        """Calculates the R2 score for the model
        
        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Prediction labels
        Returns: 
            R2: Float
        """
        try:
            logging.info("Calculating R2 score")
            r2 = r2_score(y_true, y_pred)
            logging.info("R2: %f" % r2)
            return r2
        except Exception as e:
            logging.error("Error calculating R2 Score: %s" % e)
            raise e
        
class RMSE(Evaluation):
    """Root Mean Squared Error evaluation strategy"""
    def calculate_scores(self,y_true: np.ndarray,y_pred: np.ndarray):
        """Calculates the Root Mean Squared error for the model
        
        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Prediction labels
        Returns: 
            RMSE: Float
        """
        try:
            logging.info("Calculating root mean squared error")
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            logging.info("RMSE: %f" % rmse)
            return rmse
        except Exception as e:
            logging.error("Error calculating RMSE: %s" % e)
            raise e