import logging
import pandas as pd
from zenml import step
from sklearn.base import RegressorMixin
from src.evaluation import MSE , RMSE ,R2
from typing import Tuple
from typing_extensions import Annotated

@step
def evaluate_model(model: RegressorMixin,
                   X_test: pd.DataFrame,
                   y_test: pd.DataFrame) -> Tuple[
                       Annotated[float,"r2"],
                       Annotated[float,"rmse"]]:

    """
    Evaluates the model on the ingested data

    Args:
        df (pd.DataFrame): ingested data
    """
    try:
        prediction = model.predict(X_test)
        mse_class = MSE()
        mse = mse_class.calculate_scores(y_test, prediction)

        r2_class = R2()
        r2 = r2_class.calculate_scores(y_test, prediction)

        rmse_class = RMSE()
        rmse = rmse_class.calculate_scores(y_test, prediction)
        logging.info(f"RMSE: {rmse} , MSE: {mse} R2 score: {r2}")
        return r2,rmse
    except Exception as e:
        logging.error(f"Error in evaluation of model: {e}")
        raise e