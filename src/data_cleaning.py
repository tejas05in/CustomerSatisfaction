import logging
from abc import ABC , abstractmethod
from typing import  Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class DataStrategy(ABC):
    """
    Abstract class defining strategy for handling data
    """
    @abstractmethod
    def handle_data(self,data: pd.DataFrame) -> Union[pd.DataFrame , pd.Series]:
        pass


class DataPreProcessStrategy(DataStrategy):
    """
    Strategy for preprocessing data
    """
    def handle_data(self,data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess Data
        :param data: data for preprocessing
        :return: preprocessed data
        """

        try:
            data = data.drop(
                [
                    "order_approved_at",
                    "order_delivered_carrier_date",
                    "order_delivered_customer_date",
                    "order_estimated_delivery_date",
                    "order_purchase_timestamp",
                    "customer_zip_code_prefix",
                    "order_item_id"
                ],
                axis=1)

            data["product_weight_g"].fillna(data["product_weight_g"].median(),inplace=True)
            data["product_length_cm"].fillna(data["product_length_cm"].median(), inplace=True)
            data["product_height_cm"].fillna(data["product_height_cm"].median(), inplace=True)
            data["product_width_cm"].fillna(data["product_width_cm"].median(), inplace=True)
            data["review_comment_message"].fillna("No review", inplace=True)

            data = data.select_dtypes(include=[np.number])
            return data

        except Exception as e:
            logging.error(f"Error in preprocessing data: {e}")
            raise e


class DataDivideStrategy(DataStrategy):
    """
    split the data into train and test
    """
    def handle_data(self,data: pd.DataFrame) -> Union[pd.DataFrame , pd.Series]:
        """
        Divide the data into train and test
        :param data: data for dividing
        :return: pandas dataframe or series with train and test sets
        """
        try:
            X = data.drop(["review_score"],axis=1)
            y = data["review_score"]
            X_train , X_test , y_train , y_test = train_test_split(X,y,test_size=.2, random_state=42)
            return X_train , X_test , y_train , y_test
        except Exception as e:
            logging.error(f"Error in dividing the data: {e}")
            raise e

class DataCleaning:
    """
    class for cleaning data which processes the data and divdes it into
    training and testing set
    """
    def __init__(self, data: pd.DataFrame, strategy:DataStrategy):
        self.data = data
        self.strategy = strategy

    def handle_data(self) -> Union[pd.DataFrame , pd.Series]:
        try:
            return self.strategy.handle_data(self.data)
        except Exception as e:
            logging.error(f"Error in handling data: {e}")
            raise e

