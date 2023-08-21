import logging

import pandas as pd
from zenml import step


class IngestData:
    """Ingesting the data from the data path
    """
    def __init__(self,data_path: str):
        """initiating the data ingestion

        Args:
            data_path (str): path to the data
        """
        self.data_path = data_path

    def run(self):
        """ingesting data from data path

        Returns:
            _type_: pd.DataFrame
        """
        logging.info(f"Ingesting data from  {self.data_path}")
        return pd.read_csv(self.data_path)

@step
def ingest_df(data_path:str) -> pd.DataFrame:
    """
    Ingesting the data from the data path.

    Args:
        data_path: path to the data

    Returns:
        pd.DataFrame: The dataframe of the ingested data
    """
    try:
        ingest_data = IngestData(data_path)
        df = ingest_data.run()
        return df
    except Exception as e:
        logging.error(f"Error while ingesting the data: {e}")
        raise e
