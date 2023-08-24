import os
from pipelines.training_pipeline import train_pipeline
from zenml.client import Client

if __name__ == "__main__":
    # Run the pipeline
    print(
        Client().active_stack.experiment_tracker.get_tracking_uri())
    # The above gives the file path for mlflow ui backend-store-uri
    train_pipeline(data_path="D:\Python\MLProjectsPW\CustomerSatisfaction\data\olist_customers_dataset.csv")
    # Run MLflow to visualize the model performance
    os.system(
        "mlflow ui --backend-store-uri file:C:/Users/tejas/AppData/Roaming/zenml/local_stores/f665c0cb-dda5-4efd-b2f0-cde39c8545be/mlruns")
