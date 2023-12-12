import mlflow
from mlflow import MlflowClient
import pandas as pd

def monitor_model():
    model_name = "random-forest-best"
    client = MlflowClient(tracking_uri="http://mlflow.rohaan.xyz:5000")
    model_metadata = client.get_latest_versions(model_name, stages=["None"])
    latest_model_version = model_metadata[0].version
    print(latest_model_version)
    
if __name__ == '__main__':
    monitor_model()
    