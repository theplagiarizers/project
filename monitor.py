import mlflow
from mlflow import MlflowClient, pyfunc
import subprocess
import pandas as pd
from train import preprocess_data
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score 

def fetch_model():
    model_name = "random-forest-best"
    client = MlflowClient(tracking_uri="http://mlflow.rohaan.xyz:5000")
    mlflow.set_tracking_uri("http://mlflow.rohaan.xyz:5000")
    model_metadata = client.get_latest_versions(model_name, stages=["None"])
    latest_model_version = model_metadata[0].version
    latest_model_name = model_metadata[0].name
    latest_model_id = model_metadata[0].run_id
    model_uri = client.get_model_version_download_uri(latest_model_name, latest_model_version)
    # Create app/best_model directory to store the model
    subprocess.call(['mkdir', '-p', 'app/best_model'])
    client.download_artifacts(latest_model_id, "model", 'app/best_model')
    print(latest_model_version)
    
    # Load the model from mlflow as a PyFuncModel.
    model = mlflow.pyfunc.load_model(model_uri)
    return model




def fetch_data(data_file_path = 'data/dummy_sensor_data.csv'): 
    data = pd.read_csv(data_file_path)
    data = preprocess_data(data)
    return data

def save_metrics(mae, r2, mse):
    print("Saving metrics for future analysis...")
    # Generate a pd dataframe to store the metrics
    metrics = pd.DataFrame(columns=['MAE', 'R2', 'MSE'])
    # Add the metrics to the dataframe
    metrics.loc[0] = [mae, r2, mse]
    # if metrics.csv does not exist, create it, else append the metrics to it
    try:
        existing_metrics = pd.read_csv('metrics.csv')
        updated_metrics = pd.concat([existing_metrics, metrics], ignore_index=True)
        updated_metrics.to_csv('metrics.csv', index=False)
    except FileNotFoundError:
        metrics.to_csv('metrics.csv', index=False)
    
    
def monitor_model():
    #Get the latest model from mlflow
    model = fetch_model()
    #Get the latest data from the data file
    data = fetch_data()
    
    #Evaluate model performance on the latest data
    predictions = model.predict(data)
    #Calculate error metrics
    mae = mean_absolute_error(data['Reading'], predictions)
    r2 = r2_score(data['Reading'], predictions)
    mse = mean_squared_error(data['Reading'], predictions)
    print("Model error metrics: MAE:[{}], R2:[{}], MSE:[{}]".format(mae, r2, mse))
    save_metrics(mae, r2, mse)
    
    # Compare error metrics with threshold
    if mae > 16 or mse > 500:
        print("Model performance degraded! Retraining model.")
        subprocess.call(['python', 'main.py'])
    
    
    
    
if __name__ == '__main__':
    monitor_model()
    