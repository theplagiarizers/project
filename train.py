import pandas as pd
from mlflow import MlflowClient
import mlflow.sklearn, mlflow
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Connecting to the MLflow server
client = MlflowClient(tracking_uri="http://localhost:8080")

random_forest_experiment = mlflow.set_experiment("Random Forest")
run_name = "Random Forest Run 1"
# Define an artifact path that the model will be saved to.
artifact_path = "rf_model"

# Load the preprocessed dataset
data = pd.read_csv('data/processed_data.csv')

# Split the dataset into features (X) and target variable (y)
X = data.drop(['Timestamp', 'Reading'], axis=1)
y = data['Reading']

print(X.head())
print(y.head())

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

params = {"n_estimators": 100, "random_state": 42, "max_depth": 5}

# Initialize the Random Forest regressor
rf = RandomForestRegressor(**params)

# Train the model
rf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf.predict(X_test)

# Evaluate the model using mean squared error
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"Mean Squared Error: {mse}")

# Assemble the metrics we're going to write into a collection
metrics = {
    "mse": mse,
    "mae": mae,
    "r2": r2,
    "rmse": rmse
}

# Initiate a run, setting the `run_name` parameter
with mlflow.start_run(run_name=run_name) as run:
    # Log the parameters used for the model fit
    mlflow.log_params(params)

    # Log the error metrics that were calculated during validation
    mlflow.log_metrics(metrics)

    # Log an instance of the trained model for later use
    mlflow.sklearn.log_model(sk_model=rf, input_example=X_test, artifact_path=artifact_path)