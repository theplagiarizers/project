# Use the official Python base image
FROM python:3.9-alpine

# Set the working directory in the container
WORKDIR /app

# Install required dependencies from requirements.txt
RUN pip install --trusted-host pypi.python.org -r requirements.txt

# Copy the model code and requirements
COPY requirements.txt .
COPY app.py .

# Copy the trained model from the MLflow Model Registry
RUN mlflow models pull -m <model_uri> --no-conda

# Expose the port that the Flask application will use
EXPOSE 5000

# Set the MLflow tracking URI
ENV MLFLOW_TRACKING_URI=<mlflow_tracking_uri>

# Set the environment variable for Flask
ENV FLASK_APP=app.py

# Run the Flask application
CMD ["flask", "run", "--host=0.0.0.0"]
```

In the above `Dockerfile`, replace `<model_uri>` with the URI of your registered model in the MLflow Model Registry, and replace `<mlflow_tracking_uri>` with the URI of your MLflow tracking server.