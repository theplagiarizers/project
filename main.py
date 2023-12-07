import pandas as pd
import numpy as np

def load_data():
    """_summary_
    This function is used to load the data from the csv file.
    """
    # Load the data from the csv file
    data = pd.read_csv('data/dummy_sensor_data.csv')
    
    return data
def preprocess_data(data):
    # removing the null values
    data = data.dropna()
    # removing the duplicates
    data = data.drop_duplicates()
    
    # Hot Encoding the categorical data
    data = pd.get_dummies(data, columns=['Machine_ID', 'Sensor_ID'])
    # Feature Extraction
    data['Timestamp'] = pd.to_datetime(data['Timestamp'])
    data['Hour'] = data['Timestamp'].dt.hour
    data['DayOfWeek'] = data['Timestamp'].dt.dayofweek
    data['Month'] = data['Timestamp'].dt.month
    return data
def start():
    """_summary_
    This function is the starting point of the program.
    """
    #Loading the data
    initial_data = load_data()
    print("Data Loaded Successfully")
    print(initial_data.head())
    
    #Preprocessing the data
    print("Preprocessing the data")
    processed_data = preprocess_data(initial_data)
    print(processed_data.head())
    processed_data.to_csv('data/processed_data.csv', index=False)
    

if __name__ == '__main__':
    start()
    