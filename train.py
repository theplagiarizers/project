import pandas as pd

def preprocess_data(data):
    """_summary_

    Args:
        data (_type_): _description_
    
    Returns:
        _type_: _description_
    """
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