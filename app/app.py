from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# Load the pre-trained model
with open('best_model/model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET'])
def predict():
    # Get the data from the form
    timestamp = request.args.get('timestamp')
    machine_id = request.args.get('machine_id')
    sensor_id = request.args.get('sensor_id')
    
    # From timeframe, extract hour, day of week and month
    timestamp = pd.to_datetime(timestamp)
    hour = timestamp.hour
    day_of_week = timestamp.dayofweek
    month = timestamp.month
    
    # Machine_ID_Machine_1	Machine_ID_Machine_2	Machine_ID_Machine_3	Machine_ID_Machine_4	Machine_ID_Machine_5	Sensor_ID_Sensor_1	Sensor_ID_Sensor_2	Sensor_ID_Sensor_3	Hour	DayOfWeek	Month

    # Create a DataFrame with the input values
    data = pd.DataFrame({
        'Machine_ID_Machine_1': [1 if machine_id == 'Machine_1' else 0],
        'Machine_ID_Machine_2': [1 if machine_id == 'Machine_2' else 0],
        'Machine_ID_Machine_3': [1 if machine_id == 'Machine_3' else 0],
        'Machine_ID_Machine_4': [1 if machine_id == 'Machine_4' else 0],
        'Machine_ID_Machine_5': [1 if machine_id == 'Machine_5' else 0],
        'Sensor_ID_Sensor_1': [1 if sensor_id == 'Sensor_1' else 0],
        'Sensor_ID_Sensor_2': [1 if sensor_id == 'Sensor_2' else 0],
        'Sensor_ID_Sensor_3': [1 if sensor_id == 'Sensor_3' else 0],
        'Hour': [hour],
        'DayOfWeek': [day_of_week],
        'Month': [month]
    })
    
    # Make the prediction using the pre-trained model
    prediction = model.predict(data)[0]
    
    # Render the result template with the prediction
    return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)