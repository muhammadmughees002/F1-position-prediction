from flask import Flask, render_template, request
import joblib
import numpy as np
import xgboost

app = Flask(__name__)
model = joblib.load('xgboost_position_model.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        try:
            features = [
                float(request.form['grid']),
                int(request.form['driverId']),
                int(request.form['constructorId']),
                int(request.form['statusId']),
                int(request.form['laps']),
                float(request.form['milliseconds']),
                int(request.form['fastestLap']),
                float(request.form['rank']),
                float(request.form['fastestLapTime']),
                float(request.form['fastestLapSpeed'])
            ]
            prediction = int(model.predict([features])[0])
        except Exception as e:
            prediction = f"Error: {e}"
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
