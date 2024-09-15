from flask import Flask, render_template, request
import joblib
import pandas as pd
import os
app = Flask(__name__, template_folder='app/templates')
model = joblib.load('model/model.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    age = float(request.form['age'])
    weight = float(request.form['weight'])
    height = float(request.form['height'])
    blood_pressure = float(request.form['blood_pressure'])
    cholesterol = float(request.form['cholesterol'])

    features = pd.DataFrame([[age, weight, height, blood_pressure, cholesterol]], 
                            columns=['age', 'weight', 'height', 'blood_pressure', 'cholesterol'])
    prediction = model.predict(features)

    prediction_text = 'High Risk' if prediction[0] == 1 else 'Low Risk'
    return render_template('index.html', prediction_text=prediction_text)

if __name__ == '__main__':
    app.run(debug=True)
