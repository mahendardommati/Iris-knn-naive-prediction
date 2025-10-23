# Iris-knn-naive-prediction
Flask-based web app for Iris Flower prediction using KNN.

# Iris Flower Prediction Web App

## Overview
This is a **Flask-based web application** that predicts the species of an Iris flower using user-provided features. The app supports **KNN** and **Naive Bayes** models. Users can input the flower's measurements and get predictions instantly.  

![Iris Flower](https://upload.wikimedia.org/wikipedia/commons/5/56/Iris_dataset_scatterplot.svg)

---

## Features
- Predict Iris flower species based on **Sepal Length, Sepal Width, Petal Length, Petal Width**
- Choose between **KNN** and **Naive Bayes** models
- Simple, user-friendly web interface
- Instant prediction display

---

## Technologies
- Python 3.x
- Flask
- scikit-learn
- HTML/CSS (Bootstrap optional)

---

## How to Run Locally
1. **Clone the repository**:
   ```bash
   git clone https://github.com/<mahendardommati>/Iris-KNN-Prediction.git
   cd Iris-KNN-Prediction

## Create a virtual environment:

bash
Copy code
python -m venv knn_env

## Activate the virtual environment:

Windows: knn_env\Scripts\activate

## Install dependencies:

bash

pip install -r requirements.txt

## Run the Flask app:

bash
Copy code
python app.py

## Open the web page in your browser:

cpp
Copy code
http://127.0.0.1:5000




## Code Snippets

Flask app (app.py):

from flask import Flask, render_template, request

import numpy as np

import pickle

app = Flask(__name__)

knn_model = pickle.load(open('knn_model.pkl', 'rb'))

nb_model = pickle.load(open('naive_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])

def predict():

    sl = float(request.form['sepal_length'])
    sw = float(request.form['sepal_width'])
    pl = float(request.form['petal_length'])
    pw = float(request.form['petal_width'])
    model_type = request.form['model']
    data = np.array([[sl, sw, pl, pw]])

    if model_type == 'KNN':
        prediction = knn_model.predict(data)[0]
    else:
        prediction = nb_model.predict(data)[0]

    return render_template('index.html', prediction_text=f'Predicted Iris Species: {prediction}')

## HTML Form (index.html):

<form action="/predict" method="post">

    <input type="text" name="sepal_length" placeholder="Sepal Length" required>
    <input type="text" name="sepal_width" placeholder="Sepal Width" required>
    <input type="text" name="petal_length" placeholder="Petal Length" required>
    <input type="text" name="petal_width" placeholder="Petal Width" required>
    <select name="model">
        <option value="KNN">KNN</option>
        <option value="Naive">Naive Bayes</option>
    </select>
    <button type="submit">Predict</button>
</form>
