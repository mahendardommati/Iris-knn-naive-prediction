import flask
from flask import Flask, render_template, request
import numpy as np
import pickle

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

app = Flask(__name__)

# Load saved models
knn_model = pickle.load(open('knn_model.pkl', 'rb'))
nb_model = pickle.load(open('naive_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Collect feature values from form
    try:
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

        return render_template(
            'index.html',
            prediction_text=f'Predicted Iris Species: {prediction}',
            selected_model=model_type
        )

    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {e}')

if __name__ == '__main__':
    app.run(debug=True)