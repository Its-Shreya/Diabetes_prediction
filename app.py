# app.py
from flask import Flask, render_template, request
import joblib  # For loading the saved model
import numpy as np

app = Flask(__name__, template_folder='templates')


# Load the saved diabetes prediction model
model = joblib.load('diabetes_model.pkl')  # Assuming you saved your model using joblib

@app.route('/result')
def result():
    return render_template('result.html')

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Get the input values from the form
        preg = float(request.form['pregnancies'])
        glucose = float(request.form['glucose'])
        bp = float(request.form['blood_pressure'])
        skin_thickness = float(request.form['skin_thickness'])
        insulin = float(request.form['insulin'])
        bmi = float(request.form['bmi'])
        dpf = float(request.form['diabetes_pedigree_function'])
        age = float(request.form['age'])

        # Create a numpy array with the input values
        input_data = np.array([[preg, glucose, bp, skin_thickness, insulin, bmi, dpf, age]])

        # Use the loaded model to make predictions
        prediction = model.predict(input_data)[0]

        # Display the prediction
        if prediction == 1:
            result = "Diabetic"
        else:
            result = "Not Diabetic"

        return render_template('result.html', result=result)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
