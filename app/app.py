from flask import Flask, render_template, request
import joblib
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd

app = Flask(__name__, template_folder='/Users/mannormal/Desktop/app')


# DL
dl_model1 = tf.keras.models.load_model('/Users/mannormal/Downloads/Edisease1_model5.h5')
dl_model2 = keras.models.load_model('/Users/mannormal/Downloads/Edisease2_model5.h5')
dl_model3 = keras.models.load_model('/Users/mannormal/Downloads/Edisease3_model5.h5')
dl_model4 = keras.models.load_model('/Users/mannormal/Downloads/Edisease4_model5.h5')

dl_model1.make_predict_function()
dl_model2.make_predict_function()
dl_model3.make_predict_function()
dl_model4.make_predict_function()


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the user-selected model type.
    model_type = request.form['model']

    # Get the user input feature values.
    input_values = [
        float(request.form['Alcohol']),
        float(request.form['Moisture']),
        float(request.form['Vitamin B2']),
        float(request.form['Grams']),
        float(request.form['0ctadecatrienoic'])
    ]

    # Construct feature vector
    
    
    # Choose the model based on user selection
    if model_type == 'Angina':
        model = dl_model1
    elif model_type == 'Overwhight':
        model = dl_model2
    elif model_type == 'Diabetes':
        model = dl_model3
    elif model_type == 'High blood pressure':
        model = dl_model4
    else:
        return "Invalid model selection!"
        
    # 读取CSV文件
    indices_to_replace = [24, 74, 77, 26, 27]
    df = pd.read_csv('/Users/mannormal/Desktop/app/DefultValue.csv')
    for i, index in enumerate(indices_to_replace):
        df.iloc[index] = input_values[i]

    input_array = df.to_numpy()
    input_array = input_array.reshape(1, 166)
    prediction = model.predict(input_array)

    # Convert prediction to disease label

    if prediction == 0:
        disease = 'you are healthy'
    else:
        if model_type == 'Angina':
            disease = "We are 95.63% confident that you may have Angina."
        elif model_type == 'Overwhight':
            disease = "With a likelihood of 60.27%, you may be Overweight."
        elif model_type == 'Diabetes':
            disease = "There is a 78.98% chance that you have Diabetes."
        elif model_type == 'High blood pressure':
            disease = "Your chances of having High Blood Pressure are at 53.59%."
        else:
            return "Unexpected errors!"

    return render_template('result.html', prediction=disease)


if __name__ == '__main__':
    app.run(debug=True)
