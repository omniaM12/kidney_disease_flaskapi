import numpy as np
import pandas as pd
import seaborn as sns
from flask import Flask, request, jsonify, render_template
import joblib


app = Flask(__name__)
processor=joblib.load("C:/Users/GENERAL/.vscode/extensions/sourcery.sourcery-1.33.0-win32-x64/processor.pkl")
clf = joblib.load("C:/Users/GENERAL/.vscode/extensions/sourcery.sourcery-1.33.0-win32-x64/kidney_rf.pkl")
@app.route('/')
def home():
    return render_template("kidindex.html")

@app.route('/predict', methods=['POST'])
def predict():
    features= [float(x) for x in request.form.values()]
    features_array = np.array([features]) 
    column_names = ['al', 'su', 'bgr', 'bu','sc', 'pot', 'wc','rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']
    df = pd.DataFrame(features_array, columns=column_names)
    processed_features = processor.transform(df)
    prediction = clf.predict(processed_features)[0]
    predictions = 'Positive' if prediction == 1 else 'Negative'
    return render_template("kidindex.html", prediction_text= f'The case is: {predictions}')

if __name__ == '__main__':
    app.run(debug=True)