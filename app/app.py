from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
from custom_transformer import OutliersDetector

import __main__
__main__.OutliersDetector = OutliersDetector


model_package = joblib.load('./model.joblib')
model = model_package['model']
feature_names = model_package['feature_names']

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    features = pd.DataFrame(np.array([request.json]), columns=feature_names)
    return jsonify({'prediction': model.predict(features)[0]})

@app.route('/get_features', methods=['GET'])
def get_features():
    return jsonify(feature_names)


if __name__ == '__main__':
    app.run()