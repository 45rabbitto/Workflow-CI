import mlflow
import pandas as pd
from flask import Flask, request, jsonify
import json
import os

app = Flask(__name__)

model_path = "/app/model"
print(f"Loading model from {model_path}")
model = mlflow.pyfunc.load_model(model_path)
print("Model loaded successfully")

@app.route('/invocations', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if 'dataframe_records' in data:
            df = pd.DataFrame(data['dataframe_records'])
        else:
            df = pd.DataFrame(data)
        
        predictions = model.predict(df)
        return jsonify({'predictions': predictions.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/ping', methods=['GET'])
def ping():
    return jsonify({'status': 'alive'})

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
