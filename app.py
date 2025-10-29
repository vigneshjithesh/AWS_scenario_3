from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load model
model = joblib.load('model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        feature1 = float(request.form['feature1'])
        feature2 = float(request.form['feature2'])
        prediction = model.predict([[feature1, feature2]])[0]
        return render_template('index.html', prediction_text=f"Predicted Price: ${prediction:.2f}")
    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000, debug=True)
