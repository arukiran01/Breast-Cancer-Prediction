from flask import Flask, render_template, request
import numpy as np
import pickle

# loading model
model = pickle.load(open('model1.pkl', 'rb'))

# flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index1.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get features from form input
    features = request.form['feature']
    features = features.split(',')
    
    try:
        # Convert features to numpy array and predict
        np_features = np.asarray(features, dtype=np.float32)
        pred = model.predict(np_features.reshape(1, -1))
        
        # Message based on prediction result
        message = ['Cancerous' if pred[0] == 1 else 'Not Cancerous']
        
    except Exception as e:
        message = [f"Error: {str(e)}"]
    
    return render_template('index1.html', message=message)

if __name__ == '__main__':
    app.run(debug=True)
