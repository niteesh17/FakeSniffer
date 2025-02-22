#!flask/bin/python
import sys
import os

# Add the 'src' directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from flask import Flask, request, render_template
from flask_bootstrap import Bootstrap
import pickle

# Import the cleaning and prediction modules
from cleaning import process_text
from prediction import get_predictions

app = Flask(__name__)
Bootstrap(app)

# Define the path for the model and feature transformer
model_path = "models/lr_final_model.pkl"
transformer_path = "models/transformer.pkl"

# Load the model and feature transformer with pickle
loaded_model = pickle.load(open(model_path, 'rb'))
loaded_transformer = pickle.load(open(transformer_path, 'rb'))

@app.route('/')
def index():
    return render_template('index.html')
        
@app.route('/predict', methods=['POST'])
def predict():
    """
    Collect the input and predict the outcome
    Returns:
        Results.html with prediction
    """
    if request.method == 'POST':
        # Get input statement
        namequery = request.form['namequery']
        data = [namequery]
        # Get the clean data
        clean_data = process_text(str(data))
        test_features = loaded_transformer.transform([" ".join(clean_data)])
        my_prediction = get_predictions(loaded_model, test_features)
    return render_template('results.html', prediction=my_prediction, name=namequery)

if __name__ == '__main__':
    app.run(debug=True)
