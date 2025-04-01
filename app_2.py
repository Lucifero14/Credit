from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load the trained model
rf_model = joblib.load("random_forest_model.pkl")

# Load processed dataset to get column format
processed_df = pd.read_csv("processed_dataset.csv")
feature_columns = processed_df.columns[:-1]  # Exclude the target column

@app.route('/')
def home():
    return render_template("index_2.html")  # Load the HTML page

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        input_data = {col: float(request.form[col]) for col in feature_columns}

        # Convert input to DataFrame
        new_data = pd.DataFrame([input_data])

        # Make prediction
        prediction = rf_model.predict(new_data)[0]
        result = "Approved" if prediction == 1 else "Not Approved"

        return f"<h2>Prediction: {result}</h2>"

    except Exception as e:
        return f"<h2>Error: {str(e)}</h2>"

if __name__ == '__main__':
    app.run(debug=True, port=5001)
