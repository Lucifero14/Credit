from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)

# Load the saved model, scaler, and label encoders
model = joblib.load("xgboost_model_regularized_with_dti_1.pkl")
scaler = joblib.load("scaler_1.pkl")
label_encoders = joblib.load("label_encoders.pkl")

# Manually define expected feature names based on processing steps
expected_features = [
    'Gender', 'Age', 'Debt', 'Married', 'BankCustomer', 'Industry',
    'Ethnicity', 'YearsEmployed', 'PriorDefault', 'Employed', 'CreditScore',
    'DriversLicense', 'Citizen', 'LogIncome', 'LogDebt', 'LogCreditScore',
    'DebtToIncome', 'CreditToDebt', 'DebtToCredit'
]

print("Manually defined expected features:")
print(expected_features)

@app.route('/')
def home():
    return render_template('index_2.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect input data from form
        input_data = {key: request.form[key] for key in request.form}
        print("Raw Input Data:", input_data)

        # Convert numeric values
        numeric_features = [
            'Gender', 'Age', 'Debt', 'Married', 'BankCustomer', 'YearsEmployed', 
            'PriorDefault', 'Employed', 'CreditScore', 'DriversLicense', 'Income'
        ]
        for key in numeric_features:
            input_data[key] = float(input_data[key])

        # Encode categorical variables
        categorical_features = ['Industry', 'Ethnicity', 'Citizen']
        for col in categorical_features:
            input_data[col] = label_encoders[col].transform([input_data[col]])[0]

        # Apply log transformation
        input_data['LogIncome'] = np.log1p(input_data['Income'])
        input_data['LogDebt'] = np.log1p(input_data['Debt'])
        input_data['LogCreditScore'] = np.log1p(input_data['CreditScore'])

        # Compute feature engineering ratios
        input_data['DebtToIncome'] = (
            input_data['LogDebt'] / input_data['LogIncome'] if input_data['LogIncome'] != 0 else 0
        )
        input_data['CreditToDebt'] = (
            input_data['LogCreditScore'] / input_data['LogDebt'] if input_data['LogDebt'] != 0 else 0
        )
        input_data['DebtToCredit'] = (
            input_data['LogDebt'] / input_data['LogCreditScore'] if input_data['LogCreditScore'] != 0 else 0
        )

        # Remove 'Income' as we are using 'LogIncome'
        input_data.pop('Income', None)

        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])
        print("Features Before Scaling:", list(input_df.columns))

        # Ensure column order matches the model
        try:
            input_df = input_df[expected_features]
        except KeyError as e:
            print("Feature mismatch error:", e)
            print("Expected Features:", expected_features)
            print("Received Features:", list(input_df.columns))
            return f"Error: Feature mismatch detected. {str(e)}"

        print("Features After Reordering:", list(input_df.columns))

        # Scale the input data
        input_scaled = scaler.transform(input_df)

        # Make prediction
        probability = model.predict_proba(input_scaled)[0][1]
        threshold = 0.95
        prediction = "Approved" if probability > threshold else "Not Approved"

        return render_template('index_2.html', prediction=prediction)
    
    except Exception as e:
        print("Full error:", e)
        return f"Error: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
