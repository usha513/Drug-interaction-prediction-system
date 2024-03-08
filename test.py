import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/predict_drug', methods=['POST'])
def predict_drug():
    input_medical_condition = request.form.get('medical_condition')
    if not input_medical_condition:
        return jsonify({'error': 'No medical condition provided'}), 400

    # Load the dataset
    df_drugs = pd.read_csv('path_to/Drug.csv')

    # Logic to determine drugName and Dose_mg
    # This is a placeholder - replace with your actual logic or model
    filtered_drugs = df_drugs[df_drugs['medical_condition'] == input_medical_condition]
    if filtered_drugs.empty:
        return jsonify({'error': 'No drugs found for the given condition'}), 404
	
	dose=filter(df.Dose_mg<20 )
    # For simplicity, just returning the first match
    result = filtered_drugs.iloc[0][['drugName', 'dose']].to_dict()
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)