from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import datetime
import os
import sys

# Import prediction functions from predictt.py
from predictt import predict_f1_race_results, format_f1_results

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/api/predict', methods=['POST'])
def predict_race():
    """API endpoint to predict F1 race results"""
    try:
        # Get data from request
        data = request.json
        circuit_name = data.get('circuit')
        race_date = data.get('raceDate')
        driver_data = data.get('drivers', [])
        
        # Convert to the format expected by the predictt.py functions
        race_info = []
        for driver in driver_data:
            race_info.append({
                'driver_name': driver['name'],
                'constructor_name': driver['team'],
                'start_position': driver['startPosition'],
                'circuit_name': circuit_name,
                'race_date': race_date
            })
        
        # Run the prediction
        results_df = predict_f1_race_results(race_info)
        
        # Convert DataFrame to list of dictionaries for JSON response
        results_list = []
        for _, row in results_df.iterrows():
            results_list.append({
                'position': int(row['Position']),
                'positionOrdinal': row['Position (Ordinal)'],
                'driver': row['Driver'],
                'constructor': row['Constructor'],
                'startPosition': int(row['Starting Position']),
                'startPositionOrdinal': row['Starting (Ordinal)'],
                'positionChange': int(row['Positions Gained/Lost']),
                'rawPrediction': float(row['Raw Model Prediction'])
            })
        
        return jsonify({
            'success': True,
            'results': results_list
        })
        
    except Exception as e:
        print(f"Error in prediction: {str(e)}", file=sys.stderr)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    # Ensure the model is available
    model_path = 'f1_race_model.keras'
    if not os.path.exists(model_path):
        print("No existing F1 model found. Running initial training...")
        from predictt import train_keras_model
        train_keras_model()
    
    # Run the Flask app
    app.run(debug=True, port=5000)
