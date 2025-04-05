import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import datetime
import joblib
import os

def load_and_process_data():
    """Load and process data - placeholder for your actual data loading function"""
    # This is a simplified version - in practice, you would use your actual data loading function
    # You would need to adapt your existing data_utils.py functionality
    
    # For demonstration purposes, loading a sample dataset:
    # In reality, you should use your original load_and_process_data function
    try:
        # Try to use the original function if available
        from data_utils import load_and_process_data as original_load_function
        return original_load_function()
    except ImportError:
        print("Warning: Using sample data. Replace with your actual data loading function.")
        # Create a sample dataset for demonstration
        # In real use, replace this with your actual data loading code
        
        np.random.seed(42)
        n_samples = 1000
        
        # Generate sample data - focused on F1 with realistic positions 1-20
        data = {
            'result': np.random.randint(1, 21, n_samples),  # F1 has positions 1-20
            'start_position': np.random.randint(1, 21, n_samples),  # Grid positions 1-20
            'year': np.random.randint(2010, 2025, n_samples),
            'month': np.random.randint(1, 13, n_samples),
            'day': np.random.randint(1, 29, n_samples),
            'circuit': np.random.choice(['Monaco', 'Silverstone', 'Monza', 'Spa', 'Suzuka', 'Singapore', 
                                         'Melbourne', 'Bahrain', 'Shanghai', 'Barcelona'], n_samples),
            'name': np.random.choice(['Lewis Hamilton', 'Max Verstappen', 'Charles Leclerc', 'Lando Norris', 
                                      'Fernando Alonso', 'George Russell', 'Carlos Sainz', 'Sergio Perez'], n_samples),
            'constructor': np.random.choice(['Mercedes', 'Red Bull', 'Ferrari', 'McLaren', 'Aston Martin', 
                                             'Alpine', 'Williams', 'Haas', 'RB', 'Sauber'], n_samples)
        }
        
        return pd.DataFrame(data)

def prepare_data_for_keras():
    """Prepare data for Keras model"""
    # Load the data
    race_data = load_and_process_data()
    
    # Create a copy to avoid modifying the original
    data = race_data.copy()
    
    # Extract target variable
    y = data['result'].values
    
    # Extract features
    numerical_features = ['start_position', 'year', 'month', 'day']
    categorical_features = ['circuit', 'name', 'constructor']
    
    # Create scalers and encoders
    scalers = {}
    for feature in numerical_features:
        scaler = StandardScaler()
        data[feature] = scaler.fit_transform(data[feature].values.reshape(-1, 1))
        scalers[feature] = scaler
    
    # Create one-hot encoder for categorical features
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoder.fit(data[categorical_features])
    
    # Transform categorical features
    encoded_features = encoder.transform(data[categorical_features])
    
    # Get feature names
    encoded_feature_names = encoder.get_feature_names_out(categorical_features)
    
    # Create DataFrame with encoded features
    encoded_df = pd.DataFrame(encoded_features, columns=encoded_feature_names)
    
    # Combine numerical and encoded features
    X_data = pd.concat([data[numerical_features].reset_index(drop=True), 
                        encoded_df.reset_index(drop=True)], axis=1)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_data, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test, scalers, encoder, numerical_features, categorical_features

def build_keras_model(input_dim):
    """Build a Keras model for F1 position prediction"""
    model = keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=(input_dim,)),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1)  # Output layer for position
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def train_keras_model():
    """Train a Keras model and save it"""
    # Prepare data
    X_train, X_test, y_train, y_test, scalers, encoder, numerical_features, categorical_features = prepare_data_for_keras()
    
    # Build model
    model = build_keras_model(X_train.shape[1])
    
    # Train model
    print("Training model...")
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        callbacks=[
            keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.1, patience=5, min_lr=1e-6)
        ],
        verbose=1
    )
    
    # Evaluate model
    test_loss = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss: {test_loss[0]:.4f}, Test MAE: {test_loss[1]:.4f}")
    
    # Save model and preprocessing components
    model_path = 'f1_race_model.keras'
    model.save(model_path)
    
    # Save scalers and encoder using joblib
    joblib.dump(scalers, 'f1_scalers.joblib')
    joblib.dump(encoder, 'f1_encoder.joblib')
    joblib.dump(numerical_features, 'f1_numerical_features.joblib')
    joblib.dump(categorical_features, 'f1_categorical_features.joblib')
    
    print(f"Model saved to {model_path}")
    print("Preprocessing components saved successfully")
    
    return model, scalers, encoder, numerical_features, categorical_features

def predict_race_outcome(driver_name, constructor_name, circuit_name, start_position, race_date):
    """
    Predict race outcome using the trained Keras model
    
    Args:
        driver_name (str): Full name of the driver
        constructor_name (str): Name of the constructor/team
        circuit_name (str): Name of the circuit
        start_position (int): Starting grid position (1-20 for F1)
        race_date (str or datetime): Date of the race in 'YYYY-MM-DD' format
        
    Returns:
        float: Predicted finishing position
    """
    model_path = 'f1_race_model.keras'
    
    try:
        # Load the trained model
        model = keras.models.load_model(model_path)
        
        # Load scalers and encoder
        scalers = joblib.load('f1_scalers.joblib')
        encoder = joblib.load('f1_encoder.joblib')
        numerical_features = joblib.load('f1_numerical_features.joblib')
        categorical_features = joblib.load('f1_categorical_features.joblib')
    except (FileNotFoundError, OSError):
        print(f"Model files not found. Training a new model...")
        model, scalers, encoder, numerical_features, categorical_features = train_keras_model()
    
    # Parse date if it's a string
    if isinstance(race_date, str):
        race_date = pd.to_datetime(race_date)
    
    # Create input data
    input_data = pd.DataFrame({
        'start_position': [start_position],
        'year': [race_date.year],
        'month': [race_date.month],
        'day': [race_date.day],
        'circuit': [circuit_name],
        'name': [driver_name],
        'constructor': [constructor_name]
    })
    
    # Scale numerical features
    for feature in numerical_features:
        try:
            input_data[feature] = scalers[feature].transform(input_data[feature].values.reshape(-1, 1))
        except KeyError:
            # If the feature is not in scalers, use a default scaler
            print(f"Warning: No scaler found for {feature}. Using default standardization.")
            input_data[feature] = (input_data[feature] - input_data[feature].mean()) / input_data[feature].std()
    
    # Encode categorical features
    encoded_features = encoder.transform(input_data[categorical_features])
    encoded_df = pd.DataFrame(
        encoded_features, 
        columns=encoder.get_feature_names_out(categorical_features)
    )
    
    # Combine features for prediction
    X_pred = pd.concat([input_data[numerical_features].reset_index(drop=True), 
                      encoded_df.reset_index(drop=True)], axis=1)
    
    # Make prediction
    prediction = model.predict(X_pred)[0][0]
    
    return prediction

def number_to_ordinal(n):
    """Converts a number into its ordinal representation."""
    n = int(round(n))
    if 10 <= n % 100 <= 20:
        suffix = 'th'
    else:
        suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')
    return f"{n}{suffix}"

def predict_f1_race_results(race_info_list):
    """
    Predict F1 race outcomes for all 20 drivers and sort by predicted position.
    
    Args:
        race_info_list (list): List of dictionaries containing driver information for all 20 F1 drivers
            Each dict should have: driver_name, constructor_name, start_position, circuit_name, race_date
    
    Returns:
        pd.DataFrame: Sorted DataFrame with race predictions
    """
    # Validate input - F1 races have exactly 20 drivers
    if len(race_info_list) != 20:
        print(f"Warning: F1 races normally have 20 drivers, but {len(race_info_list)} were provided.")
    
    results = []
    raw_predictions = []
    
    # Get raw predictions for all drivers
    for info in race_info_list:
        # Validate start position is within F1 range
        if not 1 <= info['start_position'] <= 20:
            print(f"Warning: Invalid starting position ({info['start_position']}) for {info['driver_name']}. F1 uses positions 1-20.")
        
        # Make prediction
        predicted_pos = predict_race_outcome(
            info['driver_name'], 
            info['constructor_name'], 
            info['circuit_name'], 
            info['start_position'], 
            info['race_date']
        )
        
        raw_predictions.append({
            'Driver': info['driver_name'],
            'Constructor': info['constructor_name'],
            'Starting Position': info['start_position'],
            'Raw Predicted Position': predicted_pos
        })
    
    # Sort by raw predicted position
    sorted_predictions = sorted(raw_predictions, key=lambda x: x['Raw Predicted Position'])
    
    # Assign final race positions (1-20) based on the sorted order
    for i, pred in enumerate(sorted_predictions, 1):
        results.append({
            'Position': i,
            'Position (Ordinal)': number_to_ordinal(i),
            'Driver': pred['Driver'],
            'Constructor': pred['Constructor'],
            'Starting Position': pred['Starting Position'],
            'Starting (Ordinal)': number_to_ordinal(pred['Starting Position']),
            'Positions Gained/Lost': pred['Starting Position'] - i,
            'Raw Model Prediction': round(pred['Raw Predicted Position'], 3)
        })
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    return results_df

def format_f1_results(results_df):
    """
    Format F1 race results for easy visualization
    
    Args:
        results_df: DataFrame with race predictions
        
    Returns:
        str: Formatted string of race results
    """
    formatted_output = "Formula 1 Race Prediction Results\n"
    formatted_output += "=" * 70 + "\n\n"
    
    # Create a formatted table
    headers = ["Pos", "Driver", "Constructor", "Grid", "Δ Pos"]
    formatted_output += f"{headers[0]:<6} {headers[1]:<20} {headers[2]:<15} {headers[3]:<6} {headers[4]:<7}\n"
    formatted_output += "-" * 70 + "\n"
    
    for _, row in results_df.iterrows():
        pos_change = row['Positions Gained/Lost']
        if pos_change > 0:
            pos_change_str = f"+{pos_change}"
        else:
            pos_change_str = str(pos_change)
            
        formatted_output += f"{row['Position (Ordinal)']:<6} {row['Driver']:<20} {row['Constructor']:<15} {row['Starting (Ordinal)']:<6} {pos_change_str:<7}\n"
    
    return formatted_output

def run_f1_prediction(race_info):
    """
    Run F1 prediction for a complete grid of 20 drivers
    
    Args:
        race_info: List of dictionaries with driver info
        
    Returns:
        DataFrame with race results and formatted string output
    """
    # Check if model exists, if not train a new one
    model_path = 'f1_race_model.keras'
    
    if not os.path.exists(model_path):
        print("No existing F1 model found. Training a new model...")
        train_keras_model()
    else:
        print(f"Found existing F1 model at {model_path}. Ready for race predictions.")
    
    # Predict race results
    results_df = predict_f1_race_results(race_info)
    
    # Format results for display
    formatted_results = format_f1_results(results_df)
    
    print("\nPredicted F1 Race Results:")
    print(formatted_results)
    
    return results_df, formatted_results

# Example usage
if __name__ == "__main__":
    # Example: Set up race information for all 20 F1 drivers
    circuit = "Losail International Circuit"
    race_date = "2025-04-06"
    
    # This would be replaced with user input for all 20 drivers
    race_info = [
    {"driver_name": "Max Verstappen", "constructor_name": "Red Bull", "start_position": 20, "circuit_name": "Silverstone Circuit", "race_date": "2025-07-14"},
    {"driver_name": "Lando Norris", "constructor_name": "McLaren", "start_position": 12, "circuit_name": "Silverstone Circuit", "race_date": "2025-07-14"},
    {"driver_name": "Lewis Hamilton", "constructor_name": "Mercedes", "start_position": 1, "circuit_name": "Silverstone Circuit", "race_date": "2025-07-14"},
    {"driver_name": "Charles Leclerc", "constructor_name": "Ferrari", "start_position": 16, "circuit_name": "Silverstone Circuit", "race_date": "2025-07-14"},
    {"driver_name": "George Russell", "constructor_name": "Mercedes", "start_position": 7, "circuit_name": "Silverstone Circuit", "race_date": "2025-07-14"},
    {"driver_name": "Carlos Sainz", "constructor_name": "Ferrari", "start_position": 19, "circuit_name": "Silverstone Circuit", "race_date": "2025-07-14"},
    {"driver_name": "Sergio Perez", "constructor_name": "Red Bull", "start_position": 14, "circuit_name": "Silverstone Circuit", "race_date": "2025-07-14"},
    {"driver_name": "Oscar Piastri", "constructor_name": "McLaren", "start_position": 2, "circuit_name": "Silverstone Circuit", "race_date": "2025-07-14"},
    {"driver_name": "Fernando Alonso", "constructor_name": "Aston Martin", "start_position": 18, "circuit_name": "Silverstone Circuit", "race_date": "2025-07-14"},
    {"driver_name": "Lance Stroll", "constructor_name": "Aston Martin", "start_position": 10, "circuit_name": "Silverstone Circuit", "race_date": "2025-07-14"},
    {"driver_name": "Esteban Ocon", "constructor_name": "Alpine", "start_position": 11, "circuit_name": "Silverstone Circuit", "race_date": "2025-07-14"},
    {"driver_name": "Pierre Gasly", "constructor_name": "Alpine", "start_position": 3, "circuit_name": "Silverstone Circuit", "race_date": "2025-07-14"},
    {"driver_name": "Yuki Tsunoda", "constructor_name": "RB", "start_position": 17, "circuit_name": "Silverstone Circuit", "race_date": "2025-07-14"},
    {"driver_name": "Daniel Ricciardo", "constructor_name": "RB", "start_position": 4, "circuit_name": "Silverstone Circuit", "race_date": "2025-07-14"},
    {"driver_name": "Valtteri Bottas", "constructor_name": "Kick Sauber", "start_position": 8, "circuit_name": "Silverstone Circuit", "race_date": "2025-07-14"},
    {"driver_name": "Zhou Guanyu", "constructor_name": "Kick Sauber", "start_position": 6, "circuit_name": "Silverstone Circuit", "race_date": "2025-07-14"},
    {"driver_name": "Kevin Magnussen", "constructor_name": "Haas", "start_position": 15, "circuit_name": "Silverstone Circuit", "race_date": "2025-07-14"},
    {"driver_name": "Nico Hülkenberg", "constructor_name": "Haas", "start_position": 1, "circuit_name": "Silverstone Circuit", "race_date": "2025-07-14"},
    {"driver_name": "Alexander Albon", "constructor_name": "Williams", "start_position": 9, "circuit_name": "Silverstone Circuit", "race_date": "2025-07-14"},
    {"driver_name": "Logan Sargeant", "constructor_name": "Williams", "start_position": 13, "circuit_name": "Silverstone Circuit", "race_date": "2025-07-14"}
    ]

    
    # Run prediction for the full F1 grid
    results, formatted_output = run_f1_prediction(race_info)