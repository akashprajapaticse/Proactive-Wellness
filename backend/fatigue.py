import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import joblib
import logging

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def generate_fatigue_data(num_samples=1000):
    """
    Generates synthetic data for fatigue prediction.
    Features mimic the real-time metrics collected by the wellness app.
    """
    logging.info(f"Generating {num_samples} synthetic data samples for fatigue model training...")

    data = {
        'heart_rate': np.random.randint(60, 100, num_samples),
        'mouse_activity': np.random.uniform(0, 500, num_samples), # Pixels per second
        'keyboard_activity': np.random.uniform(0, 80, num_samples), # Words per minute
        'gaze_stability': np.random.uniform(0, 100, num_samples), # Percentage
        'emotion': np.random.choice(['Neutral', 'Happy', 'Sad', 'Angry', 'Surprise', 'Fear', 'Disgust'], num_samples, p=[0.4, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),
        'posture': np.random.choice(['Optimal', 'Slightly Slouching', 'Leaning Forward (Strained)', 'Poor (Slumped)'], num_samples, p=[0.5, 0.2, 0.2, 0.1]),
        'lighting': np.random.choice(['Optimal', 'Too Dim', 'Too Bright'], num_samples, p=[0.6, 0.2, 0.2]),
        'distance': np.random.choice(['Optimal', 'Too Close', 'Too Far'], num_samples, p=[0.6, 0.2, 0.2]),
        'hydration_level': np.random.uniform(0, 100, num_samples), # Percentage
    }
    df = pd.DataFrame(data)

    # --- Generate a synthetic 'fatigue_score' based on the features ---
    # This is a simplified linear relationship for demonstration.
    # A real fatigue score would be determined by user feedback or more complex physiological models.
    df['fatigue_score'] = (
        (100 - df['heart_rate']) * 0.2 + # Lower heart rate might imply less exertion, but could also be rest (low fatigue)
        (500 - df['mouse_activity']) * 0.05 + # Low mouse activity might indicate low engagement/fatigue
        (80 - df['keyboard_activity']) * 0.1 + # Low keyboard activity might indicate low engagement/fatigue
        (100 - df['gaze_stability']) * 0.3 + # Unstable gaze strongly correlates with fatigue
        (100 - df['hydration_level']) * 0.2 # Low hydration strongly correlates with fatigue
    )

    # Add modifiers based on categorical features
    df.loc[df['emotion'] == 'Sad', 'fatigue_score'] += 15
    df.loc[df['emotion'] == 'Angry', 'fatigue_score'] += 10
    df.loc[df['emotion'] == 'Neutral', 'fatigue_score'] += 5 # Neutral can still be fatigued
    df.loc[df['posture'] == 'Slightly Slouching', 'fatigue_score'] += 10
    df.loc[df['posture'] == 'Leaning Forward (Strained)', 'fatigue_score'] += 15
    df.loc[df['posture'] == 'Poor (Slumped)', 'fatigue_score'] += 25
    df.loc[df['lighting'] == 'Too Dim', 'fatigue_score'] += 10
    df.loc[df['lighting'] == 'Too Bright', 'fatigue_score'] += 5
    df.loc[df['distance'] == 'Too Close', 'fatigue_score'] += 8
    df.loc[df['distance'] == 'Too Far', 'fatigue_score'] += 12

    # Clip fatigue score to be between 0 and 100
    df['fatigue_score'] = np.clip(df['fatigue_score'], 0, 100)
    
    logging.info("Synthetic data generation complete.")
    return df

def train_fatigue_model(df):
    """
    Trains a RandomForestRegressor model on the generated data.
    Encodes categorical features and saves the trained model.
    """
    logging.info("Starting fatigue model training...")

    # Define features (X) and target (y)
    features = [
        'heart_rate', 'mouse_activity', 'keyboard_activity', 'gaze_stability',
        'emotion', 'posture', 'lighting', 'distance', 'hydration_level'
    ]
    target = 'fatigue_score'

    X = df[features]
    y = df[target]

    # --- One-Hot Encode Categorical Features ---
    # This is crucial for tree-based models like RandomForestRegressor.
    # It converts categorical text labels into numerical format.
    X = pd.get_dummies(X, columns=['emotion', 'posture', 'lighting', 'distance'], drop_first=True)
    
    # Store the list of columns (feature names) after one-hot encoding.
    # This is vital because the prediction input in the Flask app MUST have the same columns
    # in the same order, even if some categorical features are not present in a given frame.
    global feature_columns
    feature_columns = X.columns.tolist()
    logging.info(f"Model will be trained on the following features: {feature_columns}")

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train a RandomForestRegressor model
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1) # n_jobs=-1 uses all available cores
    model.fit(X_train, y_train)

    # Evaluate the model (optional, for verification)
    score = model.score(X_test, y_test)
    logging.info(f"Model training complete. R^2 score on test set: {score:.4f}")

    return model

if __name__ == "__main__":
    # Generate data
    data_df = generate_fatigue_data(num_samples=2000) # Increased sample size for better training

    # Train model
    trained_model = train_fatigue_model(data_df)

    # Save the model and its feature columns
    model_save_path = 'fatigue_model.joblib'
    feature_columns_save_path = 'fatigue_model_features.joblib' # Save feature names separately

    joblib.dump({'model': trained_model, 'feature_columns': feature_columns}, model_save_path)
    # joblib.dump(feature_columns, feature_columns_save_path) # Not strictly needed if saved with model
    logging.info(f"Fatigue prediction model and feature columns saved successfully to {model_save_path}")

    logging.info("Model generation script finished.")