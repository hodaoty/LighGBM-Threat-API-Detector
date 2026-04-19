import pandas as pd
import lightgbm as lgb
import joblib
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# ==========================================
# ENVIRONMENT PATH CONFIGURATION
# ==========================================
# Add project root directory to sys.path to be able to import the 'src' module
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(PROJECT_ROOT)

from src.features.common_features import build_features

# Data file path and model save directory (Modify if necessary)
DATA_PATH = r"n:/hodaoty/API-Threat-Detection/data/processed/master_dataset_cleaned.csv"
MODEL_DIR = r"n:/hodaoty/API-Threat-Detection/models"

def train():
    print("STEP 1: READING DATA...")
    if not os.path.exists(DATA_PATH):
        print(f"Data file not found at: {DATA_PATH}")
        return
        
    df = pd.read_csv(DATA_PATH)
    print(f"Successfully loaded {len(df):,} log lines.")

    # ---------------------------------------------------------
    print("\nSTEP 2: CALLING FEATURE EXTRACTION FUNCTION...")
    # Call the build_features function defined in src/features/common_features.py
    X, y = build_features(df)
    
    if y is None:
        print("ERROR: Your data does not have a 'label' column. Cannot train AI!")
        return

    print(f"Extracted {X.shape[1]} features.")
    
    # ---------------------------------------------------------
    print("\nSTEP 3: TRAINING LIGHTGBM...")
    # Split Train (80%) and Test (20%) sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Initialize LightGBM Classifier algorithm
    model = lgb.LGBMClassifier(
        n_estimators=200,          # Number of trees
        learning_rate=0.05,        # Learning rate
        max_depth=8,               # Tree depth
        random_state=42,
        class_weight='balanced',   # Auto balance if Normal > Attack
        n_jobs=-1                  # Use all CPU cores
    )
    
    # LightGBM automatically recognizes 'category' type columns (which we casted in common_features.py)
    model.fit(X_train, y_train)
    print("Training complete!")

    # ---------------------------------------------------------
    print("\nSTEP 4: SAVING MODEL...")
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, "lightgbm_threatAPI_detector.pkl")
    joblib.dump(model, model_path)
    print(f"Saved AI model at: {model_path}")

    # ---------------------------------------------------------
    print("\n" + "="*50)
    print("EVALUATION REPORT")
    print("="*50)
    
    y_pred = model.predict(X_test)
    
    print("1. Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Normal (0)', 'Attack (1)'], digits=4))
    
    print("-" * 50)
    print("2. Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"   - [True Normal]   : {cm[0][0]:,} | [False Alarm]  : {cm[0][1]:,}")
    print(f"   - [Missed Attack] : {cm[1][0]:,} | [Caught Hacker]: {cm[1][1]:,}")
    
    print("-" * 50)
    print("3. TOP 5 Most Important Features:")
    importance = pd.DataFrame({'Feature': X.columns, 'Importance': model.feature_importances_})
    importance = importance.sort_values(by='Importance', ascending=False)
    for index, row in importance.head(5).iterrows():
        print(f"   -> {row['Feature']:<20} : {row['Importance']}")

if __name__ == "__main__":
    train()