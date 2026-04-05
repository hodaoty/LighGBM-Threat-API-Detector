import os
import sys
import pandas as pd
import lightgbm as lgb
import joblib
from datetime import datetime, timezone

# ==========================================
# PATH CONFIGURATION
# ==========================================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(PROJECT_ROOT)

from src.features.common_features import build_features

# Changed to dataset_v_current.csv to match the Snowball mechanism
DATASET_PATH = os.path.join(PROJECT_ROOT, "data", "training_sets", "dataset_v_current.csv")
MODELS_ARCHIVE_DIR = os.path.join(PROJECT_ROOT, "models", "archive")

def train_new_model():
    print("="*60)
    print("STEP 3.2: TRAINING NEW MODEL WITH UPDATED DATA")
    print("="*60)

    # 1. Check directories
    os.makedirs(MODELS_ARCHIVE_DIR, exist_ok=True)
    if not os.path.exists(DATASET_PATH):
        print(f"ERROR: Cannot find the new dataset file at: {DATASET_PATH}")
        print("Hint: Make sure you have run 1_merge_data.py first.")
        sys.exit(1)

    # 2. Read data
    print(f"Reading dataset from: {DATASET_PATH}")
    df = pd.read_csv(DATASET_PATH)
    print(f"  -> Data size: {len(df):,} rows")

    # 3. Check and keep label column
    if 'label' not in df.columns:
        print("ERROR: Dataset is missing 'label' column. Cannot train.")
        sys.exit(1)

    # Keep label column as a backup
    label_col = df['label'].copy()
    
    # 4. Extract features (Feature Engineering)
    print("\nExtracting features (Feature Engineering)...")
    try:
        X, y = build_features(df)
    except Exception as e:
        print(f"ERROR during Feature Engineering: {e}")
        sys.exit(1)

    # Ensure y is a series, not None
    if y is None:
         y = label_col
         # If the length of X is reduced after build_features (e.g., dropna), sync y
         if len(X) != len(df):
             # Use X's index to filter y
             y = df.loc[X.index, 'label']
             print(f"  -> Synced y with X. New size: {len(y)} rows.")

    print(f"  -> Completed. Feature vector X shape: {X.shape}")
    print(f"  -> Label vector y shape: {y.shape}")

    # 5. Configure and Train LightGBM
    print("\nTraining LightGBM model...")
    model = lgb.LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=8,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1
    )

    # Fit model
    start_time = datetime.now()
    model.fit(X, y)
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    print(f"  -> Training successful in {duration:.2f} seconds.")

    # 6. Save Model with timestamp (Versioning)
    today_str = datetime.now(timezone.utc).strftime('%Y-%m-%d_%H-%M-%S')
    model_filename = f"model_{today_str}.pkl"
    model_path = os.path.join(MODELS_ARCHIVE_DIR, model_filename)

    joblib.dump(model, model_path)
    print(f"\nSaved new model version at:\n   {model_path}")

if __name__ == "__main__":
    train_new_model()