import os
import sys
import glob
import pandas as pd
import joblib
from datetime import datetime, timezone
from sklearn.metrics import classification_report, f1_score, roc_auc_score

# ==========================================
# PATH CONFIGURATION
# ==========================================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(PROJECT_ROOT)

from src.features.common_features import build_features

GOLDEN_DATASET_PATH = os.path.join(PROJECT_ROOT, "data", "training_sets", "golden_test_set.csv")
MODELS_ARCHIVE_DIR = os.path.join(PROJECT_ROOT, "models", "archive")
AUDIT_LOG_PATH = os.path.join(PROJECT_ROOT, "mlops_audit.log")

def evaluate_latest_model():
    print("="*60)
    print("STEP 3.3: EVALUATE MODEL QUALITY ON GOLDEN DATASET")
    print("="*60)

    # 1. Find the latest trained model in the archive directory
    list_of_models = glob.glob(os.path.join(MODELS_ARCHIVE_DIR, '*.pkl'))
    if not list_of_models:
        print("ERROR: No models found in the archive directory!")
        sys.exit(1)
        
    latest_model_path = max(list_of_models, key=os.path.getctime)
    model_name = os.path.basename(latest_model_path)
    print(f"Evaluating the latest Model: {model_name}")

    # 2. Read Golden Dataset (Standard test exam)
    if not os.path.exists(GOLDEN_DATASET_PATH):
        print(f"ERROR: Cannot find Golden Dataset at {GOLDEN_DATASET_PATH}")
        sys.exit(1)
        
    print("Loading Golden Dataset...")
    df_golden = pd.read_csv(GOLDEN_DATASET_PATH)
    
    if 'label' not in df_golden.columns:
        print("ERROR: Golden Dataset is missing the 'label' column.")
        sys.exit(1)

    label_col = df_golden['label'].copy()

    # 3. Extract features for the test set
    try:
        X_golden, y_golden = build_features(df_golden)
    except Exception as e:
        print(f"ERROR during Feature Engineering on the test set: {e}")
        sys.exit(1)
        
    if y_golden is None:
        y_golden = label_col
        if len(X_golden) != len(df_golden):
            y_golden = df_golden.loc[X_golden.index, 'label']

    # 4. Load model and start evaluation
    print("\nEvaluating...")
    model = joblib.load(latest_model_path)
    
    y_pred = model.predict(X_golden)
    y_proba = model.predict_proba(X_golden)[:, 1]

    # 5. Calculate scores
    f1 = f1_score(y_golden, y_pred)
    auc = roc_auc_score(y_golden, y_proba)
    
    print("\nEVALUATION RESULTS (METRICS):")
    print("-" * 40)
    print(f" - F1-Score : {f1:.4f} (Closer to 1.0 is better)")
    print(f" - AUC Score: {auc:.4f} (Closer to 1.0 is better)")
    print("-" * 40)
    
    print("\nDetailed Classification Report:")
    print(classification_report(y_golden, y_pred, target_names=['Normal (0)', 'Attack (1)'], digits=4))

    # 6. Write history to Audit Log
    today_str = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
    log_entry = f"[{today_str}] Model: {model_name} | F1: {f1:.4f} | AUC: {auc:.4f}\n"
    
    with open(AUDIT_LOG_PATH, "a") as f:
        f.write(log_entry)
        
    print(f"Results logged to audit file: {AUDIT_LOG_PATH}")
    print("\nIf the F1 score remains high and stable, you are ready to DEPLOY!")

if __name__ == "__main__":
    evaluate_latest_model()