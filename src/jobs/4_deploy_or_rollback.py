import os
import sys
import glob
import shutil
import pandas as pd
import joblib
from datetime import datetime, timezone
from sklearn.metrics import f1_score

# ==========================================
# PATH CONFIGURATION
# ==========================================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(PROJECT_ROOT)

from src.features.common_features import build_features

MODELS_ARCHIVE_DIR = os.path.join(PROJECT_ROOT, "models", "archive")
MODELS_PROD_DIR = os.path.join(PROJECT_ROOT, "models", "production")
ACTIVE_MODEL_PATH = os.path.join(MODELS_PROD_DIR, "active_model.pkl")

# Data Paths for Rollback logic
TEST_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "training_sets", "golden_test_set.csv")
CURRENT_DATASET_PATH = os.path.join(PROJECT_ROOT, "data", "training_sets", "dataset_v_current.csv")
BACKUP_DATASET_PATH = os.path.join(PROJECT_ROOT, "data", "training_sets", "dataset_v_current_backup.csv")

AUDIT_LOG_PATH = os.path.join(PROJECT_ROOT, "mlops_audit.log")

def get_f1_score(model_path, X_test, y_test):
    """Helper function: Takes model path, returns F1 score"""
    try:
        model = joblib.load(model_path)
        y_pred = model.predict(X_test)
        return f1_score(y_test, y_pred)
    except Exception as e:
        print(f"Error while evaluating model {model_path}: {e}")
        return -1.0

def deploy_or_rollback():
    print("="*60)
    print("STEP 4: DECIDE DEPLOY OR ROLLBACK (CI/CD)")
    print("="*60)

    # Create production directory if not exists
    os.makedirs(MODELS_PROD_DIR, exist_ok=True)

    # 1. Find the newest model in archive
    list_of_models = glob.glob(os.path.join(MODELS_ARCHIVE_DIR, '*.pkl'))
    if not list_of_models:
        print("ERROR: No models found in archive to deploy.")
        sys.exit(1)
        
    newest_model_path = max(list_of_models, key=os.path.getctime)
    newest_model_name = os.path.basename(newest_model_path)
    
    # 2. Prepare test data (Golden Dataset)
    if not os.path.exists(TEST_DATA_PATH):
        print("ERROR: Golden Dataset not found.")
        sys.exit(1)
        
    df_test = pd.read_csv(TEST_DATA_PATH)
    label_col = df_test['label'].copy()
    
    try:
        X_test, y_test = build_features(df_test)
    except Exception as e:
        print(f"ERROR in Feature Engineering: {e}")
        sys.exit(1)
        
    if y_test is None:
        y_test = label_col
        if len(X_test) != len(df_test):
            y_test = df_test.loc[X_test.index, 'label']

    # 3. Evaluate the NEW model
    print(f"Evaluating NEW Model: {newest_model_name}...")
    f1_new = get_f1_score(newest_model_path, X_test, y_test)
    print(f"  -> F1-Score (New): {f1_new:.4f}")

    today_str = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
    action_log = ""

    # 4. Compare with OLD model and decide
    if not os.path.exists(ACTIVE_MODEL_PATH):
        print("\nActive Model not found. This is the first deployment!")
        print("DECISION: DEPLOY (Accepting new model)")
        
        # Copy file to production dir and rename to active_model.pkl
        shutil.copy2(newest_model_path, ACTIVE_MODEL_PATH)
        action_log = f"[DEPLOY-FIRST] {newest_model_name} (F1: {f1_new:.4f})"
    else:
        print("\nEvaluating CURRENT Model (Production)...")
        f1_old = get_f1_score(ACTIVE_MODEL_PATH, X_test, y_test)
        print(f"  -> F1-Score (Old): {f1_old:.4f}")
        
        print("\nCOMPARISON AND DECISION:")
        # Allow slight drop of 0.005 to prioritize learning new behaviors
        if f1_new >= (f1_old - 0.005):
            print(f"DECISION: DEPLOY (New score {f1_new:.4f} meets standard vs Old {f1_old:.4f})")
            shutil.copy2(newest_model_path, ACTIVE_MODEL_PATH)
            action_log = f"[DEPLOY] Replaced by {newest_model_name} (New: {f1_new:.4f} vs Old: {f1_old:.4f})"
        else:
            print(f"DECISION: ROLLBACK (New score {f1_new:.4f} < Old {f1_old:.4f})")
            print("Deployment rejected. The system keeps the current model.")
            
            # --- DATA ROLLBACK LOGIC ---
            print("\nInitiating DATA ROLLBACK (Removing today's poisoned data)...")
            if os.path.exists(BACKUP_DATASET_PATH):
                shutil.copy2(BACKUP_DATASET_PATH, CURRENT_DATASET_PATH)
                print(f"  -> Successfully restored {CURRENT_DATASET_PATH} from backup!")
            else:
                print("  -> ERROR: Backup file not found to restore data.")
            # ---------------------------
            
            action_log = f"[ROLLBACK] Rejected {newest_model_name} (New: {f1_new:.4f} < Old: {f1_old:.4f}) - Data Restored"

    # 5. Write to Audit Log
    with open(AUDIT_LOG_PATH, "a") as f:
        f.write(f"[{today_str}] {action_log}\n")
        
    print(f"\nSystem history logged to: {AUDIT_LOG_PATH}")

if __name__ == "__main__":
    deploy_or_rollback()