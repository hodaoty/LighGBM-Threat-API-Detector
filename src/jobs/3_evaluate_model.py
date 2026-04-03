import os
import sys
import glob
import pandas as pd
import joblib
from datetime import datetime, timezone
from sklearn.metrics import classification_report, f1_score, roc_auc_score

# ==========================================
# CAU HINH DUONG DAN
# ==========================================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(PROJECT_ROOT)

from src.features.common_features import build_features

GOLDEN_DATASET_PATH = os.path.join(PROJECT_ROOT, "data", "training_sets", "golden_test_set.csv")
MODELS_ARCHIVE_DIR = os.path.join(PROJECT_ROOT, "models", "archive")
AUDIT_LOG_PATH = os.path.join(PROJECT_ROOT, "mlops_audit.log")

def evaluate_latest_model():
    print("="*60)
    print("BUOC 3.3: KIEM DINH CHAT LUONG MODEL TREN GOLDEN DATASET")
    print("="*60)

    # 1. Tim model moi nhat vua duoc train trong thu muc archive
    list_of_models = glob.glob(os.path.join(MODELS_ARCHIVE_DIR, '*.pkl'))
    if not list_of_models:
        print("LOI: Khong tim thay model nao trong thu muc archive!")
        sys.exit(1)
        
    latest_model_path = max(list_of_models, key=os.path.getctime)
    model_name = os.path.basename(latest_model_path)
    print(f"Dang kiem dinh Model moi nhat: {model_name}")

    # 2. Doc tap de thi Vang (Golden Dataset)
    if not os.path.exists(GOLDEN_DATASET_PATH):
        print(f"LOI: Khong tim thay Golden Dataset tai {GOLDEN_DATASET_PATH}")
        sys.exit(1)
        
    print("Dang tai Golden Dataset (De thi chuan)...")
    df_golden = pd.read_csv(GOLDEN_DATASET_PATH)
    
    if 'label' not in df_golden.columns:
        print("LOI: Golden Dataset bi thieu cot 'label'.")
        sys.exit(1)

    label_col = df_golden['label'].copy()

    # 3. Trich xuat dac trung cho tap thi
    try:
        X_golden, y_golden = build_features(df_golden)
    except Exception as e:
        print(f"LOI trong qua trinh Feature Engineering tren tap thi: {e}")
        sys.exit(1)
        
    if y_golden is None:
        y_golden = label_col
        if len(X_golden) != len(df_golden):
            y_golden = df_golden.loc[X_golden.index, 'label']

    # 4. Load model va Tien hanh thi
    print("\nDang tien hanh lam bai thi...")
    model = joblib.load(latest_model_path)
    
    y_pred = model.predict(X_golden)
    y_proba = model.predict_proba(X_golden)[:, 1]

    # 5. Tinh toan diem so
    f1 = f1_score(y_golden, y_pred)
    auc = roc_auc_score(y_golden, y_proba)
    
    print("\nKET QUA BAI THI (METRICS):")
    print("-" * 40)
    print(f" - F1-Score : {f1:.4f} (Cang gan 1.0 cang tot)")
    print(f" - AUC Score: {auc:.4f} (Cang gan 1.0 cang tot)")
    print("-" * 40)
    
    print("\nChi tiet tung nhan (Classification Report):")
    print(classification_report(y_golden, y_pred, target_names=['Normal (0)', 'Attack (1)'], digits=4))

    # 6. Ghi lich su vao Audit Log
    today_str = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
    log_entry = f"[{today_str}] Model: {model_name} | F1: {f1:.4f} | AUC: {auc:.4f}\n"
    
    with open(AUDIT_LOG_PATH, "a") as f:
        f.write(log_entry)
        
    print(f"Da ghi ket qua vao so xet duyet: {AUDIT_LOG_PATH}")
    print("\nNeu diem F1 giu on dinh o muc cao, ban da san sang de DEPLOY!")

if __name__ == "__main__":
    evaluate_latest_model()