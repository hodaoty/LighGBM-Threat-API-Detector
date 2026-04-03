import os
import sys
import glob
import shutil
import pandas as pd
import joblib
from datetime import datetime, timezone
from sklearn.metrics import f1_score

# ==========================================
# CAU HINH DUONG DAN
# ==========================================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(PROJECT_ROOT)

from src.features.common_features import build_features

MODELS_ARCHIVE_DIR = os.path.join(PROJECT_ROOT, "models", "archive")
MODELS_PROD_DIR = os.path.join(PROJECT_ROOT, "models", "production")
ACTIVE_MODEL_PATH = os.path.join(MODELS_PROD_DIR, "active_model.pkl")
TEST_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "training_sets", "golden_test_set.csv")
AUDIT_LOG_PATH = os.path.join(PROJECT_ROOT, "mlops_audit.log")

def get_f1_score(model_path, X_test, y_test):
    """Ham phu tro: Nhan duong dan model, tra ve diem F1"""
    try:
        model = joblib.load(model_path)
        y_pred = model.predict(X_test)
        return f1_score(y_test, y_pred)
    except Exception as e:
        print(f"Loi khi cham diem model {model_path}: {e}")
        return -1.0

def deploy_or_rollback():
    print("="*60)
    print("BUOC 4: QUYET DINH TRIEN KHAI HOAC ROLLBACK ")
    print("="*60)

    # Tao thu muc production neu chua co
    os.makedirs(MODELS_PROD_DIR, exist_ok=True)

    # 1. Tim model moi nhat trong archive
    list_of_models = glob.glob(os.path.join(MODELS_ARCHIVE_DIR, '*.pkl'))
    if not list_of_models:
        print("LOI: Khong tim thay model nao trong archive de deploy.")
        sys.exit(1)
        
    newest_model_path = max(list_of_models, key=os.path.getctime)
    newest_model_name = os.path.basename(newest_model_path)
    
    # 2. Chuan bi du lieu thi (Golden Dataset)
    if not os.path.exists(TEST_DATA_PATH):
        print("LOI: Khong tim thay Golden Dataset.")
        sys.exit(1)
        
    df_test = pd.read_csv(TEST_DATA_PATH)
    label_col = df_test['label'].copy()
    
    try:
        X_test, y_test = build_features(df_test)
    except Exception as e:
        print(f"LOI Feature Engineering: {e}")
        sys.exit(1)
        
    if y_test is None:
        y_test = label_col
        if len(X_test) != len(df_test):
            y_test = df_test.loc[X_test.index, 'label']

    # 3. Cham diem model MOI
    print(f"Dang cham diem Model MOI: {newest_model_name}...")
    f1_new = get_f1_score(newest_model_path, X_test, y_test)
    print(f"  -> F1-Score (Moi): {f1_new:.4f}")

    today_str = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
    action_log = ""

    # 4. So sanh voi model CU va Ra quyet dinh
    if not os.path.exists(ACTIVE_MODEL_PATH):
        print("\nKhong tim thay Active Model cu. Day la lan trien khai dau tien!")
        print("QUYET DINH: DEPLOY (Chap nhan model moi)")
        
        # Copy file vao thu muc production va doi ten thanh active_model.pkl
        shutil.copy2(newest_model_path, ACTIVE_MODEL_PATH)
        action_log = f"[DEPLOY-FIRST] {newest_model_name} (F1: {f1_new:.4f})"
    else:
        print("\nDang cham diem Model HIEN TAI (Production)...")
        f1_old = get_f1_score(ACTIVE_MODEL_PATH, X_test, y_test)
        print(f"  -> F1-Score (Cu): {f1_old:.4f}")
        
        print("\nSO SANH VA RA QUYET DINH:")
        # Cho phep model moi giam nhe 0.005 diem van duoc deploy de uu tien cap nhat hanh vi moi
        if f1_new >= (f1_old - 0.005):
            print(f"QUYET DINH: DEPLOY (Diem Moi {f1_new:.4f} dat chuan so voi Cu {f1_old:.4f})")
            shutil.copy2(newest_model_path, ACTIVE_MODEL_PATH)
            action_log = f"[DEPLOY] Replaced by {newest_model_name} (New: {f1_new:.4f} vs Old: {f1_old:.4f})"
        else:
            print(f"QUYET DINH: ROLLBACK (Diem Moi {f1_new:.4f} < Cu {f1_old:.4f})")
            print("Huy bo trien khai. He thong van giu nguyen model hien tai.")
            action_log = f"[ROLLBACK] Rejected {newest_model_name} (New: {f1_new:.4f} < Old: {f1_old:.4f})"

    # 5. Ghi nhat ky Audit
    with open(AUDIT_LOG_PATH, "a") as f:
        f.write(f"[{today_str}] {action_log}\n")
        
    print(f"\nDa ghi lich su he thong vao: {AUDIT_LOG_PATH}")

if __name__ == "__main__":
    deploy_or_rollback()