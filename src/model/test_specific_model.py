import os
import sys
import pandas as pd
import joblib
from sklearn.metrics import classification_report, f1_score, roc_auc_score, confusion_matrix

# ==========================================
# CAU HINH DUONG DAN
# ==========================================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(PROJECT_ROOT)

from src.features.common_features import build_features

# Duong dan co dinh toi model ban muon test
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "lightgbm_threatAPI_detector.pkl")
# Duong dan toi tap de thi chuan
TEST_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "training_sets", "golden_test_set.csv")

def test_custom_model():
    print("="*60)
    print(f"KIEM TRA MO HINH: {os.path.basename(MODEL_PATH)}")
    print("="*60)

    if not os.path.exists(MODEL_PATH):
        print(f"LOI: Khong tim thay mo hinh tai {MODEL_PATH}")
        return
    if not os.path.exists(TEST_DATA_PATH):
        print(f"LOI: Khong tim thay du lieu test tai {TEST_DATA_PATH}")
        return

    print("1. Dang tai du lieu test (Golden Dataset)...")
    df_test = pd.read_csv(TEST_DATA_PATH)
    label_col = df_test['label'].copy()

    print("2. Dang trich xuat dac trung (Feature Engineering)...")
    try:
        X_test, y_test = build_features(df_test)
    except Exception as e:
        print(f"LOI trich xuat dac trung: {e}")
        return

    # Dong bo label neu co dong bi xoa trong qua trinh build_features
    if y_test is None:
        y_test = label_col
        if len(X_test) != len(df_test):
            y_test = df_test.loc[X_test.index, 'label']

    print("3. Dang tai mo hinh va du doan...")
    model = joblib.load(MODEL_PATH)
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("\n" + "="*60)
    print("KET QUA DANH GIA (EVALUATION REPORT)")
    print("="*60)
    
    print(f" - F1-Score : {f1_score(y_test, y_pred):.4f}")
    print(f" - AUC Score: {roc_auc_score(y_test, y_proba):.4f}")
    print("-" * 50)

    print("\nBao cao chi tiet (Classification Report):")
    print(classification_report(y_test, y_pred, target_names=['Normal (0)', 'Attack (1)'], digits=4))

    print("-" * 50)
    print("Ma tran nham lan (Confusion Matrix):")
    cm = confusion_matrix(y_test, y_pred)
    print(f"   - [Binh thuong doan dung] : {cm[0][0]:,} | [Canh bao gia] : {cm[0][1]:,}")
    print(f"   - [Bo lot tan cong]       : {cm[1][0]:,} | [Bat song Hacker]: {cm[1][1]:,}")

if __name__ == "__main__":
    test_custom_model()