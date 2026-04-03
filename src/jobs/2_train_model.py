import os
import sys
import pandas as pd
import lightgbm as lgb
import joblib
from datetime import datetime, timezone

# ==========================================
# CAU HINH DUONG DAN
# ==========================================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(PROJECT_ROOT)

from src.features.common_features import build_features

DATASET_PATH = os.path.join(PROJECT_ROOT, "data", "training_sets", "dataset_v_new.csv")
MODELS_ARCHIVE_DIR = os.path.join(PROJECT_ROOT, "models", "archive")

def train_new_model():
    print("="*60)
    print("BUOC 3.2: HUAN LUYEN MODEL MOI VOI DU LIEU CAP NHAT")
    print("="*60)

    # 1. Kiem tra thu muc
    os.makedirs(MODELS_ARCHIVE_DIR, exist_ok=True)
    if not os.path.exists(DATASET_PATH):
        print(f"LOI: Khong tim thay file dataset moi tai: {DATASET_PATH}")
        print("Meo: Hay chac chan ban da chay 1_merge_data.py truoc do.")
        sys.exit(1)

    # 2. Doc du lieu
    print(f"Dang doc dataset tu: {DATASET_PATH}")
    df = pd.read_csv(DATASET_PATH)
    print(f"  -> So luong data: {len(df):,} dong")

    # 3. Kiem tra va giu lai label de day vao build_features
    if 'label' not in df.columns:
        print("LOI: Dataset bi thieu cot 'label'. Khong the Huan Luyen.")
        sys.exit(1)

    # Giu lai cot label de build_features khong bi loi
    label_col = df['label'].copy()
    
    # 4. Trich xuat dac trung (Feature Engineering)
    print("\nDang tien hanh Trich Xuat Dac Trung (Feature Engineering)...")
    try:
        X, y = build_features(df)
    except Exception as e:
        print(f"LOI trong qua trinh Feature Engineering: {e}")
        sys.exit(1)

    # Đảm bảo y là series chứ không phải None
    if y is None:
         y = label_col
         # Nếu độ dài của X sau khi build_features bị giảm (do dropna), cần đồng bộ y
         if len(X) != len(df):
             # Lấy index của X để lọc y
             y = df.loc[X.index, 'label']
             print(f"  -> Da dong bo lai y voi X. Kich thuoc moi: {len(y)} dong.")

    print(f"  -> Hoan tat. Kich thuoc Vector dac trung X: {X.shape}")
    print(f"  -> Kich thuoc vector Nhan y: {y.shape}")

    # 5. Cau hinh va Huan luyen LightGBM
    print("\nDang tien hanh Huan luyen LightGBM...")
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
    print(f"  -> Huan luyen thanh cong trong {duration:.2f} giay.")

    # 6. Luu Model voi the thoi gian (Versioning)
    today_str = datetime.now(timezone.utc).strftime('%Y-%m-%d_%H-%M-%S')
    model_filename = f"model_{today_str}.pkl"
    model_path = os.path.join(MODELS_ARCHIVE_DIR, model_filename)

    joblib.dump(model, model_path)
    print(f"\nDa luu Model phien ban moi tai:\n   {model_path}")

if __name__ == "__main__":
    train_new_model()