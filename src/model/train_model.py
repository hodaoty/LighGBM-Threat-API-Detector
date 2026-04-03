import pandas as pd
import lightgbm as lgb
import joblib
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# ==========================================
# ⚙️ CẤU HÌNH ĐƯỜNG DẪN MÔI TRƯỜNG
# ==========================================
# Thêm thư mục gốc của dự án vào sys.path để có thể import được module 'src'
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(PROJECT_ROOT)

from src.features.common_features import build_features

# Đường dẫn file dữ liệu và thư mục lưu model (Hãy sửa lại nếu cần)
DATA_PATH = r"n:/hodaoty/API-Threat-Detection/data/processed/master_dataset_cleaned.csv"
MODEL_DIR = r"n:/hodaoty/API-Threat-Detection/models"

def train():
    print("BƯỚC 1: ĐANG ĐỌC DỮ LIỆU...")
    if not os.path.exists(DATA_PATH):
        print(f"Không tìm thấy file dữ liệu tại: {DATA_PATH}")
        return
        
    df = pd.read_csv(DATA_PATH)
    print(f"Tải thành công {len(df):,} dòng log.")

    # ---------------------------------------------------------
    print("\nBƯỚC 2: GỌI HÀM TRÍCH XUẤT ĐẶC TRƯNG...")
    # Gọi hàm build_features mà bạn đã định nghĩa trong src/features/common_features.py
    X, y = build_features(df)
    
    if y is None:
        print("LỖI: Dữ liệu của bạn không có cột 'label'. Không thể huấn luyện AI!")
        return

    print(f"Đã trích xuất xong {X.shape[1]} đặc trưng.")
    
    # ---------------------------------------------------------
    print("\nBƯỚC 3: HUẤN LUYỆN LIGHTGBM...")
    # Chia tập Train (80%) và Test (20%)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Khởi tạo thuật toán LightGBM Classifier
    model = lgb.LGBMClassifier(
        n_estimators=200,          # Số lượng cây
        learning_rate=0.05,        # Tốc độ học
        max_depth=8,               # Độ sâu của cây
        random_state=42,
        class_weight='balanced',   # Tự cân bằng nếu Normal > Attack
        n_jobs=-1                  # Dùng toàn bộ nhân CPU
    )
    
    # LightGBM tự động nhận diện các cột có kiểu 'category' (mà ta đã ép kiểu ở common_features.py)
    model.fit(X_train, y_train)
    print("Huấn luyện hoàn tất!")

    # ---------------------------------------------------------
    print("\nBƯỚC 4: LƯU MÔ HÌNH...")
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, "lightgbm_threatAPI_detector.pkl")
    joblib.dump(model, model_path)
    print(f"Đã lưu bộ não AI tại: {model_path}")

    # ---------------------------------------------------------
    print("\n" + "="*50)
    print("BÁO CÁO ĐÁNH GIÁ (EVALUATION REPORT)")
    print("="*50)
    
    y_pred = model.predict(X_test)
    
    print("1. Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Normal (0)', 'Attack (1)'], digits=4))
    
    print("-" * 50)
    print("2. Ma trận nhầm lẫn (Confusion Matrix):")
    cm = confusion_matrix(y_test, y_pred)
    print(f"   - [Bình thường đoán đúng] : {cm[0][0]:,} | [Cảnh báo giả] : {cm[0][1]:,}")
    print(f"   - [Bỏ lọt tấn công]       : {cm[1][0]:,} | [Bắt sống Hacker]: {cm[1][1]:,}")
    
    print("-" * 50)
    print("3. TOP 5 Tính năng Quan trọng nhất:")
    importance = pd.DataFrame({'Feature': X.columns, 'Importance': model.feature_importances_})
    importance = importance.sort_values(by='Importance', ascending=False)
    for index, row in importance.head(5).iterrows():
        print(f"   -> {row['Feature']:<20} : {row['Importance']}")

if __name__ == "__main__":
    train()