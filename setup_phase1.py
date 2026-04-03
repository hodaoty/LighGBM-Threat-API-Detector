import os
import pandas as pd
from sklearn.model_selection import train_test_split

# ==========================================
# CẤU HÌNH ĐƯỜNG DẪN
# ==========================================
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))

# File dữ liệu gốc hiện tại của bạn
MASTER_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "master_dataset_cleaned.csv")

# Các thư mục cần tạo theo Checklist
DIRS_TO_CREATE = [
    "data/daily_logs",
    "data/training_sets",
    "models/archive",
    "models/production"
]

def setup_directories():
    print("BƯỚC 1: ĐANG TẠO CẤU TRÚC THƯ MỤC...")
    for d in DIRS_TO_CREATE:
        path = os.path.join(PROJECT_ROOT, d)
        os.makedirs(path, exist_ok=True)
        print(f"  -> Đã kiểm tra/tạo thư mục: {d}")

def create_golden_dataset():
    print("\nBƯỚC 2: ĐANG TẠO GOLDEN DATASET (TẬP ĐỀ THI VÀNG)...")
    if not os.path.exists(MASTER_DATA_PATH):
        print(f"LỖI: Không tìm thấy file gốc tại {MASTER_DATA_PATH}")
        print("Vui lòng kiểm tra lại đường dẫn file dữ liệu sạch của bạn.")
        return

    df = pd.read_csv(MASTER_DATA_PATH)
    total_rows = len(df)
    print(f"  -> Đã tải Master Dataset với {total_rows:,} dòng.")

    if total_rows < 3000:
        print("CẢNH BÁO: Dataset của bạn khá nhỏ, việc trích 2000 dòng có thể chiếm quá nhiều dữ liệu train.")
        test_size = int(total_rows * 0.2) # Nếu quá ít thì lấy 20%
    else:
        test_size = 2000

    # Trích xuất Golden Dataset (Bắt buộc phải stratify để đảm bảo tỷ lệ Bình thường/Tấn công đồng đều)
    if 'label' in df.columns:
        df_train, df_golden = train_test_split(df, test_size=test_size, random_state=42, stratify=df['label'])
    else:
        df_train, df_golden = train_test_split(df, test_size=test_size, random_state=42)

    # Lưu Golden Dataset
    golden_path = os.path.join(PROJECT_ROOT, "data", "training_sets", "golden_test_set.csv")
    df_golden.to_csv(golden_path, index=False)
    print(f"  -> Đã tạo thành công Golden Dataset ({len(df_golden):,} dòng) tại: {golden_path}")

    # Lưu phần còn lại thành Version 1 của Training Set
    v1_path = os.path.join(PROJECT_ROOT, "data", "training_sets", "dataset_v_current.csv")
    df_train.to_csv(v1_path, index=False)
    print(f"  -> Đã lưu dữ liệu huấn luyện hiện tại ({len(df_train):,} dòng) tại: {v1_path}")

    # Thống kê Golden Dataset
    if 'label' in df_golden.columns:
        print("\nTHỐNG KÊ GOLDEN DATASET (Đề thi chuẩn):")
        labels = df_golden['label'].value_counts()
        print(f"   Bình thường (0): {labels.get(0, 0):,}")
        print(f"   Tấn công (1)   : {labels.get(1, 0):,}")

if __name__ == "__main__":
    print("BẮT ĐẦU CHẠY THIẾT LẬP PHASE 1...")
    setup_directories()
    create_golden_dataset()
    print("\nHOÀN TẤT PHASE 1! Hệ thống lưu trữ đã sẵn sàng.")