import os
import sys
import pandas as pd
from datetime import datetime, timezone

# ==========================================
# CẤU HÌNH ĐƯỜNG DẪN
# ==========================================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))

DAILY_LOGS_DIR = os.path.join(PROJECT_ROOT, "data", "daily_logs")
TRAINING_SETS_DIR = os.path.join(PROJECT_ROOT, "data", "training_sets")

CURRENT_DATASET_PATH = os.path.join(TRAINING_SETS_DIR, "dataset_v_current.csv")
NEW_DATASET_PATH = os.path.join(TRAINING_SETS_DIR, "dataset_v_new.csv")

def merge_data():
    print("="*60)
    print("BƯỚC : GỘP DỮ LIỆU LOG MỚI VÀO DATASET HUẤN LUYỆN")
    print("="*60)
    
    # 1. Tìm file log của ngày hôm nay (Do realtime_defender.py sinh ra)
    today_str = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    daily_log_path = os.path.join(DAILY_LOGS_DIR, f"log_{today_str}.csv")
    
    if not os.path.exists(daily_log_path):
        print(f"LỖI: Không tìm thấy file log ngày hôm nay tại: {daily_log_path}")
        #print("Hãy chắc chắn Nginx có log và realtime_defender.py đã chạy để sinh file csv này.")
        sys.exit(1)
        
    print(f"Đã tìm thấy log hôm nay: log_{today_str}.csv")
    df_daily = pd.read_csv(daily_log_path)
    print(f"   -> Số lượng request thu thập được trong ngày: {len(df_daily):,} dòng")
    
    # 2. Đọc file dataset hiện tại (Bản dùng để train lần trước)
    if not os.path.exists(CURRENT_DATASET_PATH):
        print(f"LỖI: Không tìm thấy dataset hiện tại tại: {CURRENT_DATASET_PATH}")
        #print("Mẹo: Hãy chắc chắn bạn đã chạy setup_phase1.py để tạo file này.")
        sys.exit(1)
        
    print(f"\nĐang đọc Dataset Master hiện tại: dataset_v_current.csv")
    df_current = pd.read_csv(CURRENT_DATASET_PATH)
    print(f"   -> Số lượng data cũ: {len(df_current):,} dòng")
    
    # 3. Gộp dữ liệu
    print("\nĐang gộp dữ liệu (Merge & Concat)...")
    df_new = pd.concat([df_current, df_daily], ignore_index=True)
    
    # Dọn dẹp: Đảm bảo không bị trùng lặp log nếu lỡ chạy script này 2 lần trong 1 ngày
    original_len = len(df_new)
    if 'request_id' in df_new.columns:
        df_new = df_new.drop_duplicates(subset=['request_id'], keep='last')
        
    dupe_count = original_len - len(df_new)
    if dupe_count > 0:
        print(f"   -> Đã xóa {dupe_count:,} dòng bị trùng lặp.")
        
    print(f"Gộp thành công! Tổng kích thước Dataset mới: {len(df_new):,} dòng")
    
    # 4. In thông kê nhãn (Label)
    if 'label' in df_new.columns:
        labels = df_new['label'].value_counts()
        print(f"\nTHỐNG KÊ NHÃN TRONG DATASET MỚI:")
        print(f"   Bình thường (0): {labels.get(0, 0):,}")
        print(f"   Tấn công (1)   : {labels.get(1, 0):,}")

    # 5. Lưu ra file version mới
    df_new.to_csv(NEW_DATASET_PATH, index=False)
    print(f"\nĐã lưu Dataset mới (Sẵn sàng cho Train) tại:\n   {NEW_DATASET_PATH}")

if __name__ == "__main__":
    merge_data()