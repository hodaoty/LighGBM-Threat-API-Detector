import pandas as pd
import os

# ==========================================
# ⚙️ CẤU HÌNH ĐƯỜNG DẪN FILE (Theo ổ N: của bạn)
# ==========================================
# Lưu ý: Vui lòng sửa lại tên file .csv ở cuối mỗi chuỗi nếu bạn đặt tên khác
FILES_TO_MERGE = [
    r"n:/hodaoty/API-Threat-Detection/data/internal/api_gateway_logs/api-gateway_cleaned.csv",
    r"n:/hodaoty/API-Threat-Detection/data/external/injection/injection_cleaned.csv",
    r"n:/hodaoty/API-Threat-Detection/data/internal/bola_logs/bola_cleaned.csv",         # Sửa tên file nếu cần
    r"n:/hodaoty/API-Threat-Detection/data/internal/fla_logs/bfla_cleaned.csv",
    r"n:/hodaoty/API-Threat-Detection/data/internal/rate_limiting_logs/rate_limiting_cleaned.csv"
]

# Đường dẫn lưu file tổng hợp
OUTPUT_DIR = r"n:/hodaoty/API-Threat-Detection/data/processed/mergeandsort"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "master_dataset_sorted.csv")

def merge_and_sort():
    print("BƯỚC 1: ĐANG ĐỌC VÀ GỘP CÁC FILE CSV...")
    df_list = []
    
    for file_path in FILES_TO_MERGE:
        if os.path.exists(file_path):
            print(f"  -> Đã tìm thấy và đọc: {os.path.basename(file_path)}")
            df = pd.read_csv(file_path)
            df_list.append(df)
        else:
            print(f"  [!] LỖI: Không tìm thấy file tại:\n      {file_path}\n      -> Vui lòng kiểm tra lại tên file. Bỏ qua...")

    if not df_list:
        print("Không có file nào được đọc thành công. Dừng chương trình.")
        return

    # Gộp tất cả DataFrames lại thành 1
    master_df = pd.concat(df_list, ignore_index=True)
    print(f"\nĐã gộp xong! Tổng số dòng sau khi gộp: {len(master_df):,} dòng.")

    print("\nBƯỚC 2: ĐANG CHUẨN HÓA VÀ SẮP XẾP THEO THỜI GIAN (@timestamp)...")
    # Chuyển đổi cột @timestamp sang định dạng Datetime chuẩn của Pandas
    master_df['@timestamp'] = pd.to_datetime(master_df['@timestamp'], errors='coerce')
    
    # Xóa các dòng bị lỗi thời gian (nếu có) trước khi sort
    master_df = master_df.dropna(subset=['@timestamp'])
    
    # Sắp xếp toàn bộ dữ liệu theo thứ tự thời gian từ cũ đến mới (cực kỳ quan trọng)
    master_df = master_df.sort_values(by='@timestamp', ascending=True)

    print("\nBƯỚC 3: ĐANG LƯU RA FILE TỔNG HỢP...")
    # Tự động tạo thư mục processed nếu chưa có
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Lưu ra file CSV
    master_df.to_csv(OUTPUT_FILE, index=False)
    
    print(f"THÀNH CÔNG! File tổng hợp đã được lưu tại:\n   {OUTPUT_FILE}")
    
    # --- IN THỐNG KÊ ĐỂ NGHIỆM THU ---
    print("\n" + "="*50)
    print("BẢNG THỐNG KÊ NHÃN (LABEL) TRONG FILE MASTER:")
    print("="*50)
    
    if 'label' in master_df.columns:
        labels = master_df['label'].value_counts()
        normal_count = labels.get(0, 0)
        attack_count = labels.get(1, 0)
        total_count = len(master_df)
        
        print(f"   Bình thường (0): {normal_count:,} dòng ({(normal_count/total_count)*100:.1f}%)")
        print(f"   Tấn công (1)   : {attack_count:,} dòng ({(attack_count/total_count)*100:.1f}%)")
        print(f"   TỔNG CỘNG      : {total_count:,} dòng")
    else:
        print("   Không tìm thấy cột 'label' trong dữ liệu.")
        
    print("-" * 50)
    print(f"⏱️ Dữ liệu bắt đầu từ: {master_df['@timestamp'].min()}")
    print(f"⏱️ Dữ liệu kết thúc ở: {master_df['@timestamp'].max()}")
    print("="*50)
    print("💡 BƯỚC TIẾP THEO: Dùng file master_dataset_sorted.csv này để chạy Feature Engineering!")

if __name__ == "__main__":
    merge_and_sort()