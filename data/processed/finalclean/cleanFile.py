import pandas as pd
import os

# ==========================================
# ⚙️ CẤU HÌNH ĐƯỜNG DẪN FILE
# ==========================================
INPUT_FILE = r"n:/hodaoty/API-Threat-Detection/data/processed/mergeandsort/master_dataset_sorted.csv"
OUTPUT_FILE = r"n:/hodaoty/API-Threat-Detection/data/processed/finalclean/master_dataset_cleaned.csv"

def clean_empty_values():
    print(f"BƯỚC 1: Đang đọc dữ liệu từ:\n   {INPUT_FILE}")
    if not os.path.exists(INPUT_FILE):
        print("LỖI: Không tìm thấy file đầu vào. Vui lòng kiểm tra lại đường dẫn!")
        return
        
    df = pd.read_csv(INPUT_FILE, dtype=str) # Đọc tất cả dưới dạng chuỗi để dễ xử lý replace
    
    print(f"Đã tải xong {len(df):,} dòng dữ liệu.")
    
    # --- THỐNG KÊ TRƯỚC KHI XỬ LÝ ---
    print("\nKiểm tra sự tồn tại của '(empty)' trước khi xử lý:")
    empty_counts = (df == '(empty)').sum()
    cols_with_empty = empty_counts[empty_counts > 0]
    if not cols_with_empty.empty:
        for col, count in cols_with_empty.items():
            print(f"   Cột '{col}' chứa {count:,} giá trị '(empty)'")
    else:
        print("   Không tìm thấy giá trị '(empty)' nào trong dataset.")

    print("\nBƯỚC 2: TIẾN HÀNH XỬ LÝ VÀ THAY THẾ...")
    
    # 1. Xử lý cột user_role: Thay '(empty)' hoặc NaN thành 'GUEST'
    if 'user_role' in df.columns:
        df['user_role'] = df['user_role'].replace('(empty)', 'GUEST')
        df['user_role'] = df['user_role'].fillna('GUEST')
        # Chuẩn hóa luôn các giá trị rỗng hoặc khoảng trắng thành GUEST
        df.loc[df['user_role'].str.strip() == '', 'user_role'] = 'GUEST'
        print("   -> Đã chuẩn hóa cột 'user_role' thành 'GUEST'.")

    # 2. Xử lý các cột WAF và các cột số học: Thay '(empty)' thành '0'
    columns_to_zero = ['waf_action', 'waf_rule_id', 'sampling_flag', 'response_size', 'response_time_ms']
    for col in columns_to_zero:
        if col in df.columns:
            df[col] = df[col].replace('(empty)', '0')
            df[col] = df[col].fillna('0')
            # Quét sạch cả các ô trống trơn
            df.loc[df[col].str.strip() == '', col] = '0'
            print(f"   -> Đã chuẩn hóa cột '{col}' thành '0'.")

    # 3. Ép kiểu dữ liệu (Data Type Casting) để chuẩn bị cho AI
    print("\nBƯỚC 3: ÉP KIỂU DỮ LIỆU SỐ HỌC...")
    df['response_size'] = pd.to_numeric(df['response_size'], errors='coerce').fillna(0)
    df['response_time_ms'] = pd.to_numeric(df['response_time_ms'], errors='coerce').fillna(0)
    df['status'] = pd.to_numeric(df['status'], errors='coerce').fillna(200).astype(int)
    df['label'] = pd.to_numeric(df['label'], errors='coerce').fillna(0).astype(int)
    
    # --- THỐNG KÊ SAU KHI XỬ LÝ ---
    print("\nKiểm tra lại sau khi xử lý:")
    empty_counts_after = (df.astype(str) == '(empty)').sum().sum()
    if empty_counts_after == 0:
        print("   Đã quét sạch 100% giá trị '(empty)' ra khỏi dataset!")
    else:
        print(f"   Vẫn còn sót lại {empty_counts_after} giá trị '(empty)' ở đâu đó.")

    # 4. Lưu ra file mới
    print("\nBƯỚC 4: LƯU FILE KẾT QUẢ...")
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"THÀNH CÔNG! File dữ liệu SẠCH HOÀN TOÀN đã được lưu tại:\n   {OUTPUT_FILE}")

if __name__ == "__main__":
    clean_empty_values()