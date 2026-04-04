import pandas as pd
import joblib
import time
import os
import sys
from datetime import datetime, timedelta, timezone
from elasticsearch import Elasticsearch
from colorama import init, Fore, Style

# Khởi tạo màu sắc cho Terminal
init(autoreset=True)

# ==========================================
# CẤU HÌNH ĐƯỜNG DẪN MÔI TRƯỜNG
# ==========================================
# Script nằm trong thư mục /run. Ta lùi lại 1 cấp (..) để ra thư mục gốc
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT) 

# Import module từ thư mục /src (Sau khi đã thêm PROJECT_ROOT vào path)
from src.features.common_features import build_features

# ==========================================
# CẤU HÌNH HỆ THỐNG
# ==========================================
ES_URL = "http://127.0.0.1:9200"
INDEX_NAME = "mlops-api-logs-*"

# Tự động trỏ đường dẫn tới file pkl nằm trong thư mục /models của dự án
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "production", "active_model.pkl")
#MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "lightgbm_threatAPI_detector.pkl")
# Tần suất quét (Ví dụ: 5 giây quét 1 lần)
POLLING_INTERVAL_SEC = 5
# Cửa sổ thời gian lùi lại để tính toán tính năng Rolling (30 giây cho an toàn)
CONTEXT_WINDOW_MINUTES = 0.5

def connect_elasticsearch():
    try:
        es = Elasticsearch(
            ES_URL,
            request_timeout=10,
            headers={"Accept": "application/vnd.elasticsearch+json; compatible-with=8",
                     "Content-Type": "application/vnd.elasticsearch+json; compatible-with=8"}
        )
        if es.info():
            print(Fore.GREEN + "Đã kết nối thành công tới Elasticsearch!")
            return es
    except Exception as e:
        print(Fore.RED + f"Lỗi kết nối ES: {e}")
        sys.exit(1)

def load_ai_model():
    if not os.path.exists(MODEL_PATH):
        print(Fore.RED + f"Không tìm thấy Model tại {MODEL_PATH}")
        sys.exit(1)
    model = joblib.load(MODEL_PATH)
    print(Fore.GREEN + f"Đã tải thành công 'Bộ não' LightGBM!")
    return model

def run_realtime_defender():
    print(Fore.CYAN + Style.BRIGHT + "="*60)
    print(Fore.CYAN + Style.BRIGHT + "HỆ THỐNG API THREAT DEFENDER ĐANG HOẠT ĐỘNG (PHASE 2)...")
    print(Fore.CYAN + Style.BRIGHT + "="*60)

    es = connect_elasticsearch()
    model = load_ai_model()
    
    # Tập hợp lưu trữ các request_id đã được quét để không cảnh báo trùng lặp
    processed_request_ids = set()

    while True:
        try:
            # 1. Xác định khung thời gian truy vấn
            now_utc = datetime.now(timezone.utc)
            start_time = now_utc - timedelta(minutes=CONTEXT_WINDOW_MINUTES)
            
            start_time_str = start_time.strftime('%Y-%m-%dT%H:%M:%S.000Z')
            end_time_str = now_utc.strftime('%Y-%m-%dT%H:%M:%S.000Z')

            # Câu truy vấn lấy log trong khung thời gian
            query = {
                "query": {
                    "range": {
                        "@timestamp": {
                            "gte": start_time_str,
                            "lte": end_time_str
                        }
                    }
                },
                "sort": [{"@timestamp": {"order": "asc"}}],
                "size": 5000 # Giới hạn lấy 5000 log gần nhất
            }

            response = es.search(index=INDEX_NAME, body=query)
            hits = response['hits']['hits']

            if len(hits) == 0:
                print(Fore.YELLOW + f"[{now_utc.strftime('%H:%M:%S')}] Không có traffic mới...")
                time.sleep(POLLING_INTERVAL_SEC)
                continue

            # 2. Chuyển đổi dữ liệu ES thành DataFrame
            raw_logs = [hit['_source'] for hit in hits]
            df = pd.DataFrame(raw_logs)

            # Xóa log trùng lặp
            df = df.drop_duplicates(subset=['request_id'], keep='last')

            # Lọc ra NHỮNG DÒNG LOG MỚI TINH
            new_logs_mask = ~df['request_id'].isin(processed_request_ids)
            if not new_logs_mask.any():
                time.sleep(POLLING_INTERVAL_SEC)
                continue

            # 3. Trích xuất Đặc trưng
            try:
                X, _ = build_features(df)
            except Exception as e:
                print(Fore.RED + f"Lỗi Feature Engineering: {e}")
                time.sleep(POLLING_INTERVAL_SEC)
                continue

            # Lấy vector đặc trưng mới. Dùng .copy() để tránh cảnh báo SettingWithCopyWarning
            X_new = X[new_logs_mask]
            df_new = df[new_logs_mask].copy() 

            # =========================================================
            # 4. AI CHẤM ĐIỂM (PROBABILITY SCORING)
            # =========================================================
            # Lấy mảng xác suất dự đoán là Tấn công (cột index 1)
            probabilities = model.predict_proba(X_new)[:, 1]

            labels = []
            attack_count = 0

            # =========================================================
            # 5. PHÂN LOẠI, GẮN NHÃN VÀ XUẤT FILE CSV
            # =========================================================
            for i in range(len(probabilities)):
                score = probabilities[i]
                req_id = df_new.iloc[i]['request_id']
                ip = df_new.iloc[i].get('remote_ip', 'Unknown IP')
                path = df_new.iloc[i].get('path', '/')
                method = df_new.iloc[i].get('method', 'GET')
                
                processed_request_ids.add(req_id)
                
                # Logic gán Label theo mức độ rủi ro (Risk Score)
                if score >= 0.85:
                    attack_count += 1
                    labels.append(1)
                    print(Fore.RED + Style.BRIGHT + f"[HIGH] Score: {score:.2f} | IP: {ip:<15} | Method: {method:<4} | Path: {path} | request_id: {req_id}")
                elif score >= 0.5:
                    labels.append(0)
                    print(Fore.YELLOW + f"[MEDIUM] Score: {score:.2f} | IP: {ip:<15} | Method: {method:<4} | Path: {path} | request_id: {req_id}")
                else:
                    labels.append(0)
                    print(Fore.BLUE + f"[LOW] Score: {score:.2f} | IP: {ip:<15} | Method: {method:<4} | Path: {path} | request_id: {req_id}")

            # Gắn trực tiếp mảng nhãn 0/1 vào DataFrame
            df_new['label'] = labels
            
            # ---------------------------------------------------------
            # CHUẨN HÓA DATAFRAME TRƯỚC KHI XUẤT CSV (GIẢI QUYẾT ĐỊNH DẠNG)
            # ---------------------------------------------------------
            TARGET_COLUMNS = [
                "@timestamp", "auth_token_hash", "method", "path", "path_normalized",
                "remote_ip", "request_id", "response_size", "response_time_ms",
                "sampling_flag", "status", "upstream", "user_agent", "user_id_hash",
                "user_role", "waf_action", "waf_rule_id", "label"
            ]

            # 1. Đảm bảo có đủ cột (nếu Elasticsearch thiếu thì điền rỗng)
            for col in TARGET_COLUMNS:
                if col not in df_new.columns:
                    df_new[col] = ""

            # 2. Xử lý giá trị rỗng
            df_new['waf_action'] = df_new['waf_action'].replace('', '0').fillna('0')
            df_new['waf_rule_id'] = df_new['waf_rule_id'].replace('', '0').fillna('0')
            df_new['sampling_flag'] = df_new['sampling_flag'].replace('', '0').fillna('0')
            df_new['user_role'] = df_new['user_role'].replace('', 'GUEST').fillna('GUEST')
            df_new['auth_token_hash'] = df_new['auth_token_hash'].replace('', '(empty)').fillna('(empty)')
            df_new['user_id_hash'] = df_new['user_id_hash'].replace('', '(empty)').fillna('(empty)')

            # 3. Ép kiểu số thực (Float)
            df_new['response_size'] = pd.to_numeric(df_new['response_size'], errors='coerce').fillna(0).astype(float)
            df_new['response_time_ms'] = pd.to_numeric(df_new['response_time_ms'], errors='coerce').fillna(0).astype(float)

            # 4. Định dạng lại thời gian
            df_new['@timestamp'] = pd.to_datetime(df_new['@timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S+00:00')

            # 5. Lọc bỏ các cột rác và sắp xếp đúng thứ tự yêu cầu
            df_export = df_new[TARGET_COLUMNS]
            
            # ---------------------------------------------------------
            # Cấu hình lưu file CSV hàng ngày
            today_str = datetime.now(timezone.utc).strftime('%Y-%m-%d')
            daily_log_dir = os.path.join(PROJECT_ROOT, "data", "daily_logs")
            os.makedirs(daily_log_dir, exist_ok=True)
            
            daily_csv_path = os.path.join(daily_log_dir, f"log_{today_str}.csv")
            
            # Nếu file chưa tồn tại thì ghi Header, nếu có rồi thì ghi tiếp (append mode)
            header_needed = not os.path.exists(daily_csv_path)
            df_export.to_csv(daily_csv_path, mode='a', index=False, header=header_needed)

            # Xóa bớt cache ID để tránh tràn RAM
            if len(processed_request_ids) > 10000:
                processed_request_ids = set(list(processed_request_ids)[-5000:])

            if attack_count == 0:
                print(Fore.GREEN + f"[{now_utc.strftime('%H:%M:%S')}] Đã quét {len(df_new)} reqs. Đã lưu log vào file ngày {today_str} (An toàn).")
            else:
                print(Fore.RED + f"[{now_utc.strftime('%H:%M:%S')}] Đã quét {len(df_new)} reqs. Cảnh báo: {attack_count} truy cập rủi ro cao!")

        except Exception as e:
            print(Fore.RED + f"Lỗi hệ thống trong lúc quét: {e}")
        
        time.sleep(POLLING_INTERVAL_SEC)

if __name__ == "__main__":
    run_realtime_defender()