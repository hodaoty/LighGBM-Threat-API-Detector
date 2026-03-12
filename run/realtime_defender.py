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
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "lightgbm_threatAPI_detector.pkl")

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
    print(Fore.CYAN + Style.BRIGHT + "HỆ THỐNG API THREAT DEFENDER ĐANG HOẠT ĐỘNG...")
    print(Fore.CYAN + Style.BRIGHT + "="*60)

    es = connect_elasticsearch()
    model = load_ai_model()
    
    # Tập hợp lưu trữ các request_id đã được quét để không cảnh báo trùng lặp
    processed_request_ids = set()

    while True:
        try:
            # 1. Xác định khung thời gian truy vấn (Từ [Bây giờ - 1.5 phút] đến [Bây giờ])
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
                "size": 5000 # Giới hạn lấy 5000 log gần nhất (để tránh tràn RAM)
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

            # Xóa log trùng lặp (do Logstash đẩy trùng)
            df = df.drop_duplicates(subset=['request_id'], keep='last')

            # Lọc ra NHỮNG DÒNG LOG MỚI TINH (Chưa từng dự đoán)
            new_logs_mask = ~df['request_id'].isin(processed_request_ids)
            if not new_logs_mask.any():
                time.sleep(POLLING_INTERVAL_SEC)
                continue

            # 3. Trích xuất Đặc trưng (Đưa cả context 1.5 phút vào để tính toán)
            try:
                # Gọi hàm build_features mà không cần cột label
                X, _ = build_features(df)
            except Exception as e:
                print(Fore.RED + f"Lỗi Feature Engineering: {e}")
                time.sleep(POLLING_INTERVAL_SEC)
                continue

            # Chỉ lấy các vector đặc trưng của những dòng log MỚI để AI phán xét
            X_new = X[new_logs_mask]
            df_new = df[new_logs_mask]

            # 4. AI Đưa ra phán quyết
            predictions = model.predict(X_new)

            # 5. Phân tích kết quả và hiển thị
            attack_count = 0
            for i in range(len(predictions)):
                req_id = df_new.iloc[i]['request_id']
                ip = df_new.iloc[i].get('remote_ip', 'Unknown IP')
                path = df_new.iloc[i].get('path', '/')
                method = df_new.iloc[i].get('method', 'GET')
                
                # Thêm vào danh sách đã xử lý
                processed_request_ids.add(req_id)
                
                if predictions[i] == 1:
                    attack_count += 1
                    # CẢNH BÁO MÀU ĐỎ KHI CÓ TẤN CÔNG
                    print(Fore.RED + Style.BRIGHT + f"[ATTACK DETECTED] IP: {ip:<15} | Method: {method:<4} | Path: {path}")

            # Xóa bớt cache ID để tránh tràn RAM (giữ lại 10000 ID gần nhất)
            if len(processed_request_ids) > 10000:
                processed_request_ids = set(list(processed_request_ids)[-5000:])

            if attack_count == 0:
                print(Fore.GREEN + f"[{now_utc.strftime('%H:%M:%S')}] Đã quét {len(df_new)} requests mới. Hệ thống an toàn.")

        except Exception as e:
            print(Fore.RED + f"Lỗi hệ thống trong lúc quét: {e}")
        
        # Ngủ 5 giây rồi quét tiếp
        time.sleep(POLLING_INTERVAL_SEC)

if __name__ == "__main__":
    run_realtime_defender()