import pandas as pd
import joblib
import time
import os
import sys
import requests
import hashlib
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta, timezone
from elasticsearch import Elasticsearch
from colorama import init, Fore, Style
from dotenv import load_dotenv

# Initialize terminal colors
init(autoreset=True)

# ==========================================
# ENVIRONMENT PATH CONFIGURATION
# ==========================================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT) 

from src.features.common_features import build_features

# ==========================================
# LOAD ENVIRONMENT VARIABLES (.env)
# ==========================================
load_dotenv(os.path.join(PROJECT_ROOT, '.env'))

GMAIL_USER = os.getenv("GMAIL_USER")
GMAIL_APP_PASSWORD = os.getenv("GMAIL_APP_PASSWORD")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# Kibana Discover URL template updated with the user's specific index and layout
# SỬA LỖI: Đổi 'localhost' thành '127.0.0.1' để Telegram API không chặn URL
KIBANA_URL_TEMPLATE = "http://127.0.0.1:5601/app/discover#/?_g=(filters:!(),refreshInterval:(pause:!t,value:60000),time:(from:now-30d%2Fd,to:now))&_a=(columns:!(),filters:!(),index:bcb4c7e3-e358-4578-ac1f-0364892a565a,interval:auto,query:(language:kuery,query:'request_id%20:%20{req_id}'),sort:!(!('@timestamp',desc)))"

# ==========================================
# SYSTEM & ELASTICSEARCH CONFIGURATION
# ==========================================
ES_URL = "http://127.0.0.1:9200"
INDEX_NAME = "mlops-api-logs-*"
#MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "production", "active_model.pkl")
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "lightgbm_threatAPI_detector.pkl")

POLLING_INTERVAL_SEC = 5
CONTEXT_WINDOW_MINUTES = 0.5

# ==========================================
# FIREWALL ADMIN COMMUNICATION CONFIGURATION
# ==========================================
FIREWALL_API_URL = "http://127.0.0.1:9000/api/autoban"
FW_ADMIN_USER = "admin"
FW_ADMIN_PASS = "admin123"
FW_AUTH_TOKEN = hashlib.sha256(f"{FW_ADMIN_USER}:{FW_ADMIN_PASS}_secure_firewall".encode()).hexdigest()


def connect_elasticsearch():
    try:
        es = Elasticsearch(
            ES_URL,
            request_timeout=10,
            headers={"Accept": "application/vnd.elasticsearch+json; compatible-with=8",
                     "Content-Type": "application/vnd.elasticsearch+json; compatible-with=8"}
        )
        if es.info():
            print(Fore.GREEN + "Successfully connected to Elasticsearch.")
            return es
    except Exception as e:
        print(Fore.RED + f"Elasticsearch connection error: {e}")
        sys.exit(1)

def load_ai_model():
    if not os.path.exists(MODEL_PATH):
        print(Fore.RED + f"Model not found at {MODEL_PATH}")
        sys.exit(1)
    model = joblib.load(MODEL_PATH)
    print(Fore.GREEN + "LightGBM model loaded successfully.")
    return model

def trigger_firewall_ban(ip, req_id):
    """Call Firewall Admin API to auto-ban IP with request_id metadata"""
    try:
        response = requests.post(
            FIREWALL_API_URL,
            json={"ip": ip, "request_id": req_id},
            cookies={"session": FW_AUTH_TOKEN},
            timeout=2
        )
        if response.status_code == 200:
            print(Fore.MAGENTA + Style.BRIGHT + f"   -> [FIREWALL] Automatically blocked IP {ip}")
        else:
            print(Fore.YELLOW + f"   -> [FIREWALL] Error blocking IP {ip}. Status: {response.status_code}")
    except Exception as e:
        print(Fore.YELLOW + f"   -> [FIREWALL] Cannot connect to Firewall Admin: {e}")

def send_email_alert(ip, req_id, score, risk_level="HIGH"):
    """Send an email alert to the admin via Gmail SMTP"""
    if not GMAIL_USER or not GMAIL_APP_PASSWORD:
        return
        
    try:
        msg = MIMEMultipart()
        msg['From'] = GMAIL_USER
        msg['To'] = GMAIL_USER  
        msg['Subject'] = f"[{risk_level} RISK] API Threat Detected: {ip}"
        
        if risk_level == "HIGH":
            action_text = "Added to Auto-ban list."
        else:
            action_text = "System HAS NOT blocked this IP. Admin review required."
            
        body = f"""
SECURITY ALERT - SYSTEM AUTOMATED RESPONSE

The AI model has detected a {risk_level} risk attack.

Incident Details:
- IP Address: {ip}
- Threat Score: {score:.4f}
- Request ID: {req_id}
- Action Taken: {action_text}

Please review the Kibana dashboard for detailed traffic analysis.
        """
        msg.attach(MIMEText(body, 'plain'))
        
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(GMAIL_USER, GMAIL_APP_PASSWORD)
        server.send_message(msg)
        server.quit()
        
        print(Fore.CYAN + f"   -> [EMAIL] Alert sent successfully for {ip}")
    except Exception as e:
        print(Fore.YELLOW + f"   -> [EMAIL] Failed to send alert: {e}")

def send_telegram_alert(ip, req_id, score, risk_level="HIGH"):
    """Send a Telegram alert with interactive inline keyboard"""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
        
    try:
        kibana_url = KIBANA_URL_TEMPLATE.replace("{req_id}", str(req_id))
        
        if risk_level == "HIGH":
            action_taken = "IP Auto-banned"
            buttons = [
                [{"text": "View on Kibana", "url": kibana_url}],
                [{"text": "False Positive (Unblock)", "callback_data": f"unblock|{ip}|{req_id}"}]
            ]
        else:
            action_taken = "Pending Admin Review"
            buttons = [
                [{"text": "View on Kibana", "url": kibana_url}],
                [{"text": "True Positive (Block IP)", "callback_data": f"block|{ip}|{req_id}"}]
            ]
            
        text = (
            f"*[{risk_level} RISK] AI SECURITY ALERT*\n\n"
            f"*Action:* `{action_taken}`\n"
            f"*Source IP:* `{ip}`\n"
            f"*Threat Score:* `{score:.4f}`\n"
            f"*Request ID:* `{req_id}`"
        )
        
        reply_markup = {"inline_keyboard": buttons}
        
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": text,
            "parse_mode": "Markdown",
            "reply_markup": reply_markup
        }
        
        response = requests.post(url, json=payload, timeout=5)
        if response.status_code == 200:
            print(Fore.CYAN + "   -> [TELEGRAM] Interactive alert sent successfully.")
        else:
            print(Fore.YELLOW + f"   -> [TELEGRAM] Failed to send alert. Status: {response.status_code}")
            print(Fore.YELLOW + f"   -> [TELEGRAM] Detail: {response.text}")
    except Exception as e:
        print(Fore.YELLOW + f"   -> [TELEGRAM] Error: {e}")

def run_realtime_defender():
    print(Fore.CYAN + Style.BRIGHT + "="*60)
    print(Fore.CYAN + Style.BRIGHT + "API THREAT DEFENDER & AUTO-RESPONSE SYSTEM IS RUNNING...")
    print(Fore.CYAN + Style.BRIGHT + "="*60)

    es = connect_elasticsearch()
    model = load_ai_model()
    
    processed_request_ids = set()
    recently_banned_ips = set()

    while True:
        try:
            now_utc = datetime.now(timezone.utc)
            start_time = now_utc - timedelta(minutes=CONTEXT_WINDOW_MINUTES)
            
            start_time_str = start_time.strftime('%Y-%m-%dT%H:%M:%S.000Z')
            end_time_str = now_utc.strftime('%Y-%m-%dT%H:%M:%S.000Z')

            query = {
                "query": {"range": {"@timestamp": {"gte": start_time_str, "lte": end_time_str}}},
                "sort": [{"@timestamp": {"order": "asc"}}],
                "size": 5000
            }

            response = es.search(index=INDEX_NAME, body=query)
            hits = response['hits']['hits']

            if len(hits) == 0:
                print(Fore.YELLOW + f"[{now_utc.strftime('%H:%M:%S')}] No new traffic...")
                time.sleep(POLLING_INTERVAL_SEC)
                continue

            raw_logs = [hit['_source'] for hit in hits]
            df = pd.DataFrame(raw_logs)

            df = df.drop_duplicates(subset=['request_id'], keep='last')

            new_logs_mask = ~df['request_id'].isin(processed_request_ids)
            if not new_logs_mask.any():
                time.sleep(POLLING_INTERVAL_SEC)
                continue

            try:
                X, _ = build_features(df)
            except Exception as e:
                print(Fore.RED + f"Feature Engineering error: {e}")
                time.sleep(POLLING_INTERVAL_SEC)
                continue

            X_new = X[new_logs_mask]
            df_new = df[new_logs_mask].copy() 

            probabilities = model.predict_proba(X_new)[:, 1]

            labels = []
            attack_count = 0

            for i in range(len(probabilities)):
                score = probabilities[i]
                req_id = df_new.iloc[i]['request_id']
                ip = df_new.iloc[i].get('remote_ip', 'Unknown IP')
                path = df_new.iloc[i].get('path', '/')
                method = df_new.iloc[i].get('method', 'GET')
                
                processed_request_ids.add(req_id)
                
                if score >= 0.85:
                    attack_count += 1
                    labels.append(1)
                    print(Fore.RED + Style.BRIGHT + f"[HIGH] Score: {score:.2f} | IP: {ip:<15} | Method: {method:<4} | Path: {path} | request_id: {req_id}")
                    
                    # --- COMMAND FIREWALL TO BAN IP ---
                    if ip not in recently_banned_ips and ip != "Unknown IP":
                        trigger_firewall_ban(ip, req_id)
                        
                        # --- SEND NOTIFICATIONS (HIGH) ---
                        send_email_alert(ip, req_id, score, "HIGH")
                        send_telegram_alert(ip, req_id, score, "HIGH")
                        
                        recently_banned_ips.add(ip)
                        
                elif score >= 0.5:
                    labels.append(0) # Giữ nguyên label 0 vì chưa chắc chắn
                    print(Fore.YELLOW + f"[MEDIUM] Score: {score:.2f} | IP: {ip:<15} | Method: {method:<4} | Path: {path} | request_id: {req_id}")
                    
                    # --- SEND NOTIFICATIONS (MEDIUM) ---
                    # Không gọi trigger_firewall_ban, chỉ gửi cảnh báo chờ duyệt
                    if ip not in recently_banned_ips and ip != "Unknown IP":
                        send_email_alert(ip, req_id, score, "MEDIUM")
                        send_telegram_alert(ip, req_id, score, "MEDIUM")
                        
                        # Thêm vào danh sách tạm để không spam tin nhắn liên tục
                        #recently_banned_ips.add(ip)
                        
                else:
                    labels.append(0)
                    print(Fore.BLUE + f"[LOW] Score: {score:.2f} | IP: {ip:<15} | Method: {method:<4} | Path: {path} | request_id: {req_id}")

            df_new['label'] = labels
            
            TARGET_COLUMNS = [
                "@timestamp", "auth_token_hash", "method", "path", "path_normalized",
                "remote_ip", "request_id", "response_size", "response_time_ms",
                "sampling_flag", "status", "upstream", "user_agent", "user_id_hash",
                "user_role", "waf_action", "waf_rule_id", "label"
            ]

            for col in TARGET_COLUMNS:
                if col not in df_new.columns:
                    df_new[col] = ""

            df_new['waf_action'] = df_new['waf_action'].replace('', '0').fillna('0')
            df_new['waf_rule_id'] = df_new['waf_rule_id'].replace('', '0').fillna('0')
            df_new['sampling_flag'] = df_new['sampling_flag'].replace('', '0').fillna('0')
            df_new['user_role'] = df_new['user_role'].replace('', 'GUEST').fillna('GUEST')
            df_new['auth_token_hash'] = df_new['auth_token_hash'].replace('', '(empty)').fillna('(empty)')
            df_new['user_id_hash'] = df_new['user_id_hash'].replace('', '(empty)').fillna('(empty)')

            df_new['response_size'] = pd.to_numeric(df_new['response_size'], errors='coerce').fillna(0).astype(float)
            df_new['response_time_ms'] = pd.to_numeric(df_new['response_time_ms'], errors='coerce').fillna(0).astype(float)

            df_new['@timestamp'] = pd.to_datetime(df_new['@timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S+00:00')

            df_export = df_new[TARGET_COLUMNS]
            
            today_str = datetime.now(timezone.utc).strftime('%Y-%m-%d')
            daily_log_dir = os.path.join(PROJECT_ROOT, "data", "daily_logs")
            os.makedirs(daily_log_dir, exist_ok=True)
            
            daily_csv_path = os.path.join(daily_log_dir, f"log_{today_str}.csv")
            
            header_needed = not os.path.exists(daily_csv_path)
            df_export.to_csv(daily_csv_path, mode='a', index=False, header=header_needed)

            if len(processed_request_ids) > 10000:
                processed_request_ids = set(list(processed_request_ids)[-5000:])
                
            if len(recently_banned_ips) > 1000:
                recently_banned_ips.clear()

            if attack_count == 0:
                print(Fore.GREEN + f"[{now_utc.strftime('%H:%M:%S')}] Scanned {len(df_new)} reqs. Logs saved. Status: Safe.")
            else:
                print(Fore.RED + f"[{now_utc.strftime('%H:%M:%S')}] Scanned {len(df_new)} reqs. Warning: {attack_count} high-risk requests processed.")

        except Exception as e:
            print(Fore.RED + f"System error during scan: {e}")
        
        time.sleep(POLLING_INTERVAL_SEC)

if __name__ == "__main__":
    run_realtime_defender()