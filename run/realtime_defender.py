import pandas as pd
import joblib
import time
import os
import sys
import requests
import hashlib
from datetime import datetime, timedelta, timezone
from elasticsearch import Elasticsearch
from colorama import init, Fore, Style

# Initialize terminal colors
init(autoreset=True)

# ==========================================
# ENVIRONMENT PATH CONFIGURATION
# ==========================================
# Script is located in /src/models. Go back one level to the project root.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT) 

# Import module from /src (After appending PROJECT_ROOT to path)
from src.features.common_features import build_features

# ==========================================
# SYSTEM & ELASTICSEARCH CONFIGURATION
# ==========================================
ES_URL = "http://127.0.0.1:9200"
INDEX_NAME = "mlops-api-logs-*"

# Path to the active .pkl model file in the production directory
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "production", "active_model.pkl")

# Polling interval (e.g., scan every 5 seconds)
POLLING_INTERVAL_SEC = 5
# Time window for rolling features (e.g., 0.5 minutes for safety margin)
CONTEXT_WINDOW_MINUTES = 0.5

# ==========================================
# FIREWALL ADMIN COMMUNICATION CONFIGURATION
# ==========================================
FIREWALL_API_URL = "http://127.0.0.1:9000/api/autoban"
FW_ADMIN_USER = "admin"
FW_ADMIN_PASS = "admin123"

# Calculate session token for Firewall Admin authentication
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
            print(Fore.GREEN + "Successfully connected to Elasticsearch!")
            return es
    except Exception as e:
        print(Fore.RED + f"Elasticsearch connection error: {e}")
        sys.exit(1)

def load_ai_model():
    if not os.path.exists(MODEL_PATH):
        print(Fore.RED + f"Model not found at {MODEL_PATH}")
        sys.exit(1)
    model = joblib.load(MODEL_PATH)
    print(Fore.GREEN + "LightGBM model loaded successfully!")
    return model

def trigger_firewall_ban(ip):
    """Call Firewall Admin API to auto-ban IP"""
    if ip == "172.20.0.10": ip = "172.20.0.1"
    try:
        response = requests.post(
            FIREWALL_API_URL,
            json={"ip": ip},
            cookies={"session": FW_AUTH_TOKEN},
            timeout=2
        )
        if response.status_code == 200:
            print(Fore.MAGENTA + Style.BRIGHT + f"   -> [FIREWALL] Automatically blocked IP {ip} in Auto-ban list!")
        else:
            print(Fore.YELLOW + f"   -> [FIREWALL] Error blocking IP {ip}. Status: {response.status_code}")
    except Exception as e:
        print(Fore.YELLOW + f"   -> [FIREWALL] Cannot connect to Firewall Admin: {e}")

def run_realtime_defender():
    print(Fore.CYAN + Style.BRIGHT + "="*60)
    print(Fore.CYAN + Style.BRIGHT + "API THREAT DEFENDER & AUTO-RESPONSE SYSTEM IS RUNNING...")
    print(Fore.CYAN + Style.BRIGHT + "="*60)

    es = connect_elasticsearch()
    model = load_ai_model()
    
    # Set to store processed request_ids to avoid duplicate alerts
    processed_request_ids = set()
    
    # Cache for recently banned IPs to avoid spamming the Firewall API
    recently_banned_ips = set()

    while True:
        try:
            # 1. Define query time frame
            now_utc = datetime.now(timezone.utc)
            start_time = now_utc - timedelta(minutes=CONTEXT_WINDOW_MINUTES)
            
            start_time_str = start_time.strftime('%Y-%m-%dT%H:%M:%S.000Z')
            end_time_str = now_utc.strftime('%Y-%m-%dT%H:%M:%S.000Z')

            # Query to fetch logs within the time frame
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
                "size": 5000 # Limit to 5000 recent logs
            }

            response = es.search(index=INDEX_NAME, body=query)
            hits = response['hits']['hits']

            if len(hits) == 0:
                print(Fore.YELLOW + f"[{now_utc.strftime('%H:%M:%S')}] No new traffic...")
                time.sleep(POLLING_INTERVAL_SEC)
                continue

            # 2. Convert ES data to DataFrame
            raw_logs = [hit['_source'] for hit in hits]
            df = pd.DataFrame(raw_logs)

            # Remove duplicate logs
            df = df.drop_duplicates(subset=['request_id'], keep='last')

            # Filter out NEW LOGS only
            new_logs_mask = ~df['request_id'].isin(processed_request_ids)
            if not new_logs_mask.any():
                time.sleep(POLLING_INTERVAL_SEC)
                continue

            # 3. Feature Extraction
            try:
                X, _ = build_features(df)
            except Exception as e:
                print(Fore.RED + f"Feature Engineering error: {e}")
                time.sleep(POLLING_INTERVAL_SEC)
                continue

            # Get new feature vectors. Use .copy() to avoid SettingWithCopyWarning
            X_new = X[new_logs_mask]
            df_new = df[new_logs_mask].copy() 

            # =========================================================
            # 4. AI PROBABILITY SCORING
            # =========================================================
            # Get attack probability array (column index 1)
            probabilities = model.predict_proba(X_new)[:, 1]

            labels = []
            attack_count = 0

            # =========================================================
            # 5. CLASSIFICATION, AUTO-BAN AND CSV EXPORT
            # =========================================================
            for i in range(len(probabilities)):
                score = probabilities[i]
                req_id = df_new.iloc[i]['request_id']
                ip = df_new.iloc[i].get('remote_ip', 'Unknown IP')
                path = df_new.iloc[i].get('path', '/')
                method = df_new.iloc[i].get('method', 'GET')
                
                processed_request_ids.add(req_id)
                
                # Label assignment logic based on Risk Score
                if score >= 0.85:
                    attack_count += 1
                    labels.append(1)
                    print(Fore.RED + Style.BRIGHT + f"[HIGH] Score: {score:.2f} | IP: {ip:<15} | Method: {method:<4} | Path: {path} | request_id: {req_id}")
                    
                    # --- COMMAND FIREWALL TO BAN IP ---
                    if ip not in recently_banned_ips and ip != "Unknown IP":
                        trigger_firewall_ban(ip)
                        recently_banned_ips.add(ip)
                        
                elif score >= 0.5:
                    labels.append(0)
                    print(Fore.YELLOW + f"[MEDIUM] Score: {score:.2f} | IP: {ip:<15} | Method: {method:<4} | Path: {path} | request_id: {req_id}")
                else:
                    labels.append(0)
                    print(Fore.BLUE + f"[LOW] Score: {score:.2f} | IP: {ip:<15} | Method: {method:<4} | Path: {path} | request_id: {req_id}")

            # Attach 0/1 labels to DataFrame
            df_new['label'] = labels
            
            # ---------------------------------------------------------
            # NORMALIZE DATAFRAME BEFORE CSV EXPORT (FORMAT RESOLUTION)
            # ---------------------------------------------------------
            TARGET_COLUMNS = [
                "@timestamp", "auth_token_hash", "method", "path", "path_normalized",
                "remote_ip", "request_id", "response_size", "response_time_ms",
                "sampling_flag", "status", "upstream", "user_agent", "user_id_hash",
                "user_role", "waf_action", "waf_rule_id", "label"
            ]

            # 1. Ensure all columns exist (fill empty if ES is missing any)
            for col in TARGET_COLUMNS:
                if col not in df_new.columns:
                    df_new[col] = ""

            # 2. Handle empty values
            df_new['waf_action'] = df_new['waf_action'].replace('', '0').fillna('0')
            df_new['waf_rule_id'] = df_new['waf_rule_id'].replace('', '0').fillna('0')
            df_new['sampling_flag'] = df_new['sampling_flag'].replace('', '0').fillna('0')
            df_new['user_role'] = df_new['user_role'].replace('', 'GUEST').fillna('GUEST')
            df_new['auth_token_hash'] = df_new['auth_token_hash'].replace('', '(empty)').fillna('(empty)')
            df_new['user_id_hash'] = df_new['user_id_hash'].replace('', '(empty)').fillna('(empty)')

            # 3. Cast to Float
            df_new['response_size'] = pd.to_numeric(df_new['response_size'], errors='coerce').fillna(0).astype(float)
            df_new['response_time_ms'] = pd.to_numeric(df_new['response_time_ms'], errors='coerce').fillna(0).astype(float)

            # 4. Reformat timestamp
            df_new['@timestamp'] = pd.to_datetime(df_new['@timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S+00:00')

            # 5. Filter out junk columns and sort by required order
            df_export = df_new[TARGET_COLUMNS]
            
            # ---------------------------------------------------------
            # Daily CSV file configuration
            today_str = datetime.now(timezone.utc).strftime('%Y-%m-%d')
            daily_log_dir = os.path.join(PROJECT_ROOT, "data", "daily_logs")
            os.makedirs(daily_log_dir, exist_ok=True)
            
            daily_csv_path = os.path.join(daily_log_dir, f"log_{today_str}.csv")
            
            # Write Header if file doesn't exist, otherwise append
            header_needed = not os.path.exists(daily_csv_path)
            df_export.to_csv(daily_csv_path, mode='a', index=False, header=header_needed)

            # Clear ID cache to prevent RAM overflow
            if len(processed_request_ids) > 10000:
                processed_request_ids = set(list(processed_request_ids)[-5000:])
                
            # Clear banned IP cache to free memory
            if len(recently_banned_ips) > 1000:
                recently_banned_ips.clear()

            if attack_count == 0:
                print(Fore.GREEN + f"[{now_utc.strftime('%H:%M:%S')}] Scanned {len(df_new)} reqs. Logs saved to {today_str} file (Safe).")
            else:
                print(Fore.RED + f"[{now_utc.strftime('%H:%M:%S')}] Scanned {len(df_new)} reqs. Warning: {attack_count} high-risk requests processed!")

        except Exception as e:
            print(Fore.RED + f"System error during scan: {e}")
        
        time.sleep(POLLING_INTERVAL_SEC)

if __name__ == "__main__":
    run_realtime_defender()