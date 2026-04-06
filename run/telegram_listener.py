import os
import sys
import requests
import hashlib
import pandas as pd
from datetime import datetime, timezone
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import ApplicationBuilder, CallbackQueryHandler, ContextTypes

# ==========================================
# ENVIRONMENT PATH CONFIGURATION
# ==========================================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)

# ==========================================
# LOAD ENVIRONMENT VARIABLES
# ==========================================
load_dotenv(os.path.join(PROJECT_ROOT, '.env'))
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

# Firewall Admin API Configuration
FIREWALL_API_URL = "http://127.0.0.1:9000/api/unblock"
FW_ADMIN_USER = "admin"
FW_ADMIN_PASS = "admin123"
FW_AUTH_TOKEN = hashlib.sha256(f"{FW_ADMIN_USER}:{FW_ADMIN_PASS}_secure_firewall".encode()).hexdigest()

DAILY_LOGS_DIR = os.path.join(PROJECT_ROOT, "data", "daily_logs")

def unblock_ip_firewall(ip: str) -> bool:
    """Call Firewall Admin API to unblock the IP."""
    try:
        response = requests.post(
            FIREWALL_API_URL,
            json={"ip": ip},
            cookies={"session": FW_AUTH_TOKEN},
            timeout=5
        )
        if response.status_code == 200:
            print(f"[FIREWALL] Successfully unblocked IP: {ip}")
            return True
        else:
            print(f"[FIREWALL] Failed to unblock IP: {ip}. Status: {response.status_code}")
            return False
    except Exception as e:
        print(f"[FIREWALL] API connection error: {e}")
        return False

def rollback_csv_label(req_id: str) -> bool:
    """Find the request_id in today's CSV log and revert label from 1 to 0."""
    today_str = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    csv_path = os.path.join(DAILY_LOGS_DIR, f"log_{today_str}.csv")

    if not os.path.exists(csv_path):
        print(f"[CSV] Log file not found: {csv_path}")
        return False

    try:
        df = pd.read_csv(csv_path)
        
        if 'request_id' not in df.columns or 'label' not in df.columns:
            print("[CSV] Required columns not found in the dataset.")
            return False

        # Find the row matching the request_id
        mask = df['request_id'] == req_id
        if mask.any():
            df.loc[mask, 'label'] = 0
            df.to_csv(csv_path, index=False)
            print(f"[CSV] Successfully reverted label to 0 for request_id: {req_id}")
            return True
        else:
            print(f"[CSV] request_id not found in today's log: {req_id}")
            return False
            
    except Exception as e:
        print(f"[CSV] Error processing file: {e}")
        return False

async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle Telegram inline keyboard button clicks."""
    query = update.callback_query
    
    # Acknowledge the callback query to remove the loading state on the Telegram client
    await query.answer()

    data = query.data
    
    if data.startswith("unblock|"):
        parts = data.split("|")
        if len(parts) == 3:
            action, ip, req_id = parts
            
            print(f"\n[TELEGRAM] Received unblock request for IP: {ip} | ReqID: {req_id}")
            
            # 1. Send Unblock Command to Firewall
            fw_success = unblock_ip_firewall(ip)
            
            # 2. Rollback Label in CSV
            csv_success = rollback_csv_label(req_id)
            
            # 3. Formulate response message
            response_msg = f"[RESOLVED] Admin intervened.\n\nSource IP: {ip}\nRequest ID: {req_id}\n\n"
            
            if fw_success:
                response_msg += "- Firewall: IP Unblocked successfully.\n"
            else:
                response_msg += "- Firewall: Unblock failed (Check logs).\n"
                
            if csv_success:
                response_msg += "- ML Dataset: Label reverted to 0 (False Positive)."
            else:
                response_msg += "- ML Dataset: Revert failed (ID not found)."
            
            # Update the original message to remove the buttons and show resolution
            await query.edit_message_text(text=response_msg)

if __name__ == '__main__':
    print("=====================================================")
    print("TELEGRAM LISTENER SERVICE IS RUNNING...")
    print("Waiting for Admin commands via Telegram...")
    print("=====================================================")
    
    if not TELEGRAM_BOT_TOKEN:
        print("[ERROR] TELEGRAM_BOT_TOKEN is missing in .env file.")
        sys.exit(1)
        
    # Build the Telegram application
    application = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    # Add handler for button clicks
    application.add_handler(CallbackQueryHandler(button_callback))

    # Start polling
    application.run_polling()