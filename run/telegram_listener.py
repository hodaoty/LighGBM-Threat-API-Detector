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
FIREWALL_API_UNBLOCK = "http://127.0.0.1:9000/api/unblock"
FIREWALL_API_BLOCK = "http://127.0.0.1:9000/api/autoban"

FW_ADMIN_USER = "admin"
FW_ADMIN_PASS = "admin123"
FW_AUTH_TOKEN = hashlib.sha256(f"{FW_ADMIN_USER}:{FW_ADMIN_PASS}_secure_firewall".encode()).hexdigest()

DAILY_LOGS_DIR = os.path.join(PROJECT_ROOT, "data", "daily_logs")

def call_firewall_api(api_url: str, ip: str, req_id: str = "") -> bool:
    """Call Firewall Admin API to block or unblock an IP."""
    try:
        payload = {"ip": ip}
        if req_id:
            payload["request_id"] = req_id
            
        response = requests.post(
            api_url,
            json=payload,
            cookies={"session": FW_AUTH_TOKEN},
            timeout=5
        )
        if response.status_code == 200:
            action = "unblocked" if "unblock" in api_url else "blocked"
            print(f"[FIREWALL] Successfully {action} IP: {ip}")
            return True
        else:
            print(f"[FIREWALL] Failed API Call. Status: {response.status_code}")
            return False
    except Exception as e:
        print(f"[FIREWALL] API connection error: {e}")
        return False

def change_csv_label(req_id: str, new_label: int) -> bool:
    """Find the request_id in today's CSV log and update the label (0 or 1)."""
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
            df.loc[mask, 'label'] = new_label
            df.to_csv(csv_path, index=False)
            print(f"[CSV] Successfully updated label to {new_label} for request_id: {req_id}")
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
    
    # Acknowledge the callback query to remove the loading state
    await query.answer()

    data = query.data
    parts = data.split("|")
    
    if len(parts) == 3:
        action, ip, req_id = parts
        
        print(f"\n[TELEGRAM] Received '{action.upper()}' request for IP: {ip} | ReqID: {req_id}")
        
        # -----------------------------------------------------
        # CASE 1: FALSE POSITIVE ALARM -> UNBLOCK
        # -----------------------------------------------------
        if action == "unblock":
            fw_success = call_firewall_api(FIREWALL_API_UNBLOCK, ip)
            csv_success = change_csv_label(req_id, new_label=0)
            
            response_msg = f"*[RESOLVED - FALSE POSITIVE]*\n\n*Source IP:* `{ip}`\n*Request ID:* `{req_id}`\n\n"
            response_msg += "- Firewall: IP Unblocked successfully.\n" if fw_success else "- Firewall: Unblock failed (Check logs).\n"
            response_msg += "- ML Dataset: Label reverted to 0." if csv_success else "- ML Dataset: Revert failed (ID not found)."

        # -----------------------------------------------------
        # CASE 2: CONFIRMED ATTACK (TRUE POSITIVE) -> BLOCK
        # -----------------------------------------------------
        elif action == "block":
            fw_success = call_firewall_api(FIREWALL_API_BLOCK, ip, req_id)
            csv_success = change_csv_label(req_id, new_label=1)
            
            response_msg = f"*[RESOLVED - TRUE POSITIVE]*\n\n*Source IP:* `{ip}`\n*Request ID:* `{req_id}`\n\n"
            response_msg += "- Firewall: IP Blocked successfully.\n" if fw_success else "- Firewall: Block failed (Check logs).\n"
            response_msg += "- ML Dataset: Label updated to 1 (Active Learning)." if csv_success else "- ML Dataset: Label update failed."

        else:
            response_msg = "Unknown action received."

        # Update the original message to remove the buttons and show resolution
        await query.edit_message_text(text=response_msg, parse_mode="Markdown")

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