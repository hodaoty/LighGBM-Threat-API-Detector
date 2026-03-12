import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
import joblib
import os
import sys

# ==========================================
# PATH CONFIGURATION
# ==========================================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(PROJECT_ROOT)

from src.features.common_features import build_features

DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "master_dataset_cleaned.csv")
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "lightgbm_threatAPI_detector.pkl")
FIGURE_DIR = os.path.join(PROJECT_ROOT, "reports", "figures") # Directory to save images

def plot_evaluation_metrics():
    print("1. Loading data and recreating Test set (20%)...")
    df = pd.read_csv(DATA_PATH)
    X, y = build_features(df)
    
    # MUST split exactly as in Train (random_state=42) to get the same test set
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("2. Loading LightGBM Model...")
    model = joblib.load(MODEL_PATH)
    
    print("3. Making predictions...")
    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)[:, 1] # Get probability for label 1 (Attack)

    # Create image directory if it doesn't exist
    os.makedirs(FIGURE_DIR, exist_ok=True)

    print("4. Plotting charts...")
    
    # ---------------------------------------------------------
    # CHART 1: CONFUSION MATRIX
    # ---------------------------------------------------------
    cm = confusion_matrix(y_test, y_test_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Predicted Normal (0)', 'Predicted Attack (1)'], 
                yticklabels=['Actual Normal', 'Actual Attack'],
                annot_kws={"size": 14}) # Enlarge numbers inside cells
    
    plt.title('Confusion Matrix (LightGBM)', fontsize=16, pad=15)
    plt.xlabel('Prediction', fontsize=12)
    plt.ylabel('Reality', fontsize=12)
    plt.tight_layout()
    
    # Save and show image
    cm_path = os.path.join(FIGURE_DIR, 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=300)
    print(f"  -> Confusion Matrix saved at: {cm_path}")
    plt.show()

    # ---------------------------------------------------------
    # CHART 2: ROC CURVE & AUC
    # ---------------------------------------------------------
    fpr, tpr, thresholds = roc_curve(y_test, y_test_proba)
    roc_auc = roc_auc_score(y_test, y_test_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guessing')
    
    # Decorate ROC chart
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve (LightGBM)', fontsize=16, pad=15)
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    # Save and show image
    roc_path = os.path.join(FIGURE_DIR, 'roc_curve.png')
    plt.savefig(roc_path, dpi=300)
    print(f"  -> ROC Curve saved at: {roc_path}")
    plt.show()
    
    print("\nDONE! Evaluation charts generated successfully.")

if __name__ == "__main__":
    plot_evaluation_metrics()