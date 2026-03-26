import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    roc_auc_score, 
    f1_score, 
    accuracy_score, 
    precision_score, 
    roc_curve, 
    auc
)

def find_optimal_threshold(all_labels, all_probs):
    """
    Finds the threshold that maximizes the F1-score.
    """
    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
    f1_scores = []
    for t in thresholds:
        preds = (all_probs >= t).astype(int)
        f1_scores.append(f1_score(all_labels, preds))
    
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    print(f"Optimal Threshold Found: {best_threshold:.4f} (Max F1: {f1_scores[best_idx]:.4f})")
    return best_threshold

def calculate_and_print_metrics(all_labels, all_preds, all_probs, threshold=0.5):
    """
    Calculates and prints common classification metrics.
    """
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    roc_auc = roc_auc_score(all_labels, all_probs)
    
    print(f"\n--- Evaluation Metrics (Threshold: {threshold:.4f}) ---")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=['Non-Lens', 'Lens']))
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "f1": f1,
        "roc_auc": roc_auc
    }

def plot_evaluation_results(all_labels, all_preds, all_probs, threshold=0.5, save_path='evaluation_results.png'):
    """
    Generates and saves a side-by-side Confusion Matrix and ROC Curve plot.
    """
    # 1. Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Lens', 'Lens'], yticklabels=['Non-Lens', 'Lens'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix (Threshold: {threshold:.4f})')
    
    # 2. ROC Curve
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc_val = auc(fpr, tpr)
    
    plt.subplot(1, 2, 2)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc_val:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Evaluation plots saved to {save_path}")
    plt.show()
