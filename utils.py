from sklearn.metrics import roc_curve, auc, accuracy_score
import matplotlib.pyplot as plt
import wandb
import os 
import numpy as np 
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# def plot_roc_curve(y_true, y_probs, fold_idx=None, wandb_logger=None):
#     fpr, tpr, thresholds = roc_curve(y_true, y_probs)
#     roc_auc = auc(fpr, tpr)

#     plt.figure()
#     plt.plot(fpr, tpr, lw=2, label=f'Fold {fold_idx} (AUC = {roc_auc:.2f})')
#     plt.plot([0, 1], [0, 1], 'k--', lw=1)
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title(f'ROC Curve - Fold {fold_idx}')
#     plt.legend(loc='lower right')
#     plt.grid(True)

#     img_path = f'plots/roc_curve_fold_{fold_idx}.png'
#     plt.savefig(img_path)
#     plt.close()

#     if wandb_logger:
#         wandb_logger.experiment.log({f'ROC Curve Fold {fold_idx}': wandb.Image(img_path)})

#     return fpr, tpr, roc_auc


def calculate_auc(y_true, y_probs):
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)

    return roc_auc


def plot_roc_curve(y_true, y_probs, fold_idx=None, wandb_logger=None):
    # Compute ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)

    # Compute accuracy (default threshold 0.5)
    y_pred = (y_probs >= 0.5).astype(int)
    accuracy = accuracy_score(y_true, y_pred)

    # Plot ROC
    plt.figure()
    plt.plot(fpr, tpr, lw=2, label=f'Fold {fold_idx} (AUC = {roc_auc:.2f}, Acc = {accuracy:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - Fold {fold_idx}')
    plt.legend(loc='lower right')
    plt.grid(True)

    # Save and log image
    os.makedirs("plots", exist_ok=True)
    img_path = f'plots/roc_curve_fold_{fold_idx}.png'
    plt.savefig(img_path)
    plt.close()

    if wandb_logger:
        wandb_logger.experiment.log({
            f'ROC Curve Fold {fold_idx}': wandb.Image(img_path),
            f'fold_{fold_idx}_auc': roc_auc,
            f'fold_{fold_idx}_accuracy': accuracy
        })

    return fpr, tpr, roc_auc


def compute_weighted_accuracy(preds, targets):
    preds = (preds > 0.5).int()
    targets = targets.int()

    pos_mask = targets == 1
    neg_mask = targets == 0

    pos_correct = (preds[pos_mask] == 1).sum()
    neg_correct = (preds[neg_mask] == 0).sum()
    pos_total = pos_mask.sum()
    neg_total = neg_mask.sum()

    pos_acc = pos_correct / (pos_total + 1e-8)
    neg_acc = neg_correct / (neg_total + 1e-8)
    return ((pos_acc + neg_acc) / 2).item()


def add_gaussian_noise_so2(signal, noise_ratio=0.05, signal_min = 0.7,  signal_max = 0.8):
    """
    Add Gaussian noise relative to the signal’s dynamic range.
    noise_ratio: fraction of signal std to use as noise std
    """
    noise_std = noise_ratio * (signal_max - signal_min)
    noise = np.random.normal(0, noise_std)
    return signal + noise

def add_gaussian_noise_thb(signal, noise_ratio=0.05, signal_min =2.5, signal_max = 20):
    """
    Add Gaussian noise relative to the signal’s dynamic range.
    noise_ratio: fraction of signal std to use as noise std
    """
    noise_std = noise_ratio * (signal_max - signal_min)
    noise = np.random.normal(0, noise_std)
    return signal + noise