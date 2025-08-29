from sklearn.metrics import roc_curve, accuracy_score, confusion_matrix, auc
import numpy as np
import pickle

# Load saved ROC data (with labels & probs included)
with open(r'plots\roc_data.pkl', 'rb') as f:
#with open(r'plots\roc_data_72e7c80_with_labels.pkl', 'rb') as f:
    roc_data = pickle.load(f)

all_labels = roc_data['all_labels']
all_probs = roc_data['all_probs']

accs, sens, specs, aucs = [], [], [], []

for i, (y_true, y_prob) in enumerate(zip(all_labels, all_probs)):
    if i == 2: 
        continue  # Only first 2 folds for testing
    # Get ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)

    # Compute AUC
    aucs.append(auc(fpr, tpr))

    # Find best threshold using Youden’s J statistic
    j_scores = tpr - fpr
    j_best = np.argmax(j_scores)
    best_thresh = thresholds[j_best]

    # Apply threshold
    y_pred = (y_prob >= best_thresh).astype(int)

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Metrics
    accs.append(accuracy_score(y_true, y_pred))
    sens.append(tp / (tp + fn))  # Sensitivity (Recall)
    specs.append(tn / (tn + fp)) # Specificity

# Mean scores across folds
mean_acc = np.mean(accs)
mean_sens = np.mean(sens)
mean_spec = np.mean(specs)
mean_auc = np.mean(aucs)

print(f"Mean Accuracy:    {mean_acc:.3f}")
print(f"Mean Sensitivity: {mean_sens:.3f}")
print(f"Mean Specificity: {mean_spec:.3f}")
print(f"Mean AUC:         {mean_auc:.3f}")
