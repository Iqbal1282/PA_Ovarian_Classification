import torch
import wandb
import numpy as np
import random
import re
import subprocess
import os
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import roc_curve, auc
from torch.utils.data import DataLoader
from fusion_models import ThreeModalTransformerWithRadiomics
from dataset import MultimodalDatasetWithRadiomics
from utils import plot_roc_curve, compute_weighted_accuracy, calculate_auc
from tqdm import tqdm 
from torchmetrics.classification import BinaryAccuracy, BinaryAUROC
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, StepLR, ReduceLROnPlateau, ExponentialLR
# Set deterministic behavior
SEED = 42
np.random.seed(SEED); torch.manual_seed(SEED); random.seed(SEED)

# Settings
max_epochs = 50
batch_size = 16
k_fold = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# Git Commit Info
try:
    commit_string = subprocess.check_output(["git", "log", "-1", "--pretty=%s"]).decode("utf-8").strip()
    commit_string = re.sub(r'\W+', '_', commit_string)
    commit_log = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode("utf-8").strip()
except Exception as e:
    commit_string, commit_log = "no_commit", "0000"
    print(f"Git commit fetch failed: {e}")

# WandB Settings
project_title = "PA-SO2-radiomics-Ovarian-Cancer-Classification-Washu2"
experiment_group = f"Exp1:{commit_string}_{commit_log}"
train_config = {
    "k_fold": k_fold,
    "batch_size": batch_size,
    "radiomics": False,
    "encoder_checkpoint": "normtverskyloss_binary_segmentation",
    "input_dim": '256x256:64',
    "model_type": "BinaryClassificationTorch",
    "info": "Training without PyTorch Lightning",
}

# Storage for fold-wise metrics
all_fprs, all_tprs, all_aucs = [], [], []

# --- Start K-Fold Training ---
for fold in range(k_fold):
    run = wandb.init(project=project_title, name=f"Fold_{fold}", group=experiment_group, config=train_config)
    
    # # Datasets & Loaders
    # train_dataset = Classificaiton_Dataset(phase='train', k_fold=k_fold, fold=fold, radiomics_dir=False)
    # val_dataset = Classificaiton_Dataset(phase='val', k_fold=k_fold, fold=fold, radiomics_dir=False)
    # test_dataset = Classificaiton_Dataset(phase='test', radiomics_dir=False)

    train_dataset = MultimodalDatasetWithRadiomics(
        so2_csv_path ='PAT features/roi_so2_image_metadata.csv',
        thb_csv_path= 'PAT features/roi_thb_image_metadata.csv',
        mat_root_dir='PAT features/ROI_MAT',
        phase='train',
        k_fold=5,
        fold=fold 
    )


    val_dataset = MultimodalDatasetWithRadiomics(
        so2_csv_path ='PAT features/roi_so2_image_metadata.csv',
        thb_csv_path= 'PAT features/roi_thb_image_metadata.csv',
        mat_root_dir='PAT features/ROI_MAT',
        phase='va',
        k_fold=5,
        fold=fold
    )

    test_dataset = MultimodalDatasetWithRadiomics(
        so2_csv_path ='PAT features/roi_so2_image_metadata.csv',
        thb_csv_path= 'PAT features/roi_thb_image_metadata.csv',
        mat_root_dir='PAT features/ROI_MAT',
        phase='test',
        k_fold=5,
        fold=0
    )



    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Model
    #model = MultiClassificationTorch_Imagenet(num_classes= 1, backbone_name= 'resnet18').to(device)
    #model = MultiModalCancerClassifierWithAttention(num_modalities=2, out_dim=1, fusion_dim=64, backbone_name='resnet18', dropout_prob=0.3).to(device)
    #model = MultiModalTransformerClassifier(num_classes=1, img_size = 448, patch_size= 32, num_layers = 4).to(device) #)
    model = ThreeModalTransformerWithRadiomics(img_size=448, patch_size=32, embed_dim=256, num_heads=4, num_layers=6, num_classes=1, rad_dim=2, dropout=0.1).to(device)
    #optimizer = torch.optim.Adam(model.parameters(), lr=5e-5, weight_decay=1e-5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-6, weight_decay=1e-2)
    #scheduler = ExponentialLR(optimizer=optimizer, gamma=0.9) #
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)

    best_val_auc = -1
    best_combined_score = -1 
    best_model_state = None 
    
    # Metrics
    accuracy_metric = BinaryAccuracy().to(device)
    auc_metric = BinaryAUROC().to(device)

    # --- Training Loop ---
    for epoch in tqdm(range(max_epochs), leave= False):
        model.train()
        epoch_loss = 0.0

        # batch in train_loader:
        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            x1, x2, x3, y = batch
            loss = model.compute_loss(x1.to(device), x2.to(device), x3.to(device), y.to(device)) #, x2.to(device))
            loss.backward()
            optimizer.step()
            #optimizer.step()
            scheduler.step(epoch + batch_idx / len(train_loader))
            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_loader)
        #scheduler.step()
        wandb.log({f"train/loss_fold_{fold}": avg_train_loss, "epoch": epoch})
        current_lr = scheduler.get_last_lr()[0]
        wandb.log({f"train/lr_fold_{fold}": current_lr, "epoch": epoch})

        # --- Train Evaluation (AUC) ---
        model.eval()
        train_y_true, train_y_probs = [], []
        with torch.no_grad():
            for batch in train_loader:
                x1, x2, x3, y = batch
                x1, x2, x3, y = x1.to(device), x2.to(device), x3.to(device), y.to(device)
                scores = model(x1, x2, x3)
                probs = torch.sigmoid(scores)
                train_y_probs.append(probs)
                train_y_true.append(y)

        train_y_true = torch.cat(train_y_true)
        train_y_probs = torch.cat(train_y_probs)

        train_auc = calculate_auc(
            train_y_true.cpu().numpy(), train_y_probs.cpu().numpy())

        wandb.log({f"train/roc_auc_fold_{fold}": train_auc, "epoch": epoch})


        # --- Validation Evaluation ---
        model.eval()
        y_true, y_probs = [], []
        with torch.no_grad():
            for batch in val_loader:
                x1, x2, x3, y = batch
                x1, x2, x3, y = x1.to(device), x2.to(device), x3.to(device), y.to(device)
                scores = model(x1, x2, x3)

                probs = torch.sigmoid(scores)
                y_probs.append(probs)
                y_true.append(y)

                accuracy_metric.update(probs, y.int())
                auc_metric.update(probs, y.int())

        y_true = torch.cat(y_true)
        y_probs = torch.cat(y_probs)

        # Compute metrics
        val_accuracy = accuracy_metric.compute().item()
        val_auc = auc_metric.compute().item()
        val_wacc = compute_weighted_accuracy(y_probs, y_true)

        # Reset metrics
        accuracy_metric.reset()
        auc_metric.reset()

        # ROC
        fpr, tpr, roc_auc = plot_roc_curve(y_true.cpu().numpy(), y_probs.cpu().numpy(), fold_idx=fold + 1)
        
        combined_score = 0.2*val_wacc + 0.3* val_accuracy + 0.5* roc_auc 

        # Log all metrics
        wandb.log({
            f"val/roc_auc_fold_{fold}": val_auc,
            f"val/accuracy_fold_{fold}": val_accuracy,
            f"val/weighted_accuracy_fold_{fold}": val_wacc,
            #f"val/roc_curve_fold_{fold}": wandb.Image(f"plots/roc_curve_fold_{fold+1}.png"),
            "epoch": epoch
        })

        if roc_auc > best_val_auc:
            best_val_auc = roc_auc
            best_model_state = model.state_dict()

        # if (train_auc*0.5 + roc_auc *0.5)  > best_val_auc:
        #     best_val_auc = train_auc *0.5 + roc_auc *0.5
        #     best_model_state = model.state_dict()

        # # if combined_score > best_combined_score:
        # #     best_combined_score = roc_auc
        # #     best_model_state = model.state_dict()

    # --- Load Best Model and Test ---
    model.load_state_dict(best_model_state)
    y_true, y_probs = model.predict_on_loader(test_loader)
    fpr, tpr, roc_auc = plot_roc_curve(y_true, y_probs, fold_idx=fold + 1)
    wandb.log({
        #f"test/roc_auc_fold_{fold}": roc_auc,
        f"test/roc_curve_fold_{fold}": wandb.Image(f"plots/roc_curve_fold_{fold+1}.png"),
    })
    
    all_fprs.append(fpr)
    all_tprs.append(tpr)
    all_aucs.append(roc_auc)
    run.finish()

# --- Plot Multi-Fold ROC Curve ---
mean_fpr = np.linspace(0, 1, 100)
interp_tprs = []

plt.figure()
for i, (fpr, tpr, auc_score) in enumerate(zip(all_fprs, all_tprs, all_aucs)):
    interp_tpr = np.interp(mean_fpr, fpr, tpr)
    interp_tpr[0] = 0.0
    interp_tprs.append(interp_tpr)
    plt.plot(fpr, tpr, lw=1.5, alpha=0.7, label=f'Fold {i+1} (AUC = {auc_score:.2f})')

mean_tpr = np.mean(interp_tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)

plt.plot(mean_fpr, mean_tpr, color='b', lw=2, linestyle='--', label=f'Mean ROC (AUC = {mean_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--', lw=1)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves Across All Folds')
plt.legend(loc='lower right')
plt.grid(True)
os.makedirs("plots", exist_ok=True)
final_img_path = 'plots/roc_all_folds.png'
plt.savefig(final_img_path)
plt.close()

import pickle

# Save ROC-related data
roc_data = {
    'all_fprs': all_fprs,
    'all_tprs': all_tprs,
    'all_aucs': all_aucs,
    'mean_fpr': mean_fpr,
    'mean_tpr': mean_tpr,
    'mean_auc': mean_auc
}

# Save as pickle
with open('plots/roc_data.pkl', 'wb') as f:
    pickle.dump(roc_data, f)

# Final ROC to WandB
final_run = wandb.init(
    project=project_title,
    name=f"All_Folds_{commit_log}",
    group=experiment_group,
)
final_run.log({"ROC Curve - All Folds": wandb.Image(final_img_path)})
final_run.finish()
