import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from sklearn.metrics import accuracy_score, roc_auc_score
import numpy as np
import os
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F 
import torch 
import torchvision.utils as vutils
import wandb
import torch
from torchmetrics.classification import BinaryAccuracy
import torchmetrics
from torchmetrics.classification import MulticlassAccuracy, MulticlassAUROC
from torchvision.models import resnet18 , ResNet18_Weights
import torchvision 


import torch.nn as nn
import segmentation_models_pytorch as smp
from einops import rearrange


class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_pos=0, gamma_neg=4, clip=0.05, eps=1e-8):
        super(AsymmetricLoss, self).__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip
        self.eps = eps

    def forward(self, inputs, targets):
        inputs_sigmoid = torch.sigmoid(inputs)
        inputs_sigmoid = torch.clamp(inputs_sigmoid, self.eps, 1 - self.eps)

        if self.clip is not None and self.clip > 0:
            inputs_sigmoid = (inputs_sigmoid - self.clip).clamp(min=0, max=1)

        targets = targets.float()
        loss_pos = targets * torch.log(inputs_sigmoid) * (1 - inputs_sigmoid) ** self.gamma_pos
        loss_neg = (1 - targets) * torch.log(1 - inputs_sigmoid) * inputs_sigmoid ** self.gamma_neg
        loss = -loss_pos - loss_neg
        return loss.mean()
    
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=3.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        probs = torch.sigmoid(inputs)
        p_t = targets * probs + (1 - targets) * (1 - probs)
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        loss = alpha_t * (1 - p_t) ** self.gamma * bce_loss
        return loss.mean() if self.reduction == 'mean' else loss.sum()


class SDFModel(nn.Module):
    def __init__(self):
        super(SDFModel, self).__init__()
        self.backbone = smp.DeepLabV3Plus(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=1,
            classes=1
        )
        self.activation = nn.Tanh()

    def forward(self, x):
        x = self.backbone(x)        # Output shape: (B, 1, H, W)
        x = self.activation(x)      # Output in [-1, 1]
        return x
    
# class AttentionFusion(nn.Module):
#     def __init__(self, embed_dim, num_modalities=4):
#         super().__init__()
#         self.query = nn.Linear(embed_dim, embed_dim)
#         self.key = nn.Linear(embed_dim, embed_dim)
#         self.value = nn.Linear(embed_dim, embed_dim)
#         self.scale = embed_dim ** 0.5

#     def forward(self, x):  # x: (B, M, D) where M=modalities, D=embed_dim
#         Q = self.query(x)  # (B, M, D)
#         K = self.key(x)    # (B, M, D)
#         V = self.value(x)  # (B, M, D)

#         attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (B, M, M)
#         attn_weights = torch.softmax(attn_scores, dim=-1)                # (B, M, M)
#         fused = torch.matmul(attn_weights, V)                            # (B, M, D)

#         # Optionally pool across modalities (e.g., mean)
#         return fused.mean(dim=1)  # (B, D)

class AttentionFusion(nn.Module):
    def __init__(self, embed_dim, num_modalities):
        super().__init__()
        self.embed_dim = embed_dim
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.scale = embed_dim ** -0.5
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):  # (B, M, D)
        Q = self.query(x)  # (B, M, D)
        K = self.key(x)
        V = self.value(x)

        attn_scores = torch.bmm(Q, K.transpose(1, 2)) * self.scale  # (B, M, M)
        attn_weights = self.softmax(attn_scores)  # (B, M, M)
        fused = torch.bmm(attn_weights, V)  # (B, M, D)
        fused = fused.mean(dim=1)  # Aggregate across modalities
        return fused


class MultiModalCancerClassifierWithAttention(nn.Module):
    def __init__(self, out_dim=1, fusion_dim=None, backbone_name='resnet18', dropout_prob=0.0, use_layernorm=True, num_modalities=3):
        super().__init__()
        self.num_modalities = num_modalities
        self.dropout_prob = dropout_prob
        self.use_layernorm = use_layernorm

        # Create modality-specific backbones
        self.backbones = nn.ModuleList([
            timm.create_model(backbone_name, pretrained=True, num_classes=0)
            for _ in range(self.num_modalities)
        ])
        self.backbone_out_dim = self.backbones[0].num_features

        # Set a sensible default fusion dim
        if fusion_dim is None:
            fusion_dim = min(512, self.backbone_out_dim)  # Keep dimension smaller for fusion stability

        # Projection layers
        self.projs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.backbone_out_dim, fusion_dim),
                nn.LayerNorm(fusion_dim) if use_layernorm else nn.Identity(),
                nn.ReLU()
            )
            for _ in range(self.num_modalities)
        ])

        # Fusion attention
        self.attn_fusion = AttentionFusion(embed_dim=fusion_dim, num_modalities=self.num_modalities)

        # Deeper classifier for richer features
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, out_dim)
        )

        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([3.0] * out_dim))

    def forward(self, imgs):  # imgs: list of 3 tensors
        B = imgs[0].shape[0]
        device = imgs[0].device

        fused_feats = []
        for i in range(self.num_modalities):
            x = imgs[i]

            # Modality dropout
            if self.training and random.random() < self.dropout_prob :
                fused_feats.append(torch.zeros(B, self.projs[i][0].out_features, device=device))
                continue

            # Convert grayscale to RGB
            if x.shape[1] == 1:
                x = x.repeat(1, 3, 1, 1)

            feat = self.backbones[i](x)  # (B, backbone_out_dim)
            proj_feat = self.projs[i](feat)  # (B, fusion_dim)
            fused_feats.append(proj_feat)

        fused_stack = torch.stack(fused_feats, dim=1)  # (B, M, fusion_dim)
        fused_output = self.attn_fusion(fused_stack)   # (B, fusion_dim)
        out = self.classifier(fused_output)            # (B, out_dim)
        return out
    
    def compute_loss(self, x, y, x2_rad=None):
        y = y.float()  # Ensure targets are float for BCE loss
        if x2_rad is not None:
            score, tails = self.forward(x, x2_rad)
            loss = self.loss_fn(score, y) + sum(self.loss_fn(t, y) for t in tails)
        else:
            score = self.forward(x)
            loss = self.loss_fn(score, y) #+ 0.5 * self.loss_fn2(score, y)

        return loss

    def predict_on_loader(self, dataloader, threshold=0.5):
        self.eval()
        all_probs, all_targets = [], []

        device = next(self.parameters()).device

        with torch.no_grad():
            for batch in dataloader:
                if len(batch) == 2:
                    x, y = batch
                    x, y = x.to(device), y.to(device)
                    scores = self.forward(x)
                else:
                    x, x2, y = batch
                    x, x2, y = x.to(device), x2.to(device), y.to(device)
                    scores = self.forward([x, x2])

                probs = torch.sigmoid(scores)
                all_probs.append(probs.cpu())
                all_targets.append(y.cpu())

        return torch.cat(all_targets).numpy(), torch.cat(all_probs).numpy()
    


class MultiClassificationTorch(nn.Module): 
    def __init__(self, input_dim=64, num_classes=8, radiomics=False, radiomics_dim=463, 
                 encoder_weight_path=None, sdf_model_path=None):
        super().__init__()

        self.input_size = input_dim
        self.num_classes = num_classes
        self.radiomics = radiomics

        self.sdf_model = SDFModel()
        self.sdf_model.load_state_dict(torch.load(sdf_model_path))
        for p in self.sdf_model.parameters(): 
            p.requires_grad = False

        self.fusion_model = MultiModalCancerClassifierWithAttention(out_dim=num_classes, backbone_name= "resnet18", dropout_prob= 0)

        # Losses for multi-label (use BCE with logits)
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.loss_fn2 = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([3.0] * num_classes))

    def normalize_sdf(self, sdf_image):
        sdf_image = (sdf_image - sdf_image.min()) / (sdf_image.max() - sdf_image.min() + 1e-8)
        return sdf_image * 2 - 1

    def forward(self, x, x2_radiomics=None):
        x_sdf = self.sdf_model(x.mean(dim = 1, keepdim = True))
        x_sdf = self.normalize_sdf(x_sdf)

        lower_thresh = torch.empty(1).uniform_(-0.7, -0.6).item()
        upper_thresh = torch.empty(1).uniform_(0.35, 0.45).item()
        center_thresh = torch.empty(1).uniform_(0.1, 0.25).item()

        boundary_mask = (x_sdf < upper_thresh) & (x_sdf > lower_thresh)
        center_mask = (x_sdf < center_thresh)

        x3 = x * boundary_mask
        x4 = x * center_mask

        output = self.fusion_model([x, x3, x4])  # Output: (B, num_classes)

        return output

    def compute_loss(self, x, y, x2_rad=None):
        y = y.float()  # Ensure targets are float for BCE loss
        if x2_rad is not None:
            score, tails = self.forward(x, x2_rad)
            loss = self.loss_fn(score, y) + sum(self.loss_fn(t, y) for t in tails)
        else:
            score = self.forward(x)
            loss = 0.5 * self.loss_fn(score, y) + 0.5 * self.loss_fn2(score, y)

        return loss

    def predict_on_loader(self, dataloader, threshold=0.5):
        self.eval()
        all_probs, all_targets = [], []

        device = next(self.parameters()).device

        with torch.no_grad():
            for batch in dataloader:
                if len(batch) == 2:
                    x, y = batch
                    x, y = x.to(device), y.to(device)
                    scores = self.forward(x)
                else:
                    x, x2, y = batch
                    x, x2, y = x.to(device), x2.to(device), y.to(device)
                    scores = self.forward(x, x2)

                probs = torch.sigmoid(scores)
                all_probs.append(probs.cpu())
                all_targets.append(y.cpu())

        return torch.cat(all_targets).numpy(), torch.cat(all_probs).numpy()
    
class MultiClassificationTorch_Imagenet(nn.Module): 
    def __init__(self, num_classes=8, backbone_name = 'resnet50'):
        super().__init__()
        self.num_classes = num_classes
        self.backbone = timm.create_model(backbone_name, pretrained=True, num_classes=num_classes)
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([3.0] * num_classes))

    def forward(self, x, x2_radiomics=None):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        x = self.backbone(x)
        return x

    def compute_loss(self, x, y, x2_rad=None):
        y = y.float()  # Ensure targets are float for BCE loss
        if x2_rad is not None:
            score, tails = self.forward(x, x2_rad)
            loss = self.loss_fn(score, y) + sum(self.loss_fn(t, y) for t in tails)
        else:
            score = self.forward(x)
            loss = self.loss_fn(score, y) #+ 0.5 * self.loss_fn2(score, y)

        return loss

    def predict_on_loader(self, dataloader, threshold=0.5):
        self.eval()
        all_probs, all_targets = [], []

        device = next(self.parameters()).device

        with torch.no_grad():
            for batch in dataloader:
                if len(batch) == 2:
                    x, y = batch
                    x, y = x.to(device), y.to(device)
                    scores = self.forward(x)
                else:
                    x, x2, y = batch
                    x, x2, y = x.to(device), x2.to(device), y.to(device)
                    scores = self.forward(x, x2)

                probs = torch.sigmoid(scores)
                all_probs.append(probs.cpu())
                all_targets.append(y.cpu())

        return torch.cat(all_targets).numpy(), torch.cat(all_probs).numpy()
    
class BinaryClassificationTorch(nn.Module):
    def __init__(self, input_dim=64, output_size = 5, num_classes=1, radiomics=False, radiomics_dim=463,
                 encoder_weight_path=None, sdf_model_path=None):
        super().__init__()

        self.input_size = input_dim
        self.hidden_sizes = [512, 128, 64, 32]
        self.hidden_sizes2 = [64, 32]
        self.output_size = output_size 
        self.radiomics = radiomics

        self.sdf_model = SDFModel()
        self.sdf_model.load_state_dict(torch.load(sdf_model_path))
        for p in self.sdf_model.parameters(): p.requires_grad = False

        self.fusion_model = MultiModalCancerClassifierWithAttention()

        self.loss_fn = FocalLoss()
        self.loss_fn2 = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([3.0]))

    def normalize_sdf(self, sdf_image):
        sdf_image = (sdf_image - sdf_image.min()) / (sdf_image.max() - sdf_image.min() + 1e-8)
        return sdf_image * 2 - 1

    def forward(self, x, x2_radiomics=None):
        x_sdf = self.sdf_model(x)
        x_sdf = self.normalize_sdf(x_sdf)

        lower_thresh = torch.empty(1).uniform_(-0.45, -0.15).item()
        upper_thresh = torch.empty(1).uniform_(0.35, 0.65).item()
        center_thresh = torch.empty(1).uniform_(0.1, 0.25).item()

        boundary_mask = (x_sdf < upper_thresh) & (x_sdf > lower_thresh)
        center_mask = (x_sdf < center_thresh)

        x3 = x * boundary_mask
        x4 = x * center_mask

        output = self.fusion_model([x, x3, x4])

        return output


    def compute_loss(self, x, y, x2_rad=None):
        if x2_rad is not None:
            score, tails = self.forward(x, x2_rad)
            loss = self.loss_fn(score, y.float()) + sum(self.loss_fn(t, y.float()) for t in tails)
        else:
            score = self.forward(x)
            loss = (self.loss_fn(score, y.float()) * 0.5 +
                    self.loss_fn2(score, y.float()) * 0.5)
                   
        return loss

    def predict_on_loader(self, dataloader):
        self.eval()
        all_probs, all_targets = [], []

        device = next(self.parameters()).device  # Automatically detect model's device

        with torch.no_grad():
            for batch in dataloader:
                if len(batch) == 2:
                    x, y = batch
                    x, y = x.to(device), y.to(device)
                    scores = self.forward(x)
                    
                else:
                    x, x2, y = batch
                    x, x2, y = x.to(device), x2.to(device), y.to(device)
                    scores = self.forward(x, x2)

                probs = torch.sigmoid(scores)
                all_probs.append(probs.cpu())
                all_targets.append(y.cpu())

        return torch.cat(all_targets).numpy(), torch.cat(all_probs).numpy()
    

class PatchEmbed(nn.Module):
    def __init__(self, img_size=256, patch_size=16, in_chans=1, embed_dim=768):
        super().__init__()
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: [B, 1, H, W] → [B, embed_dim, H//P, W//P] → [B, num_patches, embed_dim]
        x = self.proj(x)  # [B, E, H/P, W/P]
        x = x.flatten(2).transpose(1, 2)  # [B, N, E]
        return x
    

class MultiModalTransformerClassifier(nn.Module):
    def __init__(self, img_size=448, patch_size=32, embed_dim=256, num_heads=4, num_layers=6, num_classes=8):
        super().__init__()
        
        self.patch_dim = (img_size // patch_size) ** 2
        self.patch_embed_dim = embed_dim

        # Modality-specific CNNs (or lightweight ViTs if pretrained available)
        self.so2_cnn = nn.Conv2d(1, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.thb_cnn = nn.Conv2d(1, embed_dim, kernel_size=patch_size, stride=patch_size)
        #self.us_cnn  = nn.Conv2d(1, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        # CLS token (shared)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # Positional embeddings
        self.pos_embed = nn.Parameter(torch.randn(1, 1 + 2 * self.patch_dim, embed_dim))
        
        # Modality token embeddings (added per patch token depending on source)
        self.modality_tokens = nn.Parameter(torch.randn(2, 1, embed_dim))  # 0=SO2, 1=THb, 2=US

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification head
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )

        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([3.0]))

    def forward(self, x):  #so2, thb): #, us):
        so2, thb = x[0], x[1]
        B = so2.size(0)

        # 1. Patch embeddings via modality-specific CNNs
        so2_patches = rearrange(self.so2_cnn(so2), 'b c h w -> b (h w) c')
        thb_patches = rearrange(self.thb_cnn(thb), 'b c h w -> b (h w) c')
        #us_patches  = rearrange(self.us_cnn(us),  'b c h w -> b (h w) c')

        # 2. Add modality-specific tokens
        so2_patches += self.modality_tokens[0]
        thb_patches += self.modality_tokens[1]
        #us_patches  += self.modality_tokens[2]

        # 3. Concatenate all patches with CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, D]
        x = torch.cat([cls_tokens, so2_patches, thb_patches], dim=1)  # [B, 1 + 3*N, D]

        # 4. Add positional embedding
        x += self.pos_embed[:, :x.size(1), :]

        # 5. Transformer encoding
        x = self.transformer(x)

        # 6. Classification head on CLS token
        cls_output = x[:, 0]
        return self.mlp_head(cls_output)
    
    def compute_loss(self, x, y, x2_rad=None):
        y = y.float()  # Ensure targets are float for BCE loss
        if x2_rad is not None:
            score, tails = self.forward(x, x2_rad)
            loss = self.loss_fn(score, y) + sum(self.loss_fn(t, y) for t in tails)
        else:
            score = self.forward(x)
            loss = self.loss_fn(score, y) #+ 0.5 * self.loss_fn2(score, y)

        return loss

    def predict_on_loader(self, dataloader, threshold=0.5):
        self.eval()
        all_probs, all_targets = [], []

        device = next(self.parameters()).device

        with torch.no_grad():
            for batch in dataloader:
                if len(batch) == 2:
                    x, y = batch
                    x, y = x.to(device), y.to(device)
                    scores = self.forward(x)
                else:
                    x, x2, y = batch
                    x, x2, y = x.to(device), x2.to(device), y.to(device)
                    scores = self.forward([x, x2])

                probs = torch.sigmoid(scores)
                all_probs.append(probs.cpu())
                all_targets.append(y.cpu())

        return torch.cat(all_targets).numpy(), torch.cat(all_probs).numpy()
    


class ThreeModalTransformerWithRadiomics(nn.Module):
    def __init__(self, img_size=256, patch_size=16, embed_dim=256, num_heads=4, num_layers=6, num_classes=1, rad_dim = 2, dropout=0.1):
        super().__init__()

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans=1, embed_dim=embed_dim)
        self.num_patches = self.patch_embed.num_patches

        # Modality embeddings
        self.modality_embed = nn.Parameter(torch.zeros(3, 1, embed_dim))  # SO2, THb, US

        # Radiomics embedding layers
        self.rad_embed = nn.Linear(rad_dim, embed_dim)


        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Positional embedding
        total_tokens = 1 + (self.num_patches * 2) + 1  # +2 for radiomics
        self.pos_embed = nn.Parameter(torch.zeros(1, total_tokens, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([3.0] * num_classes))
        
    def forward(self, so2_img, thb_img, rad_feats):
        B = so2_img.size(0)

        cls_tokens = self.cls_token.expand(B, 1, -1)

        so2_tokens = self.patch_embed(so2_img) + self.modality_embed[0]
        thb_tokens = self.patch_embed(thb_img) + self.modality_embed[1]

        rad_token = self.rad_embed(rad_feats).unsqueeze(1) + self.modality_embed[2]

        tokens = torch.cat([cls_tokens, so2_tokens, thb_tokens, rad_token], dim=1)
        tokens += self.pos_embed[:, :tokens.size(1)]

        out = self.transformer(tokens)
        cls_out = self.norm(out[:, 0])
        logits = self.head(cls_out)
        return logits
    
    def compute_loss(self, x1, x2, x3, y):
        y = y.float()  # Ensure targets are float for BCE loss
        score = self.forward(x1, x2, x3)
        loss = self.loss_fn(score, y) #+ 0.5 * self.loss_fn2(score, y)
        return loss
    
    def predict_on_loader(self, dataloader, threshold=0.5):
        self.eval()
        all_probs, all_targets = [], []

        device = next(self.parameters()).device

        with torch.no_grad():
            for batch in dataloader:
                x1, x2, x3, y = batch
                x1, x2, x3, y = x1.to(device), x2.to(device), x3.to(device), y.to(device)
                scores = self.forward(x1, x2, x3)
               
                probs = torch.sigmoid(scores)
                all_probs.append(probs.cpu())
                all_targets.append(y.cpu())

        return torch.cat(all_targets).numpy(), torch.cat(all_probs).numpy()

    
if __name__ == "__main__":
    #model = MultiModalCancerClassifierWithAttention(num_modalities=2, out_dim=1, fusion_dim=64, backbone_name='resnet18', dropout_prob=0.0)
    model = ThreeModalTransformerWithRadiomics(img_size=256, patch_size=32, embed_dim=256, num_heads=4, num_layers=6, num_classes=1, rad_dim=2, dropout=0.1)
    img1 = torch.randn(8, 1, 256, 256)
    img2 = torch.randn(8, 1, 256, 256)
    img3 = torch.randn(8, 1, 256, 256)
    img4 = torch.randn(8, 1, 256, 256)
    rad = torch.randn(8, 2)  # Example radiomics features

    output = model(img1, img2, rad) #, img4])  # shape: (8,)

    print(output.shape)  # Should print torch.Size([8, 1])


    model = MultiModalTransformerClassifier(num_classes=1, img_size = 64) #)
    so2_input = torch.randn(8, 1, 64, 64)  # Batch of 8 SO2 images
    thb_input = torch.randn(8, 1, 64, 64)  # Batch of 8 THb images
    us_input = torch.randn(8, 1, 64, 64)   # Batch of 8 US images

    output = model([so2_input, thb_input]) #, us_input)
    print(output.shape)  # Should be [8, num_classes]


