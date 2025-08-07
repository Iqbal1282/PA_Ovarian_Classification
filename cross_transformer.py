import torch
import torch.nn as nn
from einops import rearrange

class MultiModalTransformerClassifier(nn.Module):
    def __init__(self, img_size=64, patch_size=8, embed_dim=256, num_heads=4, num_layers=6, num_classes=8):
        super().__init__()
        
        self.patch_dim = (img_size // patch_size) ** 2
        self.patch_embed_dim = embed_dim

        # Modality-specific CNNs (or lightweight ViTs if pretrained available)
        self.so2_cnn = nn.Conv2d(1, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.thb_cnn = nn.Conv2d(1, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.us_cnn  = nn.Conv2d(1, embed_dim, kernel_size=patch_size, stride=patch_size)
        
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

    def forward(self, so2, thb): #, us):
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
        x = torch.cat([cls_tokens, so2_patches, thb_patches], dim = 1) #, us_patches], dim=1)  # [B, 1 + 3*N, D]

        # 4. Add positional embedding
        x += self.pos_embed[:, :x.size(1), :]

        # 5. Transformer encoding
        x = self.transformer(x)

        # 6. Classification head on CLS token
        cls_output = x[:, 0]
        return self.mlp_head(cls_output)


if __name__ == "__main__":
    # Example usage
    model = MultiModalTransformerClassifier(num_classes=1)
    so2_input = torch.randn(8, 1, 64, 64)  # Batch of 8 SO2 images
    thb_input = torch.randn(8, 1, 64, 64)  # Batch of 8 THb images
    us_input = torch.randn(8, 1, 64, 64)   # Batch of 8 US images

    output = model(so2_input, thb_input) #, us_input)
    print(output.shape)  # Should be [8, num_classes]