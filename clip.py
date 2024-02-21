import torch
import torch.nn as nn
from torch.nn import functional as F

# Your existing ViT components remain unchanged.

class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_dim):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        x = self.proj(x)  # [B, C, H, W] -> [B, E, H/P, W/P]
        x = x.flatten(2)  # [B, E, N]
        x = x.transpose(1, 2)  # [B, N, E]
        return x


class PositionalEmbedding(nn.Module):
    def __init__(self, n_patches, embed_dim):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))

    def forward(self, x):
        B = x.size(0)
        pos_embedding = self.pos_embed.repeat(B, 1, 1)
        return pos_embedding


class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, n_heads, n_layers, mlp_ratio, dropout):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, x):
        return self.encoder(x)


class ClassificationHead(nn.Module):
    def __init__(self, embed_dim, n_classes):
        super().__init__()
        self.classifier = nn.Linear(embed_dim, n_classes)

    def forward(self, x):
        return self.classifier(x[:, 0])

class VisionTransformer(nn.Module):
    def __init__(
        self,
        img_size,
        patch_size,
        in_channels,
        embed_dim,
        n_heads,
        n_layers,
        mlp_ratio,
        n_classes,
        dropout=0.1,
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = PositionalEmbedding(self.patch_embed.n_patches, embed_dim)
        self.transformer_encoder = TransformerEncoder(
            embed_dim, n_heads, n_layers, mlp_ratio, dropout
        )
        self.classification_head = ClassificationHead(embed_dim, n_classes)

    def forward(self, x):
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embed(x)
        x = self.transformer_encoder(x)
        x = self.classification_head(x)
        return x

class TextTransformerEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, n_heads, n_layers, mlp_ratio, dropout):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_embedding = nn.Parameter(torch.zeros(1, 1 + 512, embed_dim))  # Assuming a max sequence length of 512
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, input_ids):
        x = self.token_embedding(input_ids)  # [B, L] -> [B, L, E]
        B, L, _ = x.shape
        pos_embedding = self.positional_embedding[:, :L, :]  # Adjust positional embeddings to sequence length
        x = x + pos_embedding
        x = self.encoder(x)
        return x[:, 0]  # Returning only the [CLS] token's embedding

class CLIP(nn.Module):
    def __init__(
        self,
        img_size,
        patch_size,
        in_channels,
        embed_dim,
        n_heads,
        n_layers,
        mlp_ratio,
        vocab_size,
        dropout=0.1,
    ):
        super().__init__()
        # Vision encoder
        self.vision_transformer = VisionTransformer(
            img_size,
            patch_size,
            in_channels,
            embed_dim,
            n_heads,
            n_layers,
            mlp_ratio,
            n_classes=embed_dim,  # For CLIP, we output an embedding instead of class predictions
            dropout=dropout,
        )
        # Text encoder
        self.text_transformer = TextTransformerEncoder(
            vocab_size,
            embed_dim,
            n_heads,
            n_layers,
            mlp_ratio,
            dropout,
        )

    def forward(self, image, input_ids):
        image_features = self.vision_transformer(image)  # [B, E]
        text_features = self.text_transformer(input_ids)  # [B, E]

        # Normalize the features to unit length
        image_features = F.normalize(image_features, p=2, dim=-1)
        text_features = F.normalize(text_features, p=2, dim=-1)

        return image_features, text_features