import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbed(nn.Module):
    """
    Split 1D spectral data into patches and embed them
    """
    def __init__(self, seq_length, patch_size, embed_dim):
        super().__init__()
        self.seq_length = seq_length
        self.patch_size = patch_size
        self.n_patches = (seq_length // patch_size)
        self.proj = nn.Conv1d(1, embed_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        # x shape: [batch_size, 1, seq_length]
        # Return shape: [batch_size, n_patches, embed_dim]
        x = self.proj(x)  # [batch_size, embed_dim, n_patches]
        x = x.transpose(1, 2)  # [batch_size, n_patches, embed_dim]
        return x

class Attention(nn.Module):
    def __init__(self, dim, n_heads=8, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
    def forward(self, x):
        # x shape: [batch_size, seq_length, dim]
        batch_size, seq_length, dim = x.shape
        
        qkv = self.qkv(x).reshape(batch_size, seq_length, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each has shape [batch_size, n_heads, seq_length, head_dim]
        
        # Compute attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [batch_size, n_heads, seq_length, seq_length]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(batch_size, seq_length, dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, n_heads, mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, n_heads=n_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(in_features=dim, hidden_features=int(dim * mlp_ratio), drop=drop)
        
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class ViT(nn.Module):
    def __init__(self, seq_length=2150, patch_size=10, in_channels=1, embed_dim=128, 
                 depth=6, n_heads=8, mlp_ratio=4., qkv_bias=True, 
                 drop_rate=0.1, attn_drop_rate=0.1, device='cpu'):
        super().__init__()
        
        self.patch_embed = PatchEmbed(seq_length, patch_size, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        n_patches = seq_length // patch_size
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(drop_rate)
        
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, n_heads=n_heads, mlp_ratio=mlp_ratio, 
                  qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, 1)
        
        # Initialize parameters
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)
        
        self.device = device
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    def apply_dropout(self, apply_dropout):
        # A method to enable/disable dropout for evaluation vs. training
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.training = apply_dropout
    
    def forward(self, x):
        # x shape: [batch_size, 1, seq_length]
        batch_size = x.shape[0]
        
        # Convert to patches
        x = self.patch_embed(x)  # [batch_size, n_patches, embed_dim]
        
        # Add class token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # [batch_size, 1 + n_patches, embed_dim]
        
        # Add position embeddings
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Extract CLS token for classification
        x = self.norm(x)
        x = x[:, 0]  # Take only the class token
        
        # Classification head
        x = self.head(x)
        return x