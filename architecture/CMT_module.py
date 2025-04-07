import torch.nn as nn
import torch.nn.functional as F

class CrossModalFusion(nn.Module):
    def __init__(self, hidden_dim, num_heads=4, dropout=0.3):
        super().__init__()
        
        self.text_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.text_linear = nn.Linear(hidden_dim, hidden_dim)
        self.text_layer_norm = nn.LayerNorm(hidden_dim)

        self.audio_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.audio_linear = nn.Linear(hidden_dim, hidden_dim)
        self.audio_layer_norm = nn.LayerNorm(hidden_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, text_feat, audio_feat):
        text_feat = text_feat.unsqueeze(1)
        audio_feat = audio_feat.unsqueeze(1)

        audio_attn, _ = self.audio_attention(query=text_feat, key=audio_feat, value=audio_feat)
        audio_attn = self.audio_layer_norm(self.audio_linear(audio_attn) + text_feat)
        audio_attn = self.dropout(audio_attn).squeeze(1)

        text_attn, _ = self.text_attention(query=audio_feat, key=text_feat, value=text_feat)
        text_attn = self.text_layer_norm(self.text_linear(text_attn) + audio_feat)
        text_attn = self.dropout(text_attn).squeeze(1)

        return text_attn, audio_attn
