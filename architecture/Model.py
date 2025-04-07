import torch
import torch.nn as nn
from architecture.GAT_module import GATLayers
from architecture.CMT_module import CrossModalFusion


class MultiModalGNN(nn.Module):
    def __init__(self, hidden_dim=512, num_classes=4, dropout=0.3, heads=4, num_layers=3):
        super(MultiModalGNN, self).__init__()

        self.text_projection = nn.Sequential(
            nn.Linear(768, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.audio_projection = nn.Sequential(
            nn.Linear(768, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.cross_fusion = CrossModalFusion(hidden_dim, num_heads=heads, dropout=dropout)
        self.gnn = GATLayers(hidden_dim, heads=heads, num_layers=num_layers, dropout=dropout)

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, text_x, audio_x, edge_index):
        text_feat = self.text_projection(text_x)
        audio_feat = self.audio_projection(audio_x)

        gnn_input = torch.cat([text_feat, audio_feat], dim=1)
        graph_feat = self.gnn(gnn_input, edge_index)

        fused_text, fused_audio = self.cross_fusion(text_feat, audio_feat)

        combined = torch.cat([graph_feat, fused_text, fused_audio], dim=1)
        out = self.mlp(combined)
        return out
