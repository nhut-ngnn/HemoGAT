import torch.nn as nn
import torch_geometric.nn as pyg_nn

class GATLayers(nn.Module):
    def __init__(self, hidden_dim, heads=4, num_layers=3, dropout=0.3):
        super(GATLayers, self).__init__()

        self.conv1 = pyg_nn.GATConv(hidden_dim * 2, hidden_dim, heads=heads, concat=True)

        self.convs = nn.ModuleList([
            pyg_nn.GATConv(hidden_dim * heads, hidden_dim, heads=heads, concat=True)
            for _ in range(num_layers - 2)
        ])

        self.conv_last = pyg_nn.GATConv(hidden_dim * heads, hidden_dim, heads=1, concat=False)

        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)
        self.batch_norm = nn.BatchNorm1d(hidden_dim * heads)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.batch_norm(x)
        x = self.relu(x)

        for conv in self.convs:
            residual = x
            x = conv(x, edge_index)
            x = self.batch_norm(x)
            x = self.relu(x)
            x = self.dropout(x)
            x = x + residual

        x = self.conv_last(x, edge_index)
        return x
