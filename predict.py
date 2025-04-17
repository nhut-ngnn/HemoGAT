import torch
import torch.optim as optim
import torch.nn as nn
from architecture.Model import MultiModalGNN
from architecture.Graph_constructed import load_dataset
from utils import set_seed, compute_metrics
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

set_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

test_path = 'feature/IEMOCAP_BERT_WAV2VEC_test.pkl'

test_data  = load_dataset(test_path,k_text=3, k_audio=5, device=device)

input_dim = test_data.x.shape[1]  
hidden_dim = 256
num_classes = 4

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = MultiModalGNN(input_dim, hidden_dim, num_classes).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=5e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=30, verbose=True)
criterion = nn.CrossEntropyLoss()

model_path = "saved_models/IEMOCAP_GNN_SER.pt"

model.load_state_dict(torch.load(model_path, map_location='cpu'))
print(model.parameters())
model.eval()
test_data = test_data.to(device)

with torch.no_grad():
    test_out = model(test_data.x, test_data.edge_index)
    test_pred = test_out.argmax(dim=1)

y_true = test_data.y.cpu().numpy()
y_pred = test_pred.cpu().numpy()

wa, ua , wf1, uf1 = compute_metrics(y_true, y_pred)
print(f"Test WA (Accuracy): {wa:.4f}, Test UA (Balanced Accuracy): {ua:.4f}, Test WF1: {wf1:.4f}, Test UF1: {uf1:.4f}")

conf_matrix = confusion_matrix(y_true, y_pred)
conf_matrix_norm = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]  

plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix_norm, annot=True, fmt=".2f", cmap="Blues", xticklabels=range(num_classes), yticklabels=range(num_classes))
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Normalized Confusion Matrix")
plt.show()