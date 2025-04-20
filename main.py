import os
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from architecture.Model import MultiModalGNN
from architecture.Graph_constructed import load_dataset
from utils import set_seed, compute_metrics, train

seeds = [42, 123, 456, 789, 101112]
metrics = {'WA': [], 'UA': [], 'WF1': [], 'UF1': []}

train_path = 'feature/IEMOCAP_BERT_WAV2VEC_train.pkl'
valid_path = 'feature/IEMOCAP_BERT_WAV2VEC_val.pkl'
test_path  = 'feature/IEMOCAP_BERT_WAV2VEC_test.pkl'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
save_path = "saved_models"

for seed in seeds:
    print(f"\n=== Running seed {seed} ===")
    set_seed(seed)

    train_data = load_dataset(train_path,k_text=1, k_audio=3, device=device)
    valid_data = load_dataset(valid_path,k_text=1, k_audio=3, device=device)
    test_data  = load_dataset(test_path,k_text=1, k_audio=3, device=device)

    hidden_dim = 512
    num_classes = 4

    model = MultiModalGNN(hidden_dim, num_classes, num_layers=3).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=15)
    criterion = nn.CrossEntropyLoss()

    train(model, train_data, valid_data, optimizer, scheduler, criterion, epochs=100)

    model_path = os.path.join(save_path, f"IEMOCAP_HemoGAT.pt")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    model.eval()
    test_data = test_data.to(device)
    with torch.no_grad():
        test_out = model(test_data.text_x, test_data.audio_x, test_data.edge_index)
        test_pred = test_out.argmax(dim=1)

    y_true = test_data.y.cpu().numpy()
    y_pred = test_pred.cpu().numpy()
    wa, ua, wf1, uf1 = compute_metrics(y_true, y_pred)

    print(f"Seed {seed} - WA: {wa:.4f}, UA: {ua:.4f}, WF1: {wf1:.4f}, UF1: {uf1:.4f}")
    metrics['WA'].append(wa)
    metrics['UA'].append(ua)
    metrics['WF1'].append(wf1)
    metrics['UF1'].append(uf1)

print("\n=== Average Results over 5 seeds ===")
print(f"Avg WA:  {np.mean(metrics['WA']):.4f} ± {np.std(metrics['WA']):.4f}")
print(f"Avg UA:  {np.mean(metrics['UA']):.4f} ± {np.std(metrics['UA']):.4f}")
print(f"Avg WF1: {np.mean(metrics['WF1']):.4f} ± {np.std(metrics['WF1']):.4f}")
print(f"Avg UF1: {np.mean(metrics['UF1']):.4f} ± {np.std(metrics['UF1']):.4f}")
