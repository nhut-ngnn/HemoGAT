import os
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import argparse

from architecture.Model import MultiModalGNN
from architecture.Graph_constructed import load_dataset
from utils import set_seed, compute_metrics, train, plot_and_save_roc

parser = argparse.ArgumentParser(description="Train MultiModalGNN with flexible configs")

parser.add_argument("--data_dir", type=str, required=True, help="Folder containing feature .pkl files")
parser.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g., MELD, IEMOCAP)")
parser.add_argument("--save_path", type=str, default="saved_models", help="Path to save model and plots")

parser.add_argument("--num_classes", type=int, required=True, help="Number of classes in the dataset")
parser.add_argument("--k_text", type=int, default=2, help="K-hop for text modality")
parser.add_argument("--k_audio", type=int, default=8, help="K-hop for audio modality")
parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
parser.add_argument("--hidden_dim", type=int, default=512, help="Hidden dimension size")

args = parser.parse_args()

train_path = os.path.join(args.data_dir, f"{args.dataset}_BERT_WAV2VEC_train.pkl")
valid_path = os.path.join(args.data_dir, f"{args.dataset}_BERT_WAV2VEC_val.pkl")
test_path  = os.path.join(args.data_dir, f"{args.dataset}_BERT_WAV2VEC_test.pkl")

print(f"[INFO] Using dataset: {args.dataset}")
print(f"[INFO] Train path: {train_path}")
print(f"[INFO] Valid path: {valid_path}")
print(f"[INFO] Test path: {test_path}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

seeds = [42, 50, 103, 128, 896]
metrics = {'WA': [], 'UA': [], 'WF1': [], 'UF1': []}

for seed in seeds:
    print(f"\n=== Running seed {seed} ===")
    set_seed(seed)

    train_data = load_dataset(train_path, k_text=args.k_text, k_audio=args.k_audio, device=device)
    valid_data = load_dataset(valid_path, k_text=args.k_text, k_audio=args.k_audio, device=device)
    test_data  = load_dataset(test_path, k_text=args.k_text, k_audio=args.k_audio, device=device)

    model = MultiModalGNN(args.hidden_dim, args.num_classes, num_layers=3).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=15)
    criterion = nn.CrossEntropyLoss()
    
    print("\n[INFO] Model Parameter Summary:")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters: {total_params}")
    print(f"Trainable Parameters: {trainable_params}\n")

    train(model, train_data, valid_data, optimizer, scheduler, criterion, epochs=args.epochs)

    os.makedirs(args.save_path, exist_ok=True)
    model_path = os.path.join(args.save_path, f"{args.dataset}_MultiModalGNN_seed{seed}.pt")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    model.eval()
    test_data = test_data.to(device)
    with torch.no_grad():
        test_out = model(test_data.text_x, test_data.audio_x, test_data.edge_index)
        test_pred = test_out.argmax(dim=1)
        test_prob = torch.softmax(test_out, dim=1).cpu().numpy()

    y_true = test_data.y.cpu().numpy()
    y_pred = test_pred.cpu().numpy()
    wa, ua, wf1, uf1 = compute_metrics(y_true, y_pred)

    print(f"Seed {seed} - WA: {wa:.4f}, UA: {ua:.4f}, WF1: {wf1:.4f}, UF1: {uf1:.4f}")
    metrics['WA'].append(wa)
    metrics['UA'].append(ua)
    metrics['WF1'].append(wf1)
    metrics['UF1'].append(uf1)

    roc_save_path = os.path.join(args.save_path, f"ROC_{args.dataset}_seed{seed}.png")
    plot_and_save_roc(y_true, y_pred, test_prob, args.num_classes, roc_save_path)

print("\n=== Average Results over 5 seeds ===")
print(f"Avg WA:  {np.mean(metrics['WA']):.4f} ± {np.std(metrics['WA']):.4f}")
print(f"Avg UA:  {np.mean(metrics['UA']):.4f} ± {np.std(metrics['UA']):.4f}")
print(f"Avg WF1: {np.mean(metrics['WF1']):.4f} ± {np.std(metrics['WF1']):.4f}")
print(f"Avg UF1: {np.mean(metrics['UF1']):.4f} ± {np.std(metrics['UF1']):.4f}")
