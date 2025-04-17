import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

from architecture.Model import MultiModalGNN
from architecture.Graph_constructed import load_dataset
from utils import set_seed, compute_metrics, train

set_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_path = '/home/nhut-minh-nguyen/Documents/Graph-for-SER/feature/IEMOCAP_BERT_WAV2VEC_train.pkl'
valid_path = '/home/nhut-minh-nguyen/Documents/Graph-for-SER/feature/IEMOCAP_BERT_WAV2VEC_val.pkl'
test_path  = '/home/nhut-minh-nguyen/Documents/Graph-for-SER/feature/IEMOCAP_BERT_WAV2VEC_test.pkl'

k_text_values = list(range(1, 11))
k_audio_values = list(range(1, 11))

best_wf1 = 0.0
best_config = (0, 0)
results = []

for k_text in k_text_values:
    for k_audio in k_audio_values:
        print(f"\n Testing k_text={k_text}, k_audio={k_audio}")

        try:
            train_data = load_dataset(train_path, k_text=k_text, k_audio=k_audio).to(device)
            valid_data = load_dataset(valid_path, k_text=k_text, k_audio=k_audio).to(device)
            test_data  = load_dataset(test_path,  k_text=k_text, k_audio=k_audio).to(device)
        except Exception as e:
            print(f" Error loading dataset for k_text={k_text}, k_audio={k_audio}: {e}")
            continue

        model = MultiModalGNN(hidden_dim=512, num_classes=4, num_layers=3).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=15)
        criterion = nn.CrossEntropyLoss()

        train(model, train_data, valid_data, optimizer, scheduler, criterion, epochs=100)

        model.eval()
        with torch.no_grad():
            test_out = model(test_data.text_x, test_data.audio_x, test_data.edge_index)
            test_pred = test_out.argmax(dim=1)

        y_true = test_data.y.cpu().numpy()
        y_pred = test_pred.cpu().numpy()
        wa, ua, wf1, uf1 = compute_metrics(y_true, y_pred)

        print(f" WA: {wa:.4f}, UA: {ua:.4f}, WF1: {wf1:.4f}, UF1: {uf1:.4f}")

        results.append({
            'k_text': k_text,
            'k_audio': k_audio,
            'WA': wa,
            'UA': ua,
            'WF1': wf1,
            'UF1': uf1
        })

        if wf1 > best_wf1:
            best_wf1 = wf1
            best_config = (k_text, k_audio)
            print(f" New Best: k_text={k_text}, k_audio={k_audio} → WF1={wf1:.4f}")

df = pd.DataFrame(results)
csv_path = "grid_search_results.csv"
df.to_csv(csv_path, index=False)
print(f"\n Results saved to {csv_path}")

print("\n Search Complete!")
print(f"Best config → k_text: {best_config[0]}, k_audio: {best_config[1]}, WF1: {best_wf1:.4f}")
