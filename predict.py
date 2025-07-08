import os
import torch
import numpy as np
import argparse
from sklearn.metrics import classification_report, precision_recall_fscore_support
from fvcore.nn import FlopCountAnalysis, parameter_count_table
from scipy import stats

from architecture.Model import MultiModalGNN
from architecture.Graph_constructed import load_dataset
from utils import compute_metrics, set_seed, get_class_names

parser = argparse.ArgumentParser(description="Predict with MultiModalGNN with FLOPs and flexible configs")

parser.add_argument("--data_dir", type=str, required=True, help="Folder containing feature .pkl files")
parser.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g., MELD, IEMOCAP)")
parser.add_argument("--save_path", type=str, default="saved_models", help="Path where models and reports are saved")
parser.add_argument("--num_classes", type=int, required=True, help="Number of classes in the dataset")
parser.add_argument("--k_text", type=int, default=2, help="K-hop for text modality")
parser.add_argument("--k_audio", type=int, default=8, help="K-hop for audio modality")
parser.add_argument("--hidden_dim", type=int, default=512, help="Hidden dimension size")

args = parser.parse_args()

test_path = os.path.join(args.data_dir, f"{args.dataset}_BERT_WAV2VEC_test.pkl")
print(f"[INFO] Using dataset: {args.dataset}")
print(f"[INFO] Test path: {test_path}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

seeds = [42, 50, 103, 128, 896]
metrics = {
    "WA": [], "UA": [], "WF1": [], "UF1": [],
    "Precision": [], "Recall": [], "F1": []
}

test_data = load_dataset(test_path, k_text=args.k_text, k_audio=args.k_audio, device=device).to(device)


print("\n[INFO] Calculating FLOPs Count...")
model = MultiModalGNN(args.hidden_dim, args.num_classes, num_layers=3).to(device)
model.eval()

with torch.no_grad():
    flops = FlopCountAnalysis(model, (test_data.text_x, test_data.audio_x, test_data.edge_index))
    print(f"\n[INFO] Total FLOPs: {flops.total():,.0f}")
    print(f"[INFO] Total GFLOPs: {flops.total() / 1e9:.3f} GFLOPs")

for seed in seeds:
    print(f"\n=== Evaluating Seed {seed} ===")
    set_seed(seed)

    model = MultiModalGNN(args.hidden_dim, args.num_classes, num_layers=3).to(device)
    model_path = os.path.join(args.save_path, f"{args.dataset}_MultiModalGNN_seed{seed}.pt")
    model.load_state_dict(torch.load(model_path))
    model.eval()

    with torch.no_grad():
        test_out = model(test_data.text_x, test_data.audio_x, test_data.edge_index)
        test_pred = test_out.argmax(dim=1)

    y_true = test_data.y.cpu().numpy()
    y_pred = test_pred.cpu().numpy()

    wa, ua, wf1, uf1 = compute_metrics(y_true, y_pred)
    metrics["WA"].append(wa)
    metrics["UA"].append(ua)
    metrics["WF1"].append(wf1)
    metrics["UF1"].append(uf1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    metrics["Precision"].append(precision)
    metrics["Recall"].append(recall)
    metrics["F1"].append(f1)

    class_names = get_class_names(args.num_classes)
    target_names = [class_names[i] for i in range(args.num_classes)]

    report = classification_report(y_true, y_pred, digits=4, target_names=target_names)
    print(f"Classification Report for Seed {seed}:\n{report}")

    os.makedirs(args.save_path, exist_ok=True)
    report_path = os.path.join(args.save_path, f"classification_report_{args.dataset}_seed{seed}.txt")
    with open(report_path, "w") as f:
        f.write(f"Classification Report for Seed {seed}:\n")
        f.write(report)

def compute_mean_std_ci(arr):
    mean = np.mean(arr)
    std = np.std(arr, ddof=1)
    t_value = stats.t.ppf(1 - 0.05/2, 4)
    ci95 = t_value * std / np.sqrt(len(arr))
    return mean, std, ci95

print("\n=== Summary over 5 Seeds ===")
for key in metrics.keys():
    mean, std, ci95 = compute_mean_std_ci(metrics[key])
    print(f"{key}: {mean:.4f} +/- {std:.4f} (95% CI +/- {ci95:.4f})")