import torch
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train(model, train_data, valid_data, optimizer, scheduler, criterion, epochs=150):
    model.train()
    train_data = train_data.to(device)
    valid_data = valid_data.to(device)

    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(train_data.text_x, train_data.audio_x, train_data.edge_index)
        loss = criterion(out, train_data.y)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_out = model(valid_data.text_x, valid_data.audio_x, valid_data.edge_index)
            val_loss = criterion(val_out, valid_data.y)

        scheduler.step(val_loss)

        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {loss.item():.4f}, Valid Loss: {val_loss.item():.4f}")

def compute_metrics(y_true, y_pred):
    wa = balanced_accuracy_score(y_true, y_pred)  
    ua = accuracy_score(y_true, y_pred)  
    wf1 = f1_score(y_true, y_pred, average="weighted") 
    uf1 = f1_score(y_true, y_pred, average="macro") 
    return wa, ua, wf1, uf1 

def plot_and_save_roc(y_true, y_pred, y_score, num_classes, save_path):
    y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    if num_classes == 4:
        class_names = {
            0: "Angry",
            1: "Happy",
            2: "Sad",
            3: "Neutral"
        }
    elif num_classes == 7:
      class_names = {
            0: "Neural",
            1: "Joy",
            2: "Anger",
            3: "Sadness",
            4: "Disgust",
            5: "Fear",
            6: "Surprise"}
    else:
        class_names = {i: f"Class {i}" for i in range(num_classes)}

    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(8, 6))
    for i in range(num_classes):
        plt.plot(fpr[i], tpr[i], label=f'{class_names[i]} (AUC = {roc_auc[i]:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel(r'$\bf{False\ Positive\ Rate}$')
    plt.ylabel(r'$\bf{True\ Positive\ Rate}$')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=1000)
    plt.close()
    print(f"ROC curve saved to {save_path}")
def get_class_names(num_classes):
    if num_classes == 4:
        return {
            0: "Angry",
            1: "Happy",
            2: "Sad",
            3: "Neutral"
        }
    elif num_classes == 7:
        return {
            0: "Neutral",
            1: "Joy",
            2: "Anger",
            3: "Sadness",
            4: "Disgust",
            5: "Fear",
            6: "Surprise"
        }
    else:
        raise ValueError(f"Unsupported num_classes={num_classes}. Only 4 or 7 are supported.")
