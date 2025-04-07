import torch
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

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
    