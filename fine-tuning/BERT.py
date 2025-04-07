import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer
from torch.optim import AdamW
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd
import numpy as np

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils import set_seed

set_seed(42)

class TextDataset(Dataset):
    def __init__(self, metadata_path, tokenizer, max_length=128):
        self.metadata = pd.read_csv(metadata_path)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.metadata)

    def _tokenize_text(self, text):
        return self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

    def __getitem__(self, idx):
        text = self.metadata.iloc[idx]['raw_text'] 
        label = self.metadata.iloc[idx]['label']

        aug1 = self._create_augmentation(text)
        aug2 = self._create_augmentation(text)

        encoded1 = self._tokenize_text(aug1)
        encoded2 = self._tokenize_text(aug2)

        return {
            'input_ids1': encoded1['input_ids'].squeeze(),
            'attention_mask1': encoded1['attention_mask'].squeeze(),
            'input_ids2': encoded2['input_ids'].squeeze(),
            'attention_mask2': encoded2['attention_mask'].squeeze()
        }

    def _create_augmentation(self, text):
        words = text.split()
        if len(words) <= 1:
            return text

        dropout_prob = 0.15
        augmented_words = [word for word in words if np.random.random() > dropout_prob]

        if len(augmented_words) == 0:
            augmented_words = [words[np.random.randint(len(words))]]

        return ' '.join(augmented_words)


class BERTEmbeddingModel(nn.Module):
    def __init__(self, embedding_dim=768, projection_dim=256):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, projection_dim)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled = outputs.pooler_output
        projection = self.projection(pooled)
        return pooled, projection


class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1, z2):
        z1_norm = F.normalize(z1, dim=1)
        z2_norm = F.normalize(z2, dim=1)

        N = z1_norm.size(0)
        z_all = torch.cat([z1_norm, z2_norm], dim=0)

        sim = torch.mm(z_all, z_all.t()) / self.temperature
        sim_mask = torch.eye(2 * N, dtype=torch.bool, device=sim.device)
        sim.masked_fill_(sim_mask, -float('inf'))

        labels = torch.arange(N, device=sim.device)
        labels = torch.cat([labels + N, labels])

        loss = F.cross_entropy(sim, labels)
        return loss


def train_embeddings(metadata_path, num_epochs=10, batch_size=32, learning_rate=2e-5):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BERTEmbeddingModel()

    # Wrap the model in DataParallel for multi-GPU support
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    dataset = TextDataset(metadata_path, tokenizer)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=learning_rate)
    criterion = NTXentLoss()

    best_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            input_ids1 = batch['input_ids1'].to(device)
            attention_mask1 = batch['attention_mask1'].to(device)
            input_ids2 = batch['input_ids2'].to(device)
            attention_mask2 = batch['attention_mask2'].to(device)

            _, z1 = model(input_ids1, attention_mask1)
            _, z2 = model(input_ids2, attention_mask2)

            loss = criterion(z1, z2)
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}, Average Loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'model_state_dict': model.module.state_dict() if torch.cuda.device_count() > 1 else model.state_dict(),
                'loss': best_loss,
                'epoch': epoch
            }, '/home/minhnhut/Graph-for-SER/best_bert_embeddings.pt')

    return model


def extract_embeddings(model, text, tokenizer):
    encoded = tokenizer(
        text,
        padding='max_length',
        truncation=True,
        max_length=128,
        return_tensors='pt'
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        embeddings, _ = model(encoded['input_ids'].to(device), encoded['attention_mask'].to(device))

    return embeddings


if __name__ == "__main__":
    metadata_path = "/home/minhnhut/Graph-for-SER/metadata/IEMOCAP_metadata_train.csv"
    model = train_embeddings(
        metadata_path=metadata_path,
        num_epochs=10,
        batch_size=32,
        learning_rate=2e-5
    )
