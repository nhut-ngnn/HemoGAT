import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from transformers import Wav2Vec2Model, Wav2Vec2Processor
from torch.optim import AdamW
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd
import numpy as np
import sys
import os

try:
    torchaudio.set_audio_backend("soundfile")
except Exception as e:
    print(f"Warning: Cannot set soundfile backend, using default. {e}")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils import set_seed

set_seed(42)

class AudioDataset(Dataset):
    def __init__(self, metadata_path, processor, segment_length=16000):
        self.metadata = pd.read_csv(metadata_path)
        self.processor = processor
        self.segment_length = segment_length

    def __len__(self):
        return len(self.metadata)

    def _load_and_preprocess_audio(self, file_path):
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return torch.zeros(self.segment_length)
        try:
            waveform, sample_rate = torchaudio.load(file_path)
            if waveform.numel() == 0:
                print(f"File is empty: {file_path}")
                return torch.zeros(self.segment_length)
            if sample_rate != 16000:
                waveform = torchaudio.transforms.Resample(sample_rate, 16000)(waveform)
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            return waveform.squeeze()
        except Exception as e:
            print(f"Error loading file {file_path}: {e}")
            return torch.zeros(self.segment_length)

    def _get_random_segment(self, waveform):
        if waveform.size(0) > self.segment_length:
            start_idx = torch.randint(0, waveform.size(0) - self.segment_length, (1,)).item()
            waveform = waveform[start_idx:start_idx + self.segment_length]
        else:
            waveform = F.pad(waveform, (0, self.segment_length - waveform.size(0)))
        return waveform

    def __getitem__(self, idx):
        audio_path = self.metadata.iloc[idx]['audio_file']
        waveform = self._load_and_preprocess_audio(audio_path)
        segment1 = self._get_random_segment(waveform).contiguous()
        segment2 = self._get_random_segment(waveform).contiguous()

        processed1 = self.processor(segment1.numpy(), sampling_rate=16000, return_tensors="pt")
        processed2 = self.processor(segment2.numpy(), sampling_rate=16000, return_tensors="pt")

        return {
            'input_values1': processed1['input_values'].squeeze(0),
            'input_values2': processed2['input_values'].squeeze(0)
        }

class AudioEmbeddingModel(nn.Module):
    def __init__(self, embedding_dim=768, projection_dim=512):
        super().__init__()
        self.wav2vec = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        self.wav2vec.gradient_checkpointing_disable() 
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, projection_dim)
        )

    def forward(self, input_values):
        outputs = self.wav2vec(input_values).last_hidden_state
        pooled = outputs.mean(dim=1)
        projected = self.projection(pooled)
        return pooled, projected

class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1, z2):
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        representations = torch.cat([z1, z2], dim=0)
        similarity_matrix = torch.mm(representations, representations.T) / self.temperature
        mask = torch.eye(similarity_matrix.size(0), device=similarity_matrix.device).bool()
        similarity_matrix.masked_fill_(mask, -float('inf'))
        batch_size = z1.size(0)
        labels = torch.arange(batch_size, device=similarity_matrix.device)
        labels = torch.cat([labels + batch_size, labels], dim=0)
        loss = F.cross_entropy(similarity_matrix, labels)
        return loss

def train_embeddings(metadata_path, num_epochs=30, batch_size=32, learning_rate=1e-4, patience=5):
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
    model = AudioEmbeddingModel()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Use all available GPUs
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for training.")
        model = nn.DataParallel(model)

    model = model.to(device)

    dataset = AudioDataset(metadata_path, processor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    optimizer = AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    criterion = NTXentLoss()

    best_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            input_values1 = batch['input_values1'].to(device)
            input_values2 = batch['input_values2'].to(device)

            _, z1 = model(input_values1)
            _, z2 = model(input_values2)

            loss = criterion(z1, z2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}, Average Loss: {avg_loss:.4f}")

        scheduler.step(avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss
            epochs_no_improve = 0
            os.makedirs('fine_tuning/model', exist_ok=True)
            torch.save({
                'model_state_dict': model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
                'loss': best_loss,
                'epoch': epoch
            }, 'fine_tuning/model/best_wav2vec_embeddings.pt')
            print(f"Model improved, checkpoint saved at epoch {epoch+1}.")
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epoch(s).")

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}.")
            break

    return model

if __name__ == "__main__":
    metadata_path = "metadata/IEMOCAP_metadata_train.csv"
    train_embeddings(metadata_path, num_epochs=10, batch_size=32, learning_rate=1e-4, patience=5)
