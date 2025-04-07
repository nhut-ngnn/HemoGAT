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
        try:
            waveform, sample_rate = torchaudio.load(file_path)
            
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                waveform = resampler(waveform)
                
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            return waveform.squeeze()
        except Exception as e:
            print(f"Error loading file {file_path}: {e}")
            return torch.zeros(self.segment_length)
    
    def _get_random_segment(self, waveform):
        if waveform.size(0) > self.segment_length:
            start_idx = torch.randint(0, waveform.size(0) - self.segment_length, (1,))
            waveform = waveform[start_idx:start_idx + self.segment_length]
        else:
            padding = self.segment_length - waveform.size(0)
            waveform = F.pad(waveform, (0, padding))
            
        return waveform
    
    def __getitem__(self, idx):
        audio_path = self.metadata.iloc[idx]['audio_file']
        try:
            waveform = self._load_and_preprocess_audio(audio_path)
            segment1 = self._get_random_segment(waveform)
            segment2 = self._get_random_segment(waveform)
            
            processed1 = self.processor(
                segment1.numpy(),
                sampling_rate=16000,
                return_tensors="pt"
            )
            
            processed2 = self.processor(
                segment2.numpy(),
                sampling_rate=16000,
                return_tensors="pt"
            )
            
            return {
                'input_values1': processed1['input_values'].squeeze(),
                'input_values2': processed2['input_values'].squeeze()
            }
        except Exception as e:
            print(f"Error processing index {idx}, file {audio_path}: {e}")
            return {
                'input_values1': torch.zeros(self.segment_length),
                'input_values2': torch.zeros(self.segment_length)
            }


class AudioEmbeddingModel(nn.Module):
    def __init__(self, embedding_dim=768, projection_dim=256):
        super().__init__()
        self.wav2vec = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, projection_dim)
        )
        
    def forward(self, input_values):
        outputs = self.wav2vec(input_values)
        hidden_states = outputs.last_hidden_state
        pooled = torch.mean(hidden_states, dim=1)
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
        sim_mask = torch.eye(2*N, dtype=torch.bool, device=sim.device)
        sim.masked_fill_(sim_mask, -float('inf'))
        
        labels = torch.arange(N, device=sim.device)
        labels = torch.cat([labels + N, labels])
        
        loss = F.cross_entropy(sim, labels)
        return loss

def train_embeddings(metadata_path, num_epochs=10, batch_size=32, learning_rate=1e-4):
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
    model = AudioEmbeddingModel()
    
    dataset = AudioDataset(metadata_path, processor)
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
            input_values1 = batch['input_values1'].to(device)
            input_values2 = batch['input_values2'].to(device)
            
            _, z1 = model(input_values1)
            _, z2 = model(input_values2)
            
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
                'model_state_dict': model.state_dict(),
                'loss': best_loss,
                'epoch': epoch
            }, 'best_wav2vec_embeddings.pt')
    
    return model

def extract_embeddings(model, audio_path, processor):
    waveform, sample_rate = torchaudio.load(audio_path)
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
        waveform = resampler(waveform)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    processed = processor(
        waveform.squeeze().numpy(),
        sampling_rate=16000,
        return_tensors="pt"
    )
    
    model.eval()
    with torch.no_grad():
        embeddings, _ = model(processed['input_values'])
    
    return embeddings

if __name__ == "__main__":
    metadata_path = "/home/nhut-minh-nguyen/Documents/FuzzyFusion-SER/FlexibleMMSER/metadata/MELD_metadata_train.csv"
    model = train_embeddings(
        metadata_path=metadata_path,
        num_epochs=10,
        batch_size=32,
        learning_rate=1e-4
    )  