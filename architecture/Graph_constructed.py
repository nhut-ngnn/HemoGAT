import torch
import pickle
import numpy as np
from torch_geometric.data import Data
from scipy.spatial import KDTree

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def build_dual_knn_graph_cosine(text_embeds, audio_embeds, k_text, k_audio):
    def normalize(x):
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        return x / np.clip(norms, 1e-10, None)

    text_np = normalize(text_embeds.cpu().numpy())
    audio_np = normalize(audio_embeds.cpu().numpy())

    num_nodes = text_np.shape[0]
    edge_index = set()

    for i in range(num_nodes):
        edge_index.add((i, i))

        text_sim = np.dot(text_np, text_np[i])
        top_text = np.argsort(-text_sim)[1:k_text+1]  
        for j in top_text:
            edge_index.add((i, j))

        audio_sim = np.dot(audio_np, audio_np[i])
        top_audio = np.argsort(-audio_sim)[1:k_audio+1]
        for j in top_audio:
            edge_index.add((i, j))

    edge_index = torch.tensor(list(edge_index), dtype=torch.long).T 
    return edge_index

def load_pkl(file_path):
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    return data 
def prepare_graph_data(data_list, k_text, k_audio):
    text_features = []
    audio_features = []
    labels = []

    for data in data_list:
        if not isinstance(data, dict):
            raise ValueError(f"Expected dict but got {type(data)}")

        if 'text_embed' not in data or 'audio_embed' not in data or 'label' not in data:
            print(f"Missing keys in data: {data.keys()}")
            continue

        text_embed = torch.tensor(data['text_embed'], dtype=torch.float32).squeeze()
        audio_embed = torch.tensor(data['audio_embed'], dtype=torch.float32).squeeze()
        label = torch.tensor(data['label'], dtype=torch.long)

        text_features.append(text_embed.numpy())
        audio_features.append(audio_embed.numpy())
        labels.append(label.item())

    text_features = torch.from_numpy(np.array(text_features)).float()   
    audio_features = torch.from_numpy(np.array(audio_features)).float() 
    labels = torch.tensor(labels, dtype=torch.long)  

    x = torch.cat([text_features, audio_features], dim=1)  
    # edge_index = build_knn_graph_cosine(x, k=k)
    edge_index = build_dual_knn_graph_cosine(text_embeds=text_features, audio_embeds=audio_features, k_text=k_text, k_audio=k_audio)

    return Data(
        text_x=text_features,
        audio_x=audio_features,
        edge_index=edge_index,
        y=labels
    )

def load_dataset(path, k_text, k_audio, device=None):
    data = prepare_graph_data(load_pkl(path), k_text, k_audio)
    
    if device is not None:
        data.text_x = data.text_x.to(device)
        data.audio_x = data.audio_x.to(device)
        data.edge_index = data.edge_index.to(device)
        data.y = data.y.to(device)

    return data