import os
import json
import random
import numpy as np
import scipy.signal
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from preprocessing import process_DB_rawdata
import math
from model_find_peaks import detect_peaks_from_signal
from model_wearing_anomaly_detection_transformer import *

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def pulse_representation_encode(model, pulse_segment):
    with torch.no_grad():
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        pulse_segment = torch.tensor(pulse_segment).unsqueeze(0).unsqueeze(-1).to(device)
        src = model.encoder(pulse_segment) * math.sqrt(model.d_model)
        src = model.pos_encoder(src)
        src = src.permute(1, 0, 2)  # Transformer expects (seq_len, batch_size, d_model)
        latent_representation = model.transformer.encoder(src)
        latent_representation = latent_representation.permute(1, 0, 2)  # Convert back to (batch_size, seq_len, d_model)
    return latent_representation.squeeze(0)  # Remove the batch dimension

def pulse_representation_decode(model, latent_vector):
    with torch.no_grad():
        latent_vector = latent_vector.unsqueeze(0)  # Add the batch dimension
        memory = latent_vector.permute(1, 0, 2)  # Transformer expects (seq_len, batch_size, d_model)
        output = model.transformer.decoder(memory, memory)
        output = output.permute(1, 0, 2)  # Convert back to (batch_size, seq_len, d_model)
        pulse_segment = model.decoder(output).squeeze(-1)
    return pulse_segment.squeeze(0)  # Remove the batch dimension

def load_pulse_segments(data_folder):
    pulse_segments = []
    for root, dirs, files in os.walk(data_folder):
        for file in files:
            if file.endswith('.json'):
                with open(os.path.join(root, file), 'r') as f:
                    data = json.load(f)
                    signal = data['smoothed_data']
                    x_points = data['x_points']
                    for i in range(len(x_points) - 1):
                        pulse_segment = signal[x_points[i]:x_points[i+1]]
                        pulse_segments.append(pulse_segment)
    return pulse_segments

def main():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'torch device: {device}')

    d_model = 128
    nhead = 8
    num_encoder_layers = 2
    num_decoder_layers = 2

    model = TransformerAutoencoder(d_model, nhead, num_encoder_layers, num_decoder_layers).to(device)
    
    if os.path.exists('transformer_autoencoder.pth'):
        model.load_state_dict(torch.load('transformer_autoencoder.pth'))
    model.eval()

    # Load pulse segments from labeled_DB
    data_folder = 'labeled_DB'
    pulse_segments = load_pulse_segments(data_folder)

    # Convert pulse segments to latent vectors
    latent_vectors = []
    for pulse_segment in pulse_segments:
        latent_vector = pulse_representation_encode(model, pulse_segment)
        latent_vectors.append(latent_vector.cpu().numpy())
        input(f'Pulse segment: {pulse_segment}, Pulse segment len: {len(pulse_segment)}, Latent vector shape: {latent_vector.shape}, Latent vector: {latent_vector}')

    # Apply t-SNE for visualization
    tsne = TSNE(n_components=2, random_state=42)
    latent_vectors_2d = tsne.fit_transform(latent_vectors)

    # Plot the t-SNE visualization
    plt.figure(figsize=(8, 6))
    plt.scatter(latent_vectors_2d[:, 0], latent_vectors_2d[:, 1])
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.title('t-SNE Visualization of Latent Space')
    plt.show()




if __name__ == '__main__':
    main()
