'''
Baseline model:
    target_len = 100
    model = EPGBaselinePulseAutoencoder(target_len).to(device)

'''


import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.models import densenet121
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import numpy as np
from sklearn.manifold import TSNE
from sklearn.svm import OneClassSVM
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
import matplotlib.pyplot as plt
import os
import sys
import json
import random
import scipy
from torchviz import make_dot
from preprocessing import process_DB_rawdata, get_json_files, add_noise_with_snr, add_gaussian_noise_torch, PulseDataset
from model_find_peaks import detect_peaks_from_signal
from tqdm import tqdm
import math
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import h5py
def save_encoded_data(encoded_data, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for path, vectors in encoded_data.items():
        output_path = os.path.join(output_dir, path.replace('/', '_') + '.h5')
        with h5py.File(output_path, 'w') as f:
            f.create_dataset('data', data=vectors)
            f.attrs['original_path'] = path
# class PulseDataset(Dataset):
#     def __init__(self, json_files):
#         self.data = []
#         self.pulse_lengths = []
#         self.load_data(json_files)

#     def load_data(self, json_files):
#         for json_file in json_files:
#             try:
#                 with open(json_file, 'r') as f:
#                     json_data = json.load(f)
#                     if json_data['anomaly_list']:
#                         continue
#                     signal = json_data['smoothed_data']
#                     original_sample_rate = json_data.get('sample_rate', 100)
#                     x_points = json_data['x_points']
#                     signal = self.normalize(signal)
#                     for j in range(len(x_points) - 1):
#                         pulse_start = x_points[j]
#                         pulse_end = x_points[j + 1]
#                         if (pulse_end - pulse_start) < 0.2 * original_sample_rate:
#                             continue
#                         pulse = signal[pulse_start:pulse_end]
#                         pulse_length = (pulse_end - pulse_start) / original_sample_rate  # 計算脈衝的原始長度
#                         if (pulse_end - pulse_start) != 100:
#                             pulse = scipy.signal.resample(pulse, int(len(pulse) * 100 / (pulse_end - pulse_start)))
#                         if len(pulse) > 0:
#                             self.data.append(pulse)
#                             self.pulse_lengths.append(pulse_length)  # 將脈衝的原始長度添加到pulse_lengths列表中
#             except Exception as e:
#                 print(f'Error: {e}, json_file: {json_file}')
#                 continue
#     def normalize(self, data):
#         return (data - np.mean(data)) / (np.std(data) + 1e-8)

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         pulse = self.data[idx]
#         # input(f'pulse: {pulse.shape}')
#         return torch.tensor(pulse, dtype=torch.float32)
    

def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            pulse = batch.to(device)
            output, _ = model(pulse)
            loss = criterion(output, pulse)
            total_loss += loss.item()
    total_loss /= len(dataloader)
    return total_loss

def train_autoencoder(model, train_dataloader, test_dataloader, optimizer, criterion, device, epochs=2000, save_interval=1):
    model.train()
    min_loss = float('inf')
    model_path = 'pulse_interpolate_autoencoder2.pth'
    
    # 嘗試載入已有的模型參數
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print(f"Loaded model parameters from {model_path}")

    for epoch in range(epochs):
        total_loss = 0
        for batch in train_dataloader:
            pulse = batch.to(device)
            optimizer.zero_grad()
            for i in range(30):
                noise_pulse = add_gaussian_noise_torch(pulse)
                output, _ = model(noise_pulse)
            # output, _ = model(pulse)
                loss = criterion(output, pulse)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
        total_loss /= len(train_dataloader)
        
        test_loss = evaluate_model(model, test_dataloader, criterion, device)
        print(f"Epoch [{epoch+1}/{epochs}], Training Loss: {total_loss:.10f}, Testing Loss: {test_loss:.10f}")
        
        # Save model parameters if test loss decreases
        if (epoch + 1) % save_interval == 0 and test_loss < min_loss * 0.95:
            min_loss = test_loss
            torch.save(model.state_dict(), model_path)
            print(f"Saved model parameters at epoch {epoch+1}, Testing Loss: {test_loss:.10f}, path: {model_path}")
        # Save model parameters every save_interval epochs
        # elif (epoch + 1) % save_interval == 0:
        #     torch.save(model.state_dict(), model_path)

# 提取潛在向量並附加時間長度信息
def extract_latent_vectors(model, dataloader, device):
    latent_vectors = []
    original_lengths = []

    with torch.no_grad():
        for batch in dataloader:
            resampled_pulses, lengths = batch
            resampled_pulses = resampled_pulses.to(device)

            latent = model.encoder(resampled_pulses)
            latent_vectors.append(latent.cpu().numpy())
            original_lengths.extend(lengths)

    latent_vectors = np.concatenate(latent_vectors, axis=0)
    original_lengths = np.array(original_lengths)

    # 將原始時間長度信息附加到潛在向量上
    latent_vectors_with_lengths = np.hstack((latent_vectors, original_lengths.reshape(-1, 1)))

    return latent_vectors_with_lengths

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class AttentionPooling(nn.Module):
    def __init__(self, d_model):
        super(AttentionPooling, self).__init__()
        self.attention = nn.Linear(d_model, 1)

    def forward(self, x):
        weights = F.softmax(self.attention(x), dim=1)
        pooled = torch.sum(weights * x, dim=1)
        return pooled

class TransposedConvDecoder(nn.Module):
    def __init__(self, latent_dim, output_size=100):
        super(TransposedConvDecoder, self).__init__()
        self.latent_dim = latent_dim
        self.output_size = output_size
        
        self.initial_linear = nn.Linear(latent_dim, 256)
        
        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose1d(256, 128, kernel_size=4, stride=2, padding=1),  # Output: (128, 2)
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),  # Output: (64, 4)
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),   # Output: (32, 8)
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose1d(32, 16, kernel_size=4, stride=2, padding=1),   # Output: (16, 16)
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose1d(16, 8, kernel_size=4, stride=2, padding=1),    # Output: (8, 32)
            nn.BatchNorm1d(8),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose1d(8, 4, kernel_size=4, stride=2, padding=1),     # Output: (4, 64)
            nn.BatchNorm1d(4),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose1d(4, 1, kernel_size=4, stride=2, padding=1),     # Output: (1, 128)
        )
        
        self.final_fc = nn.Linear(128, output_size)  # Fully connected layer to adjust the output size
        
    def forward(self, x):
        # print(f'Input x: {x.shape}')
        
        x = self.initial_linear(x)
        x = x.unsqueeze(2)  # Add a dimension for Conv1d, resulting in shape (batch_size, 256, 1)
        # print(f'After initial linear x: {x.shape}')
        
        x = self.deconv_layers(x)
        # print(f'After deconv layers x: {x.shape}')
        
        x = x.squeeze(2)  # Remove the extra dimension for the fully connected layer
        # print(f'After squeeze x: {x.shape}')
        
        x = self.final_fc(x)
        # print(f'After final fully connected layer x: {x.shape}')
        
        return x
    

class TransformerAutoencoder(nn.Module):
    def __init__(self, d_model, nhead, num_encoder_layers, num_decoder_layers, latent_dim, dim_feedforward, dropout, seq_length):
        super(TransformerAutoencoder, self).__init__()
        self.d_model = d_model
        self.seq_length = seq_length
        self.encoder = nn.Linear(1, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True),
            num_encoder_layers,
            norm=nn.LayerNorm(d_model)
        )
        # Replace AttentionPooling with GlobalAveragePooling or GlobalMaxPooling
        self.global_pooling = nn.AdaptiveAvgPool1d(1)  # Replace with GlobalMaxPool1d if desired
        self.to_latent = nn.Linear(d_model, latent_dim)
        self.from_latent = nn.Linear(latent_dim, d_model)
        self.pos_decoder = PositionalEncoding(d_model)
        # Add DenseNetDecoder
        self.decoder = TransposedConvDecoder(latent_dim)
        #計算decoder的參數量
        self.decoder_param_num = sum(p.numel() for p in self.decoder.parameters())
        print(f'Total number of parameters in decoder: {self.decoder_param_num}')
        self.activation = nn.LeakyReLU()
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.from_latent.weight.data.uniform_(-initrange, initrange)
        self.from_latent.bias.data.zero_()

    def forward(self, src):
        src = self.activation(self.encoder(src.unsqueeze(-1)) * math.sqrt(self.d_model))
        src = self.pos_encoder(src)
        encoder_output = self.transformer_encoder(src)
        pooled_output = self.global_pooling(encoder_output.permute(0, 2, 1))
        pooled_output = pooled_output.squeeze(-1)
        latent = self.activation(self.to_latent(pooled_output))
        # Decode part
        output = self.decoder(latent)
        return output, latent
    

class UNet1DAutoencoder(nn.Module):
    def __init__(self, latent_dim, input_size=100):
        super(UNet1DAutoencoder, self).__init__()
        self.encoder1 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2)  # Output: (16, 50)
        )
        self.encoder2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2)  # Output: (32, 25)
        )
        self.encoder3 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(5)  # Output: (64, 5)
        )
        self.encoder4 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(5)  # Output: (128, 1)
        )

        self.to_latent = nn.Linear(128, latent_dim)
        self.from_latent = nn.Linear(latent_dim, 128)

        self.decoder4 = nn.Sequential(
            nn.ConvTranspose1d(128, 64, kernel_size=5, stride=5),  # Output: (64, 5)
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True)
        )
        self.decoder3 = nn.Sequential(
            nn.ConvTranspose1d(64, 32, kernel_size=5, stride=5),   # Output: (32, 25)
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True)
        )
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose1d(32, 16, kernel_size=2, stride=2),   # Output: (16, 50)
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True)
        )
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose1d(16, 1, kernel_size=2, stride=2),    # Output: (1, 100)
            nn.Sigmoid()
        )
        self.init_weights()
    
    def init_weights(self):
        for m in self.modules():
            if type(m) in {
                nn.Conv1d,
                nn.ConvTranspose1d,
                nn.Linear,
            }:
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias) 

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension: (batch_size, 1, 100)
        
        # Encoding path
        x1 = self.encoder1(x)
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)
        x4 = self.encoder4(x3)
        
        # Latent space
        x = x4.squeeze(-1)
        latent = F.relu(self.to_latent(x))
        x = F.relu(self.from_latent(latent))
        x = x.unsqueeze(-1)

        # Decoding path
        x = self.decoder4(x)
        x = self.decoder3(x + x3)  # Skip connection
        x = self.decoder2(x + x2)  # Skip connection
        x = self.decoder1(x + x1)  # Skip connection
        
        return x.squeeze(1), latent

def reconstruct_pulse_signal(signal, sample_rate, x_points):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet1DAutoencoder(latent_dim=32).to(device)
    model.eval()
    
    model.eval()
    
    # 記錄原始信號的均值和標準差,用於反向標準化
    signal_mean = np.mean(signal)
    signal_std = np.std(signal)
    
    # 對信號進行預處理
    normalized_signal = (signal - signal_mean) / (signal_std + 1e-8)
    
    #reconstructed_signal copy signal
    reconstructed_signal = np.copy(normalized_signal)
    latent_vectors = []
    
    # 逐個脈衝進行重構
    for j in range(len(x_points) - 1):
        pulse_start = x_points[j]
        pulse_end = x_points[j + 1]
        
        pulse = normalized_signal[pulse_start:pulse_end]
        pulse_length = pulse_end - pulse_start
        print(f'Pulse length: {pulse_length}')
        if len(pulse) != 100:
            pulse = scipy.signal.resample(pulse, 100)
        print(f'resample pulse len: {len(pulse)}')
        pulse_tensor = torch.tensor(pulse, dtype=torch.float32).unsqueeze(0).to(device)
        
        with torch.no_grad():
            reconstructed_pulse, latent = model(pulse_tensor)
        
        reconstructed_pulse = reconstructed_pulse.squeeze().cpu().numpy()
        latent = latent.squeeze().cpu().numpy()
        
        # 將重構後的脈衝還原為原始長度
        reconstructed_pulse = scipy.signal.resample(reconstructed_pulse, pulse_length)
        print(f'resample reconstructed pulse len: {len(reconstructed_pulse)}')
        # 將重構後的脈衝放回到原始信號的對應位置
        reconstructed_signal[pulse_start:pulse_end] = reconstructed_pulse
        
        # 將潛在向量添加到列表中
        latent_vectors.append(np.append(latent, pulse_length / sample_rate))

    # 對重構後的信號進行反向標準化
    reconstructed_signal = (reconstructed_signal * (signal_std + 1e-8)) + signal_mean
    
    # 將潛在向量轉換為 numpy 數組
    # latent_vectors = np.array(latent_vectors)
    for i in range(len(latent_vectors)):
        print(f'latent vector {i}: {latent_vectors[i]}')
    return reconstructed_signal#, latent_vectors

def train_bert(model, dataloader, num_epochs = 300):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(num_epochs):
        for pulses in dataloader:
            optimizer.zero_grad()
            input_ids = pulses
            attention_mask = (input_ids != 0).float()
            reconstructed_pulse = model(input_ids, attention_mask)
            loss = criterion(reconstructed_pulse, input_ids)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch + 1}, Loss: {loss.item()}')




def collate_fn(batch):
    batch = [seq for seq in batch if seq is not None]
    sequences = [seq for seq in batch]
    padded_sequences = nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=0.0)
    return padded_sequences

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.attention = nn.Linear(hidden_dim, hidden_dim)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, lstm_output):
        # lstm_output: [batch_size, seq_len, hidden_dim]
        attention_weights = self.attention(lstm_output)
        attention_weights = self.softmax(attention_weights)
        context_vector = attention_weights * lstm_output
        context_vector = torch.sum(context_vector, dim=1)
        return context_vector

class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_layers):
        super(LSTMAutoencoder, self).__init__()
        self.encoder_lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.decoder_lstm = nn.LSTM(hidden_dim, input_dim, num_layers, batch_first=True)
        self.to_latent = nn.Linear(hidden_dim, latent_dim)
        self.from_latent = nn.Linear(latent_dim, hidden_dim)
        self.attention = Attention(hidden_dim)

    def forward(self, x):
        # Encoder Part
        lstm_output, (hidden, _) = self.encoder_lstm(x)
        context_vector = self.attention(lstm_output)
        latent_vector = self.to_latent(context_vector)  # [batch_size, latent_dim]

        # Decoder Part
        decoder_input = self.from_latent(latent_vector).unsqueeze(1).repeat(1, x.size(1), 1)  # [batch_size, seq_len, hidden_dim]
        output, _ = self.decoder_lstm(decoder_input)
        return output

class LSTMVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_layers):
        super(LSTMVAE, self).__init__()
        self.encoder_lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.decoder_lstm = nn.LSTM(hidden_dim, input_dim, num_layers, batch_first=True)
        self.to_mu = nn.Linear(hidden_dim, latent_dim)
        self.to_logvar = nn.Linear(hidden_dim, latent_dim)
        self.from_latent = nn.Linear(latent_dim, hidden_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        lstm_output, (hidden, _) = self.encoder_lstm(x)
        hidden = hidden[-1]  # 取最後一個時間步的隱藏狀態
        mu = self.to_mu(hidden)  # [batch_size, latent_dim]
        logvar = self.to_logvar(hidden)  # [batch_size, latent_dim]
        z = self.reparameterize(mu, logvar)  # [batch_size, latent_dim]

        decoder_input = self.from_latent(z).unsqueeze(1).repeat(1, x.size(1), 1)  # [batch_size, seq_len, hidden_dim]
        output, _ = self.decoder_lstm(decoder_input)
        return output, mu, logvar

def loss_function(recon_x, x, mu, logvar):
    BCE = nn.MSELoss()(recon_x, x)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

def trainVAE(model, dataloader, optimizer, device, epochs=10000):
    model_path = 'pulse_vae.pt'
    min_loss = float('inf')
    model.to(device)
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in dataloader:
            pulses = batch.to(device)
            optimizer.zero_grad()
            output, mu, logvar = model(pulses)
            loss = loss_function(output, pulses, mu, logvar)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        total_loss /= len(dataloader)
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {total_loss}')

        if total_loss < min_loss * 0.95:
            min_loss = total_loss
            torch.save(model.state_dict(), model_path)
            print(f'Saved model to {model_path}, epoch loss: {total_loss}')

class EPGBaselinePulseAutoencoder(nn.Module):
    def __init__(self, target_len, hidden_dim=50, latent_dim=30):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(target_len, hidden_dim),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        self.dec = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(hidden_dim, target_len)
        )

    def forward(self, x):
        z = self.enc(x)
        pred = self.dec(z)
        return pred, z

def predict_latent_vector_list(model, signal, sample_rate, peaks):
    target_len = 100
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 重採樣信號
    resample_ratio = 1.0
    if sample_rate != 100:
        resample_ratio = 100 / sample_rate
        signal = scipy.signal.resample(signal, int(len(signal) * resample_ratio))
        peaks = [int(p * resample_ratio) for p in peaks]  # 調整peaks索引

    # 全局標準化
    mean = np.mean(signal)
    std = np.std(signal)
    signal = (signal - mean) / std
    # print(f'peaks: {peaks}')
    # 逐拍重建
    latent_vector_list = []
    for i in range(len(peaks) - 1):
        start_idx = peaks[i]
        end_idx = peaks[i + 1]
        pulse = signal[start_idx:end_idx]
        pulse_length = end_idx - start_idx  # 記錄脈衝的原始長度

        if pulse_length > 1:
            # 插值到目標長度
            interp_func = scipy.interpolate.interp1d(np.arange(pulse_length), pulse, kind='linear', fill_value="extrapolate")
            pulse_resampled = interp_func(np.linspace(0, pulse_length - 1, target_len))
            pulse_tensor = torch.tensor(pulse_resampled, dtype=torch.float32).unsqueeze(0).to(device)

            with torch.no_grad():
                _, latent_vector = model(pulse_tensor)
                latent_vector = latent_vector.squeeze().cpu().numpy()
                latent_vector = np.concatenate([latent_vector, np.array([pulse_length/100])], axis=0)

                # print(f'i:{i}, start_idx:{start_idx}, latent_vector:{latent_vector}')
                latent_vector_list.append(latent_vector)
            # 將重建的脈衝還原為原始長度

    #計算前後latent_vector之間的相似程度
    similarity_list = []
    distance_list = []
    for i in range(len(latent_vector_list) - 1):
        this_vec = latent_vector_list[i]
        next_vec = latent_vector_list[i + 1]

        similarity = np.dot(this_vec, next_vec)/(np.linalg.norm(this_vec) * np.linalg.norm(next_vec))
        similarity_list.append(similarity) 
        distance = np.linalg.norm(this_vec - next_vec)
        distance_list.append(distance)
    # print(f'similarity_list:{similarity_list}')
    # print(f'distance_list:{distance_list}')

    return latent_vector_list

def predict_reconstructed_signal(signal, sample_rate, peaks):
    target_len = 100
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = 'pulse_interpolate_autoencoder.pth'
    model = EPGBaselinePulseAutoencoder(target_len).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # 重採樣信號
    resample_ratio = 1.0
    if sample_rate != 100:
        resample_ratio = 100 / sample_rate
        signal = scipy.signal.resample(signal, int(len(signal) * resample_ratio))
        peaks = [int(p * resample_ratio) for p in peaks]  # 調整peaks索引

    # 全局標準化
    mean = np.mean(signal)
    std = np.std(signal)
    signal = (signal - mean) / std
    print(f'peaks: {peaks}')
    # 逐拍重建
    latent_vector_list = []
    reconstructed_signal = np.copy(signal)
    for i in range(len(peaks) - 1):
        start_idx = peaks[i]
        end_idx = peaks[i + 1]
        pulse = signal[start_idx:end_idx]
        pulse_length = end_idx - start_idx  # 記錄脈衝的原始長度

        if pulse_length > 1:
            # 插值到目標長度
            interp_func = scipy.interpolate.interp1d(np.arange(pulse_length), pulse, kind='linear', fill_value="extrapolate")
            pulse_resampled = interp_func(np.linspace(0, pulse_length - 1, target_len))
            pulse_tensor = torch.tensor(pulse_resampled, dtype=torch.float32).unsqueeze(0).to(device)

            with torch.no_grad():
                reconstructed_pulse, latent_vector = model(pulse_tensor)
                reconstructed_pulse = reconstructed_pulse.squeeze().cpu().numpy()
                latent_vector = latent_vector.squeeze().cpu().numpy()
                latent_vector = np.concatenate([latent_vector, np.array([pulse_length/100])], axis=0)

                # print(f'i:{i}, start_idx:{start_idx}, latent_vector:{latent_vector}')
                latent_vector_list.append(latent_vector)
            # 將重建的脈衝還原為原始長度
            interp_func_reconstructed = scipy.interpolate.interp1d(np.linspace(0, target_len - 1, target_len), reconstructed_pulse, kind='linear', fill_value="extrapolate")
            reconstructed_pulse_resampled = interp_func_reconstructed(np.linspace(0, target_len - 1, pulse_length))
            reconstructed_signal[start_idx:end_idx] = reconstructed_pulse_resampled

    #計算前後latent_vector之間的相似程度
    similarity_list = []
    distance_list = []
    for i in range(len(latent_vector_list) - 1):
        this_vec = latent_vector_list[i]
        next_vec = latent_vector_list[i + 1]

        similarity = np.dot(this_vec, next_vec)/(np.linalg.norm(this_vec) * np.linalg.norm(next_vec))
        similarity_list.append(similarity) 
        distance = np.linalg.norm(this_vec - next_vec)
        distance_list.append(distance)
    print(f'similarity_list:{similarity_list}')
    print(f'distance_list:{distance_list}')


    # 反標準化
    reconstructed_signal = reconstructed_signal * std + mean

    # 根據原始採樣率調整重構信號的長度
    original_length = int(len(reconstructed_signal) / resample_ratio)
    reconstructed_signal = scipy.signal.resample(reconstructed_signal, original_length)

    return reconstructed_signal

def encode_pulses(model, json_files, target_len=100, sample_rate=100, device='cuda'):
    model.eval()
    encoded_data = {}

    for json_file in tqdm(json_files, desc="Processing files"):
        try:
            with open(json_file, 'r') as f:
                json_data = json.load(f)
                if json_data['anomaly_list']:
                    continue
                
                signal = json_data['smoothed_data']
                original_sample_rate = json_data.get('sample_rate', 100)
                x_points = json_data['x_points']

                if original_sample_rate != sample_rate:
                    num_samples = int(len(signal) * sample_rate / original_sample_rate)
                    signal = scipy.signal.resample(signal, num_samples)
                    x_points = [int(x * sample_rate / original_sample_rate) for x in x_points]
                
                signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-8)

                latent_vectors = []
                for j in range(len(x_points) - 1):
                    pulse_start = x_points[j]
                    pulse_end = x_points[j + 1]
                    pulse = signal[pulse_start:pulse_end]
                    if len(pulse) > 40:
                        interp_func = scipy.interpolate.interp1d(np.arange(len(pulse)), pulse, kind='linear')
                        pulse_resampled = interp_func(np.linspace(0, len(pulse) - 1, target_len))
                        pulse_tensor = torch.tensor(pulse_resampled, dtype=torch.float32).unsqueeze(0).to(device)

                        with torch.no_grad():
                            _, latent_vector = model(pulse_tensor)
                        
                        latent_vector = latent_vector.squeeze().cpu().numpy()
                        latent_vector = np.concatenate([latent_vector, np.array([len(pulse)/100])])
                        latent_vectors.append(latent_vector)

                # 获取相对路径
                relative_path = os.path.relpath(json_file, 'labeled_DB')
                encoded_data[relative_path] = np.array(latent_vectors)

        except Exception as e:
            print(f'Error in processing {json_file}: {e}')

    return encoded_data

# def save_encoded_data(encoded_data, output_file):
#     with h5py.File(output_file, 'w') as f:
#         for path, vectors in encoded_data.items():
#             print(f'Saving {path}, vectors.shape: {vectors.shape}')
#             f.create_dataset(path, data=vectors)

def main():
    data_folder = 'labeled_DB'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'torch device: {device}')
    json_files = get_json_files(data_folder)  # 实现一个函数来获取所有的JSON文件路径

    # 設置參數
    input_dim = 1
    hidden_dim = 128
    latent_dim = 20
    num_layers =2
    batch_size = 32
    lr = 1e-4
    target_len = 200
    #'pulse_interpolate_autoencoder.pth' : 一般的pulse autoencoder target_len = 100
    #'pulse_interpolate_autoencoder2.pth' : denoise autoencoder  target_len = 200



    # 加載並劃分數據集
    dataset = PulseDataset(json_files, target_len)
    # dataset = PulseDataset(json_files)
    train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # 初始化模型和優化器
    # model = EPGBaselinePulseAutoencoder(target_len=200).to(device)
    # # model = LSTMVAE(input_dim, hidden_dim, latent_dim, num_layers).to(device)
    # criterion = nn.MSELoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(f'Total number of model parameters: {trainable_params}, model:{model}') 
    # train_autoencoder(model, train_dataloader, test_dataloader, optimizer, criterion, device)

    # model_path = 'pulse_interpolate_autoencoder.pth'
    # model.load_state_dict(torch.load(model_path))
    model_path = 'pulse_interpolate_autoencoder.pth'
    model = EPGBaselinePulseAutoencoder(100).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    encoded_data = {} 
    for json_file in json_files:
        try:
            #get relative path to data_folder
            if json_file.startswith('labeled_DB'):
                relative_path = os.path.relpath(json_file, 'labeled_DB').split(',')[0].replace('\\', '  ').replace('.', ' ')
                # input(f'relative_path: {relative_path}')
            with open(json_file, 'r') as f:
                json_data = json.load(f)
                signal = json_data['smoothed_data']
                original_sample_rate = json_data.get('sample_rate', 100)
                x_points = json_data['x_points']
                latent_vector_list = predict_latent_vector_list(model, signal, original_sample_rate, x_points) 
                print(f'latent_vector_list: {latent_vector_list}')
                encoded_data[relative_path] = np.array(latent_vector_list)
        except Exception as e:
            print(f'Error in loading {json_file}: {e}')   
    # save_encoded_data(encoded_data, 'latent_vectors')

    # 統計每個維度的分佈範圍
    all_latent_vectors = []
    for vectors in encoded_data.values():
        all_latent_vectors.extend(vectors)
    all_latent_vectors = np.array(all_latent_vectors)

    min_values = np.min(all_latent_vectors, axis=0)
    max_values = np.max(all_latent_vectors, axis=0)
    mean_values = np.mean(all_latent_vectors, axis=0)
    std_values = np.std(all_latent_vectors, axis=0)

    print(f"Min values: {min_values}")
    print(f"Max values: {max_values}")
    print(f"Mean values: {mean_values}")
    print(f"Std values: {std_values}")

    # # 编码所有脉冲
    # encoded_data = encode_pulses(model, json_files, target_len, device=device)

    # # 保存编码后的数据
    # save_encoded_data(encoded_data, 'encoded_pulses.h5')


    # trainVAE(model, train_dataloader, optimizer, device)                  
    # # 訓練完成後,使用編碼器提取特徵
    # model.eval()
    # train_features = []
    # test_features = []

    # with torch.no_grad():
    #     for seq in train_dataloader:
    #         if seq.size(0) == 0:
    #             continue
    #         seq = seq.to(device)  # 確保數據形狀為 [batch_size, sequence_length, input_dim]
    #         z = model(seq)
    #         train_features.append(z.cpu().numpy())

    #     for seq in test_dataloader:
    #         if seq.size(0) == 0:
    #             continue
    #         seq = seq.to(device)  # 確保數據形狀為 [batch_size, sequence_length, input_dim]
    #         z = model(seq)
    #         test_features.append(z.cpu().numpy())

    # train_features = np.concatenate(train_features, axis=0) 
    # test_features = np.concatenate(test_features, axis=0)

if __name__ == '__main__':
    main()