'''
Baseline model:
    target_len = 100
    model = EPGBaselinePulseAutoencoder(target_len).to(device)

'''


import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.models import densenet121
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import numpy as np
from sklearn.manifold import TSNE
from sklearn.svm import OneClassSVM
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog

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

from datautils import save_encoded_data, predict_latent_vector_list, predict_encoded_dataset


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

# def trainVAE(model, dataloader, optimizer, device, epochs=10000):
#     model_path = 'pulse_vae.pt'
#     min_loss = float('inf')
#     model.to(device)
#     for epoch in range(epochs):
#         model.train()
#         total_loss = 0
#         for batch in dataloader:
#             pulses = batch.to(device)
#             optimizer.zero_grad()
#             output, mu, logvar = model(pulses)
#             loss = loss_function(output, pulses, mu, logvar)
#             loss.backward()
#             optimizer.step()
#             total_loss += loss.item()
#         total_loss /= len(dataloader)
#         print(f'Epoch {epoch + 1}/{epochs}, Loss: {total_loss}')

#         if total_loss < min_loss * 0.95:
#             min_loss = total_loss
#             torch.save(model.state_dict(), model_path)
#             print(f'Saved model to {model_path}, epoch loss: {total_loss}')

def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            pulse = batch.to(device)
            output, _ = model(pulse)
            loss = criterion(output, pulse)
            # recon_batch, mu, logvar = model(pulse)
            # loss = loss_function(recon_batch, pulse, mu, logvar)
            total_loss += loss.item()
    total_loss /= len(dataloader)
    return total_loss

def train_autoencoder(model, train_dataloader, test_dataloader, optimizer, criterion, device, model_path, epochs=1000, save_interval=1):
    model.train()
    min_loss = float('inf')
    # model_path = 'pulse_interpolate_autoencoder.pth'
    # model_path = 'pulse_interpolate_autoencoder_test.pth' #Epoch [1/2000], Training Loss: 0.4247541058, Testing Loss: 0.1003478393
    # model_path = 'pulse_interpolate_autoencoder2.pth'
    # model_path = 'pulse_interpolate_autoencoder_VAE.pth' #Epoch [1/2000], Training Loss: 12.8088734311, Testing Loss: 4.2140154323
    
    # 嘗試載入已有的模型參數
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print(f"Loaded model parameters from {model_path}")

    #Compute init loss
    init_loss = evaluate_model(model, test_dataloader, criterion, device)
    print(f"Init Testing Loss: {init_loss:.10f}")

    for epoch in range(epochs):
        total_loss = 0
        for batch in train_dataloader:
            pulse = batch.to(device)
            optimizer.zero_grad()
            output, _ = model(pulse)
            loss = criterion(output, pulse)
            # for i in range(10):
            #     noise_pulse = add_gaussian_noise_torch(pulse)
            #     output, _ = model(noise_pulse)
            # # output, _ = model(pulse)
            #     loss = criterion(output, pulse)
            # recon_batch, mu, logvar = model(pulse)
            # loss = loss_function(recon_batch, pulse, mu, logvar)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        total_loss /= len(train_dataloader)
        # total_loss /= 10
        test_loss = evaluate_model(model, test_dataloader, criterion, device)
        print(f"Epoch [{epoch+1}/{epochs}], Training Loss: {total_loss:.10f}, Testing Loss: {test_loss:.10f}")
        
        # Save model parameters if test loss decreases
        if save_interval != 0 and (epoch + 1) % save_interval == 0 and test_loss < min_loss * 0.95:
            min_loss = test_loss
            torch.save(model.state_dict(), model_path)
            print(f"Saved model parameters at epoch {epoch+1}, Testing Loss: {test_loss:.10f}")

class EPGBaselinePulseAutoencoder(nn.Module):
    def __init__(self, target_len, hidden_dim=30, latent_dim=30, dropout=0.9):#, latent_dim=30, dropout=0.5):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(target_len, hidden_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        self.dec = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(hidden_dim, target_len)
        )

    def forward(self, x):
        z = self.enc(x)
        pred = self.dec(z)
        return pred, z
    
    def encode(self, x):
        return self.enc(x)
    

class EPGBaselinePulseVAE(nn.Module):
    def __init__(self, target_len, hidden_dim=50, latent_dim=30):
        super(EPGBaselinePulseVAE, self).__init__()
        # Encoder
        self.fc1 = nn.Linear(target_len, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, latent_dim)  # 均值
        self.fc22 = nn.Linear(hidden_dim, latent_dim)  # 標準差
        
        # Decoder
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, target_len)
        
    def encode(self, x):
        h1 = torch.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        h3 = torch.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))  # 使用sigmoid確保輸出在[0, 1]範圍內
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# def loss_function(recon_x, x, mu, logvar):
#     BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
#     KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
#     return BCE + KLD




def predict_reconstructed_signal(signal, sample_rate, peaks):
    target_len = 100
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = 'pulse_interpolate_autoencoder2.pth' # target_len = 200
    model_path = 'pulse_interpolate_autoencoder.pth' # target_len = 100
    model_path = 'pulse_interpolate_autoencoder_test.pth' # target_len = 100 
    model_path = 'pulse_interpolate_autoencoder_0909_30dim.pth'
    model = EPGBaselinePulseAutoencoder(target_len).to(device)


    
    model.load_state_dict(torch.load(model_path))
    model.eval()

    #複製一個原始訊號的陣列值
    origin_signal = np.copy(signal)

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
    # print(f'similarity_list:{similarity_list}')
    # print(f'distance_list:{distance_list}')


    # 反標準化
    reconstructed_signal = reconstructed_signal * std + mean

    # 根據原始採樣率調整重構信號的長度
    original_length = int(len(reconstructed_signal) / resample_ratio)
    reconstructed_signal = scipy.signal.resample(reconstructed_signal, original_length)

    #計算原始號跟重構訊號的MAE
    mae = np.mean(np.abs(origin_signal - reconstructed_signal))
    print(f'MAE: {mae}')

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
class ImprovedCNNAutoencoder(nn.Module):
    def __init__(self, latent_dim=20, input_length=128):
        super().__init__()
        self.latent_dim = latent_dim
        
        # 更平滑的channel progression
        self.channels = [32, 48, 64, 96]
        
        # Encoder
        self.encoder = nn.Sequential(
            # Layer 1: 128 -> 64
            nn.Conv1d(1, self.channels[0], kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(self.channels[0]),
            nn.MaxPool1d(2),
            
            # Layer 2: 64 -> 32
            nn.Conv1d(self.channels[0], self.channels[1], kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(self.channels[1]),
            nn.MaxPool1d(2),
            
            # Layer 3: 32 -> 16
            nn.Conv1d(self.channels[1], self.channels[2], kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(self.channels[2]),
            nn.MaxPool1d(2),
            
            # Layer 4: 16 -> 8
            nn.Conv1d(self.channels[2], self.channels[3], kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(self.channels[3]),
            nn.MaxPool1d(2),
        )
        
        # 計算flatten後的維度
        self.flatten_dim = self.channels[3] * (input_length // 16)  # 8 = 2^4 (4次下採樣)
        
        # 更平滑的latent轉換
        self.latent_encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flatten_dim, self.flatten_dim // 2),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(self.flatten_dim // 2),
            nn.Dropout(0.2),
            nn.Linear(self.flatten_dim // 2, self.flatten_dim // 4),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(self.flatten_dim // 4),
            nn.Linear(self.flatten_dim // 4, latent_dim),
        )
        
        # 同樣平滑的latent解碼
        self.latent_decoder = nn.Sequential(
            nn.Linear(latent_dim, self.flatten_dim // 4),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(self.flatten_dim // 4),
            nn.Linear(self.flatten_dim // 4, self.flatten_dim // 2),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(self.flatten_dim // 2),
            nn.Dropout(0.2),
            nn.Linear(self.flatten_dim // 2, self.flatten_dim),
            nn.LeakyReLU(0.2),
            nn.Unflatten(1, (self.channels[3], input_length // 16))
        )
        
        # Decoder with skip connections
        self.decoder = nn.Sequential(
            # Layer 1: 8 -> 16
            nn.ConvTranspose1d(self.channels[3], self.channels[2], 
                             kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(self.channels[2]),
            
            # Layer 2: 16 -> 32
            nn.ConvTranspose1d(self.channels[2], self.channels[1], 
                             kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(self.channels[1]),
            
            # Layer 3: 32 -> 64
            nn.ConvTranspose1d(self.channels[1], self.channels[0], 
                             kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(self.channels[0]),
            
            # Layer 4: 64 -> 128
            nn.ConvTranspose1d(self.channels[0], self.channels[0] // 2, 
                             kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(self.channels[0] // 2),
            
            # Final layer
            nn.Conv1d(self.channels[0] // 2, 1, kernel_size=1),
            nn.Tanh()
        )
        
        # 初始化權重
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def encode(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        features = self.encoder(x)
        return self.latent_encoder(features)
    
    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        
        # Encoding
        features = self.encoder(x)
        latent = self.latent_encoder(features)
        
        # Decoding
        decoded = self.latent_decoder(latent)
        output = self.decoder(decoded)
        
        return output.squeeze(1), latent

# 訓練相關的輔助函數
def get_optimizer(model, lr=1e-3):
    return torch.optim.AdamW(model.parameters(), 
                           lr=lr, 
                           weight_decay=1e-4,
                           betas=(0.9, 0.999))

def get_scheduler(optimizer, num_epochs):
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                                     T_max=num_epochs,
                                                     eta_min=1e-6)
class CNNAutoencoder(nn.Module):
    def __init__(self, latent_dim=20): #Default input length=128
        super().__init__()
        self.latent_dim = latent_dim

        # Encoder
        self.encoder = nn.Sequential(
            # Layer 1: 128 -> 64
            nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            # Layer 2: 64 -> 32
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            # Layer 3: 32 -> 16
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            # Layer 4: 16 -> 8
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

        # Latent space
        self.latent_encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim),
            nn.ReLU()
        )

        self.latent_decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256 * 8),
            nn.ReLU(),
            nn.Unflatten(1, (256, 8))
        )

        # Decoder
        self.decoder = nn.Sequential(
            # Layer 1: 8 -> 16
            nn.ConvTranspose1d(256, 128, kernel_size=4, stride=2, padding=1),
            # nn.BatchNorm1d(128),
            nn.ReLU(),
            
            # Layer 2: 16 -> 32
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),
            # nn.BatchNorm1d(64),
            nn.ReLU(),
            
            # Layer 3: 32 -> 64
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),
            # nn.BatchNorm1d(32),
            nn.ReLU(),
            
            # Layer 4: 64 -> 128
            nn.ConvTranspose1d(32, 16, kernel_size=4, stride=2, padding=1),
            # nn.BatchNorm1d(16),
            nn.ReLU(),
            
            # Final layer
            nn.Conv1d(16, 1, kernel_size=1),
            nn.Tanh()  # 或使用 Sigmoid，取決於數據範圍
        )

    def encode(self, x):
        """返回 latent_dim 維度的潛在向量"""
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        features = self.encoder(x)
        return self.latent_encoder(features)

    def forward(self, x):
        # 輸入處理
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # [B, L] -> [B, 1, L]
        
        # Encoding
        features = self.encoder(x)         # [B, 256, 8]
        latent = self.latent_encoder(features)  # [B, latent_dim]
        
        # Decoding
        decoded = self.latent_decoder(latent)  # [B, 256, 8]
        output = self.decoder(decoded)     # [B, 1, 128]
        
        return output.squeeze(1), latent

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x

class UNetAutoencoder(nn.Module):
    def __init__(self, target_len=128, latent_dim=30):
        super().__init__()
        self.target_len = target_len
        self.latent_dim = latent_dim
        
        # Encoder path (下採樣)
        self.enc1 = ConvBlock(1, 64)       # 128 -> 128
        self.enc2 = ConvBlock(64, 128)     # 64 -> 64
        self.enc3 = ConvBlock(128, 256)    # 32 -> 32
        self.enc4 = ConvBlock(256, 512)    # 16 -> 16
        self.pool = nn.MaxPool1d(2)

        # Latent space encoder (將特徵壓縮到 latent_dim)
        self.latent_encoder = nn.Sequential(
            nn.AdaptiveAvgPool1d(8),
            nn.Flatten(),
            nn.Linear(512 * 8, latent_dim),
            nn.ReLU()
        )

        # Latent space decoder (從 latent_dim 恢復到解碼器需要的維度)
        self.latent_decoder = nn.Sequential(
            nn.Linear(latent_dim, 512 * 8),
            nn.ReLU(),
            nn.Unflatten(1, (512, 8))
        )

        # Decoder path (上採樣)
        self.up4 = nn.ConvTranspose1d(512, 512, kernel_size=4, stride=2, padding=1)  # 8 -> 16
        self.dec4 = ConvBlock(512 + 512, 256)  # 輸入: 512(up4) + 512(skip) = 1024

        self.up3 = nn.ConvTranspose1d(256, 256, kernel_size=4, stride=2, padding=1)  # 16 -> 32
        self.dec3 = ConvBlock(256 + 256, 128)  # 輸入: 256(up3) + 256(skip) = 512

        self.up2 = nn.ConvTranspose1d(128, 128, kernel_size=4, stride=2, padding=1)  # 32 -> 64
        self.dec2 = ConvBlock(128 + 128, 64)   # 輸入: 128(up2) + 128(skip) = 256

        self.up1 = nn.ConvTranspose1d(64, 64, kernel_size=4, stride=2, padding=1)    # 64 -> 128
        self.dec1 = ConvBlock(64 + 64, 32)     # 輸入: 64(up1) + 64(skip) = 128

        self.final = nn.Conv1d(32, 1, kernel_size=1)

    def encode(self, x):
        """返回 latent_dim 維度的潛在向量"""
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
            
        # Encoder path
        e1 = self.enc1(x)
        p1 = self.pool(e1)
        e2 = self.enc2(p1)
        p2 = self.pool(e2)
        e3 = self.enc3(p2)
        p3 = self.pool(e3)
        e4 = self.enc4(p3)
        p4 = self.pool(e4)      # [B, 512, 8]
        
        # Get latent vector
        latent = self.latent_encoder(p4)  # [B, latent_dim]
        return latent

    def forward(self, x):
        # 輸入處理
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # [B, L] -> [B, 1, L]
        
        # Encoder path
        e1 = self.enc1(x)        # [B, 64, 128]
        p1 = self.pool(e1)       # [B, 64, 64]
        
        e2 = self.enc2(p1)       # [B, 128, 64]
        p2 = self.pool(e2)       # [B, 128, 32]
        
        e3 = self.enc3(p2)       # [B, 256, 32]
        p3 = self.pool(e3)       # [B, 256, 16]
        
        e4 = self.enc4(p3)       # [B, 512, 16]
        p4 = self.pool(e4)       # [B, 512, 8]

        # Get latent vector
        latent = self.latent_encoder(p4)  # [B, latent_dim]
        # print(f'latent.shape: {latent.shape}')
        # Decode from latent vector
        decoded = self.latent_decoder(latent)  # [B, 512, 8]
        # print(f'decoded.shape: {decoded.shape}')
        # Decoder path with skip connections
        d4 = self.up4(decoded)   # [B, 512, 16]
        # print(f'd4.shape: {d4.shape}')
        d4 = torch.cat([d4, e4], dim=1)  # [B, 1024, 16]
        # print(f'd4.shape: {d4.shape}')
        d4 = self.dec4(d4)       # [B, 256, 16]
        # print(f'd4.shape: {d4.shape}')
        d3 = self.up3(d4)        # [B, 256, 32]
        d3 = torch.cat([d3, e3], dim=1)  # [B, 512, 32]
        d3 = self.dec3(d3)       # [B, 128, 32]

        d2 = self.up2(d3)        # [B, 128, 64]
        d2 = torch.cat([d2, e2], dim=1)  # [B, 256, 64]
        d2 = self.dec2(d2)       # [B, 64, 64]

        d1 = self.up1(d2)        # [B, 64, 128]
        d1 = torch.cat([d1, e1], dim=1)  # [B, 128, 128]
        d1 = self.dec1(d1)       # [B, 32, 128]

        output = self.final(d1)  # [B, 1, 128]
        
        return output.squeeze(1), latent  # 現在返回的 latent 是 [B, latent_dim]

def main():
    data_folder = 'labeled_DB'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'torch device: {device}')
    json_files = get_json_files(data_folder, exclude_keywords=[])  # 实现一个函数来获取所有的JSON文件路径

    # 設置參數
    input_dim = 1
    hidden_dim = 128
    latent_dim = 20
    num_layers =2
    batch_size = 32
    lr = 1e-4
    target_len = 128#100
    #'pulse_interpolate_autoencoder.pth' : 一般的pulse autoencoder target_len = 100
    #'pulse_interpolate_autoencoder2.pth' : denoise autoencoder  target_len = 200



    # 加載並劃分數據集
    dataset = PulseDataset(json_files, target_len)
    train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    #計算訓練資料跟測試資料的長度
    print(f'train_data_len:{len(train_data)}, test_data_len:{len(test_data)}')
    # 初始化模型和優化器
    model_path = 'pulse_interpolate_autoencoder_1028_20dim.pth'
    model_path = 'pulse_interpolate_autoencoder_1028_30dim_2criterion.pth'
    model_path = 'pulse_interpolate_autoencoder_1028_30dim_2criterion_30.pth'
    model = EPGBaselinePulseAutoencoder(target_len=target_len).to(device)
    # model_path = 'pulse_interpolate_cnn_autoencoder_1028_30dim.pth'
    # model = ImprovedCNNAutoencoder(latent_dim=latent_dim).to(device)
    # model_path = 'pulse_interpolate_unet_autoencoder_1024_20dim.pth'
    # model_path = 'pulse_interpolate_unet_autoencoder_10dim.pth'
    # model = UNetAutoencoder(target_len=target_len, latent_dim=latent_dim).to(device)
    
    # model = EPGBaselinePulseVAE(target_len=200).to(device)
    # model = LSTMVAE(input_dim, hidden_dim, latent_dim, num_layers).to(device)
    criterion = nn.L1Loss()  #Use L1Loss to train first
    criterion2 = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4) #Adam
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total number of model parameters: {trainable_params}, model:{model}') 
    train_autoencoder(model, train_dataloader, test_dataloader, optimizer, criterion, device, model_path)
    # train_autoencoder(model, train_dataloader, test_dataloader, optimizer, criterion2, device, model_path)

    
    model.load_state_dict(torch.load(model_path))
    # model_path = 'pulse_interpolate_autoencoder.pth'
    # model = EPGBaselinePulseAutoencoder(100).to(device)
    # model.load_state_dict(torch.load(model_path))
    
    # encoded_data = predict_encoded_dataset(model, json_files)
    # save_encoded_data(encoded_data, 'latent_vectors_0909')

    # 分析差异向量
    # pca, n_components_95, all_diff_vectors = analyze_diff_vectors(encoded_data)

    # # 分析主成分
    # analyze_principal_components(pca, n_components_95)

    # # 可视化投影
    # visualize_projections(all_diff_vectors, pca)

    # # 分析重建误差
    # n_components_range = range(1, min(101, len(pca.explained_variance_ratio_) + 1))
    # analyze_reconstruction_error(all_diff_vectors, pca, n_components_range)
    

    # 統計每個維度的分佈範圍
    # all_latent_vectors = []
    # for vectors in encoded_data.values():
    #     all_latent_vectors.extend(vectors)
    # all_latent_vectors = np.array(all_latent_vectors)

    # min_values = np.min(all_latent_vectors, axis=0)
    # max_values = np.max(all_latent_vectors, axis=0)
    # mean_values = np.mean(all_latent_vectors, axis=0)
    # std_values = np.std(all_latent_vectors, axis=0)

    # print(f"Min values: {min_values}")
    # print(f"Max values: {max_values}")
    # print(f"Mean values: {mean_values}")
    # print(f"Std values: {std_values}")

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