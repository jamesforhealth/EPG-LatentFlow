'''
Baseline model for the wearing anomaly detection task. (Predict per second)


'''


import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.models import densenet121
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
from preprocessing import process_DB_rawdata, get_json_files, add_noise_with_snr
from model_find_peaks import detect_peaks_from_signal
from tqdm import tqdm
import math
class Application(tk.Frame):
    def __init__(self, model, device, train_files, test_files , master=None):
        super().__init__(master)
        self.master = master
        self.model = model
        self.device = device
        self.pack()
        self.create_widgets()
        self.train_files = train_files
        self.test_files = test_files

    def create_widgets(self):
        self.load_btn = tk.Button(self)
        self.load_btn["text"] = "Load JSON File"
        self.load_btn["command"] = self.load_file
        self.load_btn.pack(side="top")

        self.quit_btn = tk.Button(self, text="QUIT", fg="red",
                                  command=self.master.destroy)
        self.quit_btn.pack(side="bottom")

    def load_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])
        if file_path:
            try:
                # losses = predict_per_second(self.model, file_path, 100, self.device)
                losses = predict_per_twoseconds(self.model, file_path, 100, self.device)
                file_name = os.path.basename(file_path)
                print(f'load file_name:{file_name}, losses:{losses}')
                # if file_name in self.train_files:
                #     self.plot_losses(losses, 'Train', file_name)
                # elif file_name in self.test_files:
                #     self.plot_losses(losses, 'Test', file_name)
                # else:
                #     self.plot_losses(losses, 'Real world', file_name)
                self.plot_losses(losses, 'Real world', file_name)
            except Exception as e:
                messagebox.showerror("Error", str(e))

    def plot_losses(self, losses, type, file_name):
        plt.figure(figsize=(10, 5))
        plt.plot(losses, marker='o', linestyle='-')
        plt.title(f'Loss for {file_name}({type})')
        plt.xlabel('Seconds')
        #計算平均，最大最小值，標準差
        mean = np.mean(losses)
        std = np.std(losses)
        min_loss = np.min(losses)
        max_loss = np.max(losses)        
        plt.ylabel(f'Loss: {mean:.7f}+-{std:.7f}({min_loss:.7f},{max_loss:.7f})')
        plt.grid(True)
        plt.show()

class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride, padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.skip_conv = nn.Conv1d(in_channels, out_channels, 1, stride, 0) if in_channels != out_channels else None

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.skip_conv is not None:
            identity = self.skip_conv(identity)

        out += identity
        out = self.relu(out)
        return out

class ResNetAutoencoder(nn.Module):
    def __init__(self, input_channels, output_channels, latent_dim=64):
        super(ResNetAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            ResNetBlock(input_channels, 16),
            nn.MaxPool1d(2),
            ResNetBlock(16, 32),
            nn.MaxPool1d(2),
            nn.Flatten(),
            nn.Linear(32 * 50, latent_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32 * 50),
            nn.Unflatten(1, (32, 50)), 
            nn.ConvTranspose1d(32, 16, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose1d(16, output_channels, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv1d(output_channels, output_channels, kernel_size=1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x.squeeze(1)

class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer = self._make_layer(in_channels + i * growth_rate, growth_rate)
            self.layers.append(layer)

    def _make_layer(self, in_channels, growth_rate):
        layer = nn.Sequential(
            nn.Conv1d(in_channels, growth_rate, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(growth_rate),
            nn.ReLU(inplace=True)
        )
        return layer

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            out = layer(torch.cat(features, 1))
            features.append(out)
        return torch.cat(features, 1)

class TransitionLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TransitionLayer, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.AvgPool1d(2)
        )

    def forward(self, x):
        return self.layer(x)

class DenseNetAutoencoder(nn.Module):
    def __init__(self, input_channels, output_channels, growth_rate=12, num_init_features=32, latent_dim=64):
        super(DenseNetAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(input_channels, num_init_features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(num_init_features),
            nn.ReLU(inplace=True),
            DenseBlock(num_init_features, growth_rate, num_layers=4),
            TransitionLayer(num_init_features + 4 * growth_rate, num_init_features + 4 * growth_rate),
            DenseBlock(num_init_features + 4 * growth_rate, growth_rate, num_layers=4),
            TransitionLayer(num_init_features + 8 * growth_rate, num_init_features + 8 * growth_rate),
            nn.AdaptiveAvgPool1d(1),  # 自适应平均池化层
            nn.Flatten(),
            nn.Linear(num_init_features + 8 * growth_rate, latent_dim)  # 调整全连接层的输入维度
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, (num_init_features + 8 * growth_rate) * 8),  # 调整全连接层的输出维度
            nn.Unflatten(1, (num_init_features + 8 * growth_rate, 8)),  # 调整Unflatten层的输出形状
            nn.ConvTranspose1d(num_init_features + 8 * growth_rate, num_init_features + 4 * growth_rate, kernel_size=5, stride=5),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(num_init_features + 4 * growth_rate, num_init_features, kernel_size=5, stride=5),
            nn.ReLU(inplace=True),
            nn.Conv1d(num_init_features, output_channels, kernel_size=1)  # 最后一层确保输出形状匹配
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        # for layer in self.decoder:
        #     x = layer(x)
        #     input(f'x.shape: {x.shape}, layer: {layer}')
        return x

class StackedAutoencoder(nn.Module):
    def __init__(self, input_channels, output_channels, latent_dim, pretrained_weights):
        super(StackedAutoencoder, self).__init__()
        self.outer_autoencoder = UNetAutoencoder(input_channels, output_channels)
        self.outer_autoencoder.load_state_dict(torch.load(pretrained_weights))
        
        # 固定外層自編碼器的權重
        for param in self.outer_autoencoder.parameters():
            param.requires_grad = False
        
        self.inner_autoencoder = InnerAutoencoder(128, latent_dim)

    def forward(self, x):
        x = self.outer_autoencoder.encoder(x)
        x = self.inner_autoencoder(x)
        x = self.outer_autoencoder.decoder(x)
        return x.squeeze(1)

class UNetAutoencoder2(nn.Module):
    def __init__(self, input_channels, output_channels, latent_dim=64):
        super(UNetAutoencoder2, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Flatten(),
            nn.Linear(128 * 50, latent_dim)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128 * 50),
            nn.Unflatten(1, (128, 50)), 
            nn.ConvTranspose1d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose1d(64, output_channels, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv1d(output_channels, output_channels, kernel_size=1)
        )
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x.squeeze(1)

class UNetAutoencoder(nn.Module): # Total number of model parameters: 64408
    def __init__(self, input_channels, output_channels):
        super(UNetAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose1d(64, output_channels, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv1d(output_channels, output_channels, kernel_size=1)
        )
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x.squeeze(1)

# class DeepUNetAutoencoder(nn.Module): 
#     def __init__(self, input_channels, output_channels, latent_dim=24):
#         super(DeepUNetAutoencoder, self).__init__()
#         self.encoder = nn.Sequential(
#             nn.Conv1d(input_channels, 16, kernel_size=3, padding=1),  # (batch_size, 1, 200) -> (batch_size, 16, 200)
#             nn.LeakyReLU(),
#             nn.BatchNorm1d(16),
#             nn.MaxPool1d(2),  # (batch_size, 16, 200) -> (batch_size, 16, 100)
#             nn.Conv1d(16, 8, kernel_size=3, padding=1),  # (batch_size, 16, 100) -> (batch_size, 8, 100)
#             nn.LeakyReLU(),
#             nn.BatchNorm1d(8),
#             nn.MaxPool1d(2),  # (batch_size, 8, 100) -> (batch_size, 8, 50)
#             nn.Conv1d(8, 4, kernel_size=3, padding=1),  # (batch_size, 8, 50) -> (batch_size, 4, 50)
#             nn.LeakyReLU(),
#             nn.BatchNorm1d(4),
#             nn.MaxPool1d(2),  # (batch_size, 4, 50) -> (batch_size, 4, 25)
#             nn.Conv1d(4, 2, kernel_size=3, padding=1),  # (batch_size, 4, 25) -> (batch_size, 2, 25)
#             nn.LeakyReLU(),
#             nn.BatchNorm1d(2),
#             nn.Flatten(),  # (batch_size, 2, 25) -> (batch_size, 2*25)
#             nn.Linear(2 * 25, latent_dim),  # (batch_size, 2*25) -> (batch_size, latent_dim)
#             nn.GELU()
#         )
#         # Fully connected layer to decode from latent space
#         self.decoder = nn.Sequential(
#             nn.Linear(latent_dim, 2 * 25),  # (batch_size, latent_dim) -> (batch_size, 2*25)
#             nn.GELU(),
#             nn.Unflatten(1, (2, 25)),  # (batch_size, 2*25) -> (batch_size, 2, 25)
#             nn.ConvTranspose1d(2, 4, kernel_size=2, stride=2),  # (batch_size, 2, 25) -> (batch_size, 4, 50)
#             nn.LeakyReLU(),
#             nn.BatchNorm1d(4),
#             nn.ConvTranspose1d(4, 8, kernel_size=2, stride=2),  # (batch_size, 4, 50) -> (batch_size, 8, 100)
#             nn.LeakyReLU(),
#             nn.BatchNorm1d(8),
#             nn.ConvTranspose1d(8, 16, kernel_size=2, stride=2),  # (batch_size, 8, 100) -> (batch_size, 16, 200)
#             nn.LeakyReLU(),
#             nn.BatchNorm1d(16),
#             nn.Conv1d(16, output_channels, kernel_size=1),  # (batch_size, 16, 200) -> (batch_size, output_channels, 200)
#             nn.LeakyReLU()
#         )

#     def forward(self, x):
#         x = self.encoder(x)
#         x = self.decoder(x)
#         return x

class DeepUNetAutoencoder(nn.Module): 
    def __init__(self, input_channels, output_channels, latent_dim=24):
        super(DeepUNetAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=3, padding=1),  # (batch_size, 1, 200) -> (batch_size, 16, 200)
            nn.LeakyReLU(),
            nn.BatchNorm1d(32),
            nn.MaxPool1d(2),  # (batch_size, 16, 200) -> (batch_size, 16, 100)
            nn.Conv1d(32, 16, kernel_size=3, padding=1),  # (batch_size, 16, 100) -> (batch_size, 8, 100)
            nn.LeakyReLU(),
            nn.BatchNorm1d(16),
            nn.MaxPool1d(2),  # (batch_size, 8, 100) -> (batch_size, 8, 50)
            nn.Conv1d(16, 8, kernel_size=3, padding=1),  # (batch_size, 8, 50) -> (batch_size, 4, 50)
            nn.LeakyReLU(),
            nn.BatchNorm1d(8),
            nn.MaxPool1d(2),  # (batch_size, 4, 50) -> (batch_size, 4, 25)
            nn.Conv1d(8, 4, kernel_size=3, padding=1),  # (batch_size, 4, 25) -> (batch_size, 2, 25)
            nn.LeakyReLU(),
            nn.BatchNorm1d(4),
            nn.Flatten(),  # (batch_size, 2, 25) -> (batch_size, 2*25)
            nn.Linear(4 * 25, latent_dim),  # (batch_size, 2*25) -> (batch_size, latent_dim)
            nn.GELU()
        )
        # Fully connected layer to decode from latent space
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 4 * 25),  # (batch_size, latent_dim) -> (batch_size, 2*25)
            nn.GELU(),
            nn.Unflatten(1, (4, 25)),  # (batch_size, 2*25) -> (batch_size, 2, 25)
            nn.ConvTranspose1d(4, 8, kernel_size=2, stride=2),  # (batch_size, 2, 25) -> (batch_size, 4, 50)
            nn.LeakyReLU(),
            nn.BatchNorm1d(8),
            nn.ConvTranspose1d(8, 16, kernel_size=2, stride=2),  # (batch_size, 4, 50) -> (batch_size, 8, 100)
            nn.LeakyReLU(),
            nn.BatchNorm1d(16),
            nn.ConvTranspose1d(16, 32, kernel_size=2, stride=2),  # (batch_size, 8, 100) -> (batch_size, 16, 200)
            nn.LeakyReLU(),
            nn.BatchNorm1d(32),
            nn.Conv1d(32, output_channels, kernel_size=1),  # (batch_size, 16, 200) -> (batch_size, output_channels, 200)
            nn.LeakyReLU()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class WearingDataset(Dataset):
    def __init__(self, json_files, window_size, sample_rate=100, overlap_ratio=0.99):
        self.data = []
        self.sample_rate = sample_rate
        self.load_data(json_files, window_size, overlap_ratio)

    def load_data(self, json_files, window_size, overlap_ratio):#sliding window based
        for json_file in json_files:
            with open(json_file, 'r') as f:
                json_data = json.load(f)
                if json_data['anomaly_list'] != []: 
                    continue
                signal = json_data['smoothed_data']
                original_sample_rate = json_data.get('sample_rate', 100)

                if original_sample_rate != self.sample_rate:
                    num_samples = int(len(signal) * self.sample_rate / original_sample_rate)
                    signal = scipy.signal.resample(signal, num_samples)

                signal = np.array(signal, dtype=np.float32)
                stride = int(window_size * (1 - overlap_ratio))
                for i in range(0, len(signal) - window_size + 1, stride):
                    segment = signal[i:i + window_size]
                    segment = self.normalize(segment)
                    self.data.append(segment)
        # 對數據進行標準化處理
        # self.data = self.normalize(self.data)

class WearingDataset2(Dataset):  # Train dataset len: 1082369
    def __init__(self, json_files, window_size, sample_rate=100, stride=1):
        self.data = []
        self.sample_rate = sample_rate
        self.load_data(json_files, window_size, stride)

    def load_data(self, json_files, window_size, stride):
        for json_file in json_files:
            with open(json_file, 'r') as f:
                json_data = json.load(f)
                if json_data['anomaly_list']:
                    continue
                signal = json_data['smoothed_data']
                original_sample_rate = json_data.get('sample_rate', 100)
                x_points = json_data['x_points']

                if original_sample_rate != self.sample_rate:
                    num_samples = int(len(signal) * self.sample_rate / original_sample_rate)
                    signal = scipy.signal.resample(signal, num_samples)
                    x_points = [int(x * self.sample_rate / original_sample_rate) for x in x_points]

                # signal = np.array(signal, dtype=np.float32)
                # for i in range(0, len(signal) - window_size + 1, stride):
                #     segment = signal[i:i + window_size]
                #     segment = self.normalize(segment)
                #     self.data.append(segment)

                # 增加pulse填充到window大小的資料
                for j in range(len(x_points) - 1):
                    pulse_start = x_points[j]
                    pulse_end = x_points[j + 1]
                    pulse_duration = pulse_end - pulse_start
                    padded_pulse = np.zeros(window_size, dtype=np.float32)
                    pulse = signal[pulse_start:pulse_end]
                    padded_pulse[:pulse_duration] = self.normalize(pulse)
                    self.data.append(padded_pulse)
                    # for k in range(0, window_size - pulse_duration + 1, stride):
                    #     padded_pulse = np.zeros(window_size, dtype=np.float32)
                    #     pulse = signal[pulse_start:pulse_end]
                    #     padded_pulse[k:k + pulse_duration] = self.normalize(pulse)
                    #     # print(f'padded_pulse.shape: {padded_pulse.shape}')
                    #     self.data.append(padded_pulse)

    def normalize(self, data):
        return (data - np.mean(data)) / (np.std(data) + 1e-8)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx]).unsqueeze(0)  # 添加通道维度

        # return torch.tensor(self.data[idx]).unsqueeze(0)  # Add channel dimension 

    # def load_data(self, json_files):#pulse based
    #     all_signals = []
    #     for json_file in json_files:
    #         with open(json_file, 'r') as f:
    #             json_data = json.load(f)
    #             if json_data['anomaly_list'] != []: 
    #                 continue
    #             signal = json_data['smoothed_data']
    #             original_sample_rate = json_data.get('sample_rate', 100)
    #             x_points = json_data['x_points']

    #             if original_sample_rate != self.sample_rate:
    #                 num_samples = int(len(signal) * self.sample_rate / original_sample_rate)
    #                 signal = scipy.signal.resample(signal, num_samples)
    #                 x_points = [int(x * self.sample_rate / original_sample_rate) for x in x_points]

    #             signal = np.array(signal, dtype=np.float32)
    #             all_signals.extend(signal)
                
    #             for i in range(len(x_points) - 1):
    #                 segment = signal[x_points[i]:x_points[i+1]]
    #                 segment = self.normalize(segment)
    #                 if len(segment) > 0:
    #                     # if len(segment) < 200:
    #                     #     segment = np.pad(segment, (0, 200 - len(segment)), 'constant')
    #                     self.data.append(segment)
        
    #     self.data = np.array(self.data, dtype=object)


def load_data(data_folder, window_size, sample_rate):
    json_files = get_json_files(data_folder)    
    if not json_files:
        raise ValueError(f"No JSON files found in {data_folder}")
    
    random.shuffle(json_files)
    split_ratio = 1
    split_idx = int(len(json_files) * split_ratio)
    train_files = json_files[:split_idx]
    test_files = json_files[split_idx:]

    train_dataset = WearingDataset(train_files, window_size, sample_rate)
    test_dataset = WearingDataset(test_files, window_size, sample_rate, 0.0)
    print(f'Train dataset len: {len(train_dataset)}')
    return train_dataset, test_dataset, [os.path.basename(f) for f in train_files], [os.path.basename(f) for f in test_files]

def load_data2(data_folder, window_size, sample_rate):
    json_files = get_json_files(data_folder)    
    if not json_files:
        raise ValueError(f"No JSON files found in {data_folder}")
    
    random.shuffle(json_files)
    split_ratio = 1
    split_idx = int(len(json_files) * split_ratio)
    train_files = json_files[:split_idx]
    test_files = json_files[split_idx:]

    train_dataset = WearingDataset2(train_files, window_size, sample_rate)
    test_dataset = WearingDataset2(test_files, window_size, sample_rate, 0.0)
    print(f'Train dataset len: {len(train_dataset)}')
    return train_dataset, test_dataset, [os.path.basename(f) for f in train_files], [os.path.basename(f) for f in test_files]




def train_autoencoder(model, dataloader, optimizer, criterion, device, epochs=100):
    model.train()
    for epoch in tqdm(range(epochs)):
        total_loss = 0
        for data in dataloader:
            data = data.to(device)
            noisy_data = add_noise_with_snr(data, 20)

            optimizer.zero_grad()
            outputs = model(noisy_data)
            averaged_outputs = torch.mean(outputs, dim=1, keepdim=True)
            loss = criterion(averaged_outputs, data)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch+1}, Loss: {total_loss / len(dataloader)}')

def test_autoencoder(model, dataloader, criterion, device):

    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data in dataloader:
            data = data.to(device).unsqueeze(1)
            outputs = model(data)
            loss = criterion(outputs, data)
            total_loss += loss.item()
        print('Average Test Loss:', total_loss / len(dataloader))


def predict_per_second(model, json_file, sample_rate, device):
    with open(json_file, 'r') as f:
        json_data = json.load(f)
        signal = process_DB_rawdata(json_data) if 'smoothed_data' not in json_data else np.array(json_data['smoothed_data'], dtype=np.float32)
        if json_data.get('sample_rate', 100) != sample_rate:
            num_samples = int(len(signal) * sample_rate / json_data.get('sample_rate', 100))
            signal = scipy.signal.resample(signal, num_samples)

    # input(f'signal shape: {signal.shape}')
    model.eval()
    criterion = nn.MSELoss()
    losses = []
    for i in range(0, len(signal) - sample_rate + 1, sample_rate):
        segment = signal[i:i+sample_rate]
        segment = (segment - np.mean(segment)) / np.std(segment)  # 對每個片段進行正規化
        segment_tensor = torch.tensor(segment).view(1, 1, -1).to(device)
        with torch.no_grad():
            output = model(segment_tensor)
            loss = criterion(output, segment_tensor)
            losses.append(loss.item())
    
    return losses

def predict_per_twoseconds(model, json_file, sample_rate, device):
    with open(json_file, 'r') as f:
        json_data = json.load(f)
        signal = process_DB_rawdata(json_data) if 'smoothed_data' not in json_data else np.array(json_data['smoothed_data'], dtype=np.float32)
        if json_data.get('sample_rate', 100) != sample_rate:
            num_samples = int(len(signal) * sample_rate / json_data.get('sample_rate', 100))
            signal = scipy.signal.resample(signal, num_samples)
    
    model.eval()
    criterion = nn.MSELoss()
    losses = []

    for i in range(0, len(signal) - 2*sample_rate + 1, sample_rate):
        segment = signal[i:i+2*sample_rate]
        
        segment_mean = np.mean(segment)
        segment_std = np.std(segment)
        normalized_segment = (segment - segment_mean) / segment_std
        
        segment_tensor = torch.tensor(normalized_segment).view(1, 1, -1).to(device)
        with torch.no_grad():
            output = model(segment_tensor)
            loss = criterion(output, segment_tensor)
            losses.append(loss.item())
    
    return losses

def predict_reconstructed_signal(signal, sample_rate, model_name = 'unet_autoencoder1.pt'):
    try:
        window_size = 100  # Assuming each second contains 100 sampling points
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'torch device: {device}')
        model = UNetAutoencoder(1, window_size).to(device)
        model.load_state_dict(torch.load(model_name))   
        resample_ratio = 1.0
        if sample_rate != window_size:
            resample_ratio = window_size / sample_rate
            signal = scipy.signal.resample(signal, int(len(signal)/resample_ratio))
        
        model.eval()
        reconstructed_signal = np.array([])  # 使用空的一維 NumPy 數組初始化
        for i in range(0, len(signal) - window_size + 1, window_size):
            segment = signal[i:i+window_size]
            segment_mean = np.mean(segment)
            segment_std = np.std(segment)
            
            normalized_segment = (segment - segment_mean) / segment_std
            segment_tensor = torch.tensor(normalized_segment).float().view(1, 1, -1).to(device)
            
            with torch.no_grad():
                output = model(segment_tensor)
                reconstructed_segment = output.squeeze().cpu().numpy()[0]
                
                # 使用原始片段的平均值和縮放比例來還原重構的片段
                reconstructed_segment = reconstructed_segment * segment_std + segment_mean
                # reconstructed_signal = np.concatenate((reconstructed_signal, reconstructed_segment.flatten()))  # 將重構的片段拼接到 reconstructed_signal 中
                reconstructed_signal = np.append(reconstructed_signal, reconstructed_segment)
        
        if sample_rate != window_size:
            reconstructed_signal = scipy.signal.resample(reconstructed_signal, int(len(signal)*resample_ratio))
        return reconstructed_signal

    except Exception as e:
        print(f'Error predicting reconstructed signal: {e}')
        return []

def predict_reconstructed_signal2(signal, sample_rate, model_name='unet_autoencoder3.pt'):
    try:
        window_size = 200  # Assuming the window size is 2 seconds (200 sampling points)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'torch device: {device}')
        model = ResNetAutoencoder(1, window_size).to(device)
        model.load_state_dict(torch.load(model_name))
        
        model.eval()
        resample_ratio = 1.0
        if sample_rate != window_size:
            resample_ratio = window_size / sample_rate
            signal = scipy.signal.resample(signal, int(len(signal)/resample_ratio))        
        reconstructed_signal = signal.copy()
        x_points = detect_peaks_from_signal(signal, sample_rate)
        
        for i in range(0, len(signal) - window_size + 1, window_size // 2):
            segment = signal[i:i+window_size]
            
            segment_mean = np.mean(segment)
            segment_std = np.std(segment)
            normalized_segment = (segment - segment_mean) / segment_std
            
            segment_tensor = torch.tensor(normalized_segment).float().view(1, 1, -1).to(device)
            
            with torch.no_grad():
                output = model(segment_tensor)
                reconstructed_segment = output.squeeze().cpu().numpy()
                
                # 檢查reconstructed_segment的形狀
                if reconstructed_segment.ndim == 2:
                    reconstructed_segment = reconstructed_segment[0]  # 如果有額外的維度,移除它
                
                reconstructed_segment = reconstructed_segment * segment_std + segment_mean
                
                # 找到當前窗口內的peak點索引
                window_peaks = [p for p in x_points if i <= p < i+window_size]
                
                # 將reconstructed_segment的peak到peak區段寫回reconstructed_signal
                for j in range(len(window_peaks) - 1):
                    start = window_peaks[j] - i
                    end = window_peaks[j+1] - i
                    reconstructed_signal[i+start:i+end] = reconstructed_segment[start:end]
        
        if sample_rate != window_size:
            reconstructed_signal = scipy.signal.resample(reconstructed_signal, int(len(signal)*resample_ratio))
        return reconstructed_signal

    except Exception as e:
        print(f'Error predicting reconstructed signal: {e}')
        return []

def predict_reconstructed_signal_pulse(signal, sample_rate, peaks, model_name='unet_autoencoder3.pt'):
    try:
        window_size = 200  # Assuming the window size is 2 seconds (200 sampling points)
        step = 100
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'torch device: {device}')
        model = ResNetAutoencoder(1, window_size).to(device)
        model.load_state_dict(torch.load(model_name))
        
        model.eval()
        resample_ratio = 1.0
        if sample_rate != window_size:
            resample_ratio = window_size / sample_rate
            signal = scipy.signal.resample(signal, int(len(signal)/resample_ratio))     
        # reconstructed_signal copy signal        
        reconstructed_signal = signal.copy()
        print(f'peaks:{peaks}')
        
        for i in range(0, len(signal) - window_size + 1, step):
            segment = np.zeros(window_size, dtype=np.float32)
            # 找到當前窗口內的peak點索引
            window_peaks = [p for p in peaks if i <= p < i+window_size]
            print(f'i:{i}, window_peaks:{window_peaks}')
            # 將segment中peak到peak的區段設為原始訊號的值,其他部分保持為0
            for j in range(len(window_peaks) - 1):
                start = window_peaks[j] - i
                end = window_peaks[j+1] - i
                segment[start:end] = signal[i+start:i+end]
            
            segment_mean = np.mean(signal[i:i+window_size])
            segment_std = np.std(signal[i:i+window_size])
            normalized_segment = (segment - segment_mean) / segment_std
            
            segment_tensor = torch.tensor(normalized_segment).float().view(1, 1, -1).to(device)
            
            with torch.no_grad():
                output = model(segment_tensor)
                reconstructed_segment = output.squeeze().cpu().numpy()
                
                # 檢查reconstructed_segment的形狀
                if reconstructed_segment.ndim == 2:
                    reconstructed_segment = reconstructed_segment[0]  # 如果有額外的維度,移除它
                
                reconstructed_segment = reconstructed_segment * segment_std + segment_mean
                
                # 將reconstructed_segment的完整pulse段落寫回reconstructed_signal
                for j in range(len(window_peaks) - 1):
                    start = window_peaks[j] - i
                    end = window_peaks[j+1] - i
                    reconstructed_signal[i+start:i+end] = reconstructed_segment[start:end]
                    print(f'start:{start}, end:{end}')
        if sample_rate != window_size:
            reconstructed_signal = scipy.signal.resample(reconstructed_signal, int(len(signal)*resample_ratio))
        return reconstructed_signal

    except Exception as e:
        print(f'Error predicting reconstructed signal: {e}')
        return []

def main():
    root = tk.Tk()
    root.title("Anomaly Detection Viewer")
    data_folder = 'DB' #'labeled_DB'
    training_folder = 'labeled_DB'
    window_size = 200  # Assuming each second contains 100 sampling points
    sample_rate = 100
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'torch device: {device}')
    # Initialize and train the U-Net autoencoder
    # model = ResNetAutoencoder(1, window_size).to(device)
    model = DenseNetAutoencoder(1, window_size).to(device)
    # model = UNetAutoencoder2(1, window_size).to(device)
    # model = DeepUNetAutoencoder(1, pulse_size).to(device) 
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total number of model parameters: {trainable_params}, model:{model}')    
    # if os.path.exists('resnet_autoencoder.pt'): 
    if os.path.exists('densenet_autoencoder.pt'): 
        #unet_autoencoder1.pt: (UNetAutoencoder with data standardization per 1 second) (Default)
        #unet_autoencoder2.pt: (UNetAutoencoder2 with data standardization per 2 second)  

        #unet_autoencoder3.pt: (ResNetAutoencoder with data standardization per 2 second)  
        #unet_autoencoder4.pt: DeepUNetAutoencoder with data standardization  
        #unet_autoencoder5.pt: DeepUNetAutoencoder with padded-pulsewise input and data standardization
        model.load_state_dict(torch.load('densenet_autoencoder.pt'))
    # else:
    # Load data and create data loaders
    train_dataset, test_dataset, train_files, test_files = load_data2(training_folder, window_size, sample_rate)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=8)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train_autoencoder(model, train_loader, optimizer, criterion, device)
    # # # #save model
    torch.save(model.state_dict(), 'densenet_autoencoder.pt')

    results = {}
    for subject_folder in os.listdir(data_folder):
        subject_path = os.path.join(data_folder, subject_folder)
        if os.path.isdir(subject_path):
            results[subject_folder] = {}
            for json_file in os.listdir(subject_path):
                if json_file.endswith('.json'):
                    json_path = os.path.join(subject_path, json_file)
                    is_trainingset = ''
                    # 檢查訓練資料夾中是否存在相同相對路徑的檔案
                    training_json_path = os.path.join(training_folder, subject_folder, json_file)
                    if os.path.exists(training_json_path):
                        print(f"Skipping prediction for {subject_folder}/{json_file} (already trained)")
                        is_trainingset = '[Training set]'
                    
                    try:
                        # losses = predict_per_second(model, json_path, sample_rate, device)
                        losses = predict_per_twoseconds(model, json_path, sample_rate, device)
                        mean_loss = np.mean(losses)
                        std_loss = np.std(losses)
                        results[subject_folder][is_trainingset+json_file] = f"{mean_loss:.7f}+-{std_loss:.7f}"
                        print(f"{subject_folder}/{json_file}: {mean_loss:.7f}+-{std_loss:.7f}")
                        # print(f'losses: {losses}')
                    except Exception as e:
                        print(f"Exception {subject_folder}/{json_file}: {e}")

    # Save results to a JSON file
    with open('prediction_results_densenet.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)

    # app = Application(model, device, [], [],  master=root)
    # app.mainloop()

if __name__ == "__main__":
    main()