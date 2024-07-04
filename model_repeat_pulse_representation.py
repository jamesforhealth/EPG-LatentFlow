


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
from preprocessing import process_DB_rawdata, get_json_files, add_noise_with_snr, add_gaussian_noise_torch
from model_find_peaks import detect_peaks_from_signal
from tqdm import tqdm
import math
import torch.nn.functional as F
from sklearn.model_selection import train_test_split



class PulseDataset(Dataset):  
    def __init__(self, json_files, sample_rate=100, target_length=512):
        self.data = []
        self.lengths = []
        self.sample_rate = sample_rate
        self.target_length = target_length
        self.load_data(json_files)

    def load_data(self, json_files):
        for json_file in json_files:
            try:
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
                    signal = self.normalize(signal)
                    for j in range(len(x_points) - 1):
                        pulse_start = x_points[j]
                        pulse_end = x_points[j + 1]
                        pulse = signal[pulse_start:pulse_end]
                        if len(pulse) > 0: 
                            original_length = len(pulse)
                            pulse = self.repeat_padding(pulse, self.target_length)
                            self.data.append((pulse, original_length))
            except json.JSONDecodeError as e:
                print(f'JSON decode error in {json_file}: {e}')
            except KeyError as e:
                print(f'Key error in {json_file}: {e}')
            except Exception as e:
                print(f'Error in loading {json_file}: {e}')

    def normalize(self, data):
        return (data - np.mean(data)) / (np.std(data) + 1e-8)

    def repeat_padding(self, pulse, target_length):
        repeat_times = target_length // len(pulse) + 1
        repeated_pulse = np.tile(pulse, repeat_times)[:target_length]
        return repeated_pulse

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pulse, length = self.data[idx]
        return torch.tensor(pulse, dtype=torch.float32).unsqueeze(-1), length  # 確保輸出是 [seq_length, 1] 形狀


class PulseFFTDataset(Dataset):  
    def __init__(self, json_files, sample_rate=100, target_length=512):
        self.data = []
        self.lengths = []
        self.sample_rate = sample_rate
        self.target_length = target_length
        self.load_data(json_files)

    def load_data(self, json_files):
        for json_file in json_files:
            try:
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
                    signal = self.normalize(signal)
                    for j in range(len(x_points) - 1):
                        pulse_start = x_points[j]
                        pulse_end = x_points[j + 1]
                        pulse = signal[pulse_start:pulse_end]
                        if len(pulse) > 0: 
                            original_length = len(pulse)
                            pulse = self.repeat_padding(pulse, self.target_length)
                            pulse_fft = np.fft.fft(pulse)  # 進行FFT變換
                            self.data.append((pulse_fft, original_length))
            except json.JSONDecodeError as e:
                print(f'JSON decode error in {json_file}: {e}')
            except KeyError as e:
                print(f'Key error in {json_file}: {e}')
            except Exception as e:
                print(f'Error in loading {json_file}: {e}')

    def normalize(self, data):
        return (data - np.mean(data)) / (np.std(data) + 1e-8)

    def repeat_padding(self, pulse, target_length):
        repeat_times = target_length // len(pulse) + 1
        repeated_pulse = np.tile(pulse, repeat_times)[:target_length]
        return repeated_pulse

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pulse_fft, length = self.data[idx]
        return torch.tensor(pulse_fft, dtype=torch.complex64), length  # 確保輸出是 [seq_length] 形狀


class ConvModule(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size=3, stride=1, padding=1, activation=nn.ReLU(inplace=True)):
        super(ConvModule, self).__init__()
        self.conv = nn.Conv1d(input_dim, output_dim, kernel_size=kernel_size, stride=stride, padding=padding)
        self.activation = activation
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        x = self.pool(x)
        return x

class ConvTransposeModule(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size=3, stride=1, padding=1, activation=nn.ReLU(inplace=True)):
        super(ConvTransposeModule, self).__init__()
        self.conv = nn.ConvTranspose1d(input_dim, output_dim, kernel_size=kernel_size, stride=stride, padding=padding)
        self.activation = activation
        self.upsample = nn.Upsample(scale_factor=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        x = self.upsample(x)
        return x
    

class UnetAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim):
        super(UnetAutoencoder, self).__init__()
        self.encoder_layers = nn.ModuleList()
        self.decoder_layers = nn.ModuleList()

        # 編碼器層
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            self.encoder_layers.append(ConvModule(prev_dim, hidden_dim))
            prev_dim = hidden_dim

        # Latent space transformation
        self.flatten = nn.Flatten()
        encoder_output_size = hidden_dims[-1] * (512 // (2 ** len(hidden_dims)))
        self.fc1 = nn.Linear(encoder_output_size, latent_dim)
        self.fc2 = nn.Linear(latent_dim, encoder_output_size)
        self.unflatten = nn.Unflatten(1, (hidden_dims[-1], 512 // (2 ** len(hidden_dims))))

        # 解碼器層
        self.decoder_layers.append(ConvTransposeModule(hidden_dims[len(hidden_dims) - 1], hidden_dims[len(hidden_dims) - 2]))  # 修改這裡的輸入通道數
        for i in range(len(hidden_dims) - 2, 0, -1):
            self.decoder_layers.append(ConvTransposeModule(hidden_dims[i] * 2, hidden_dims[i - 1]))  # 修改這裡的輸入通道數
        self.decoder_layers.append(ConvTransposeModule(hidden_dims[0] * 2, input_dim, activation=nn.Sigmoid()))

    def forward(self, x):
        x = x.permute(0, 2, 1)  # [batch_size, input_dim, seq_length]
        
        # 編碼器前向傳遞
        encoder_outputs = []
        for layer in self.encoder_layers:
            x = layer(x)
            encoder_outputs.append(x)
            # print(f'encoder_outputs x.shape: {x.shape}')
        # Latent space transformation
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.unflatten(x)
        # print(f'unflatten x.shape: {x.shape}')
        # 解碼器前向傳遞
        for i, layer in enumerate(self.decoder_layers):
            if i > 0:
                # print(f'encoder_outputs[{-i-1}].shape: {encoder_outputs[-i-1].shape}')
                x = torch.cat([x, encoder_outputs[-i-1]], dim=1)
                # print(f'cat x.shape: {x.shape}')
            # print(f'layer: {layer}')
            x = layer(x)
            # print(f'x.shape: {x.shape}')

        x = x.permute(0, 2, 1)  # [batch_size, seq_length, input_dim]
        return x

def calculate_loss(output, target, lengths):
    criterion = nn.MSELoss()
    pulse_losses = []
    repeat_losses = []
    for i in range(len(lengths)):
        length = lengths[i]
        pulse_losses.append(criterion(output[i, :length], target[i, :length]))

        # 計算重複部分的損失
        repeat_output = output[i, length:]
        repeat_length = repeat_output.size(0)
        repeat_times = (repeat_length // length) + 1
        repeated_output = torch.cat([output[i, :length]] * repeat_times)[:repeat_length]
        repeat_loss = criterion(repeat_output, repeated_output)
        repeat_losses.append(repeat_loss)

    pulse_loss = torch.mean(torch.stack(pulse_losses))
    repeat_loss = torch.mean(torch.stack(repeat_losses))
    total_loss = pulse_loss + repeat_loss  # 你可以調整這裡的加權方式
    return total_loss

def train(model, dataloader, lr, device, epochs=1000):
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    if os.path.exists('repeat_unet_autoencoder.pt'):
        model.load_state_dict(torch.load('repeat_unet_autoencoder.pt'))
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in dataloader:
            pulses, lengths = batch
            pulses = pulses.to(device).squeeze(2)
            # print(f'pulses.shape: {pulses.shape}')
            optimizer.zero_grad()
            output = model(pulses)
            loss = calculate_loss(output, pulses, lengths)
            #loss = criterion(output, pulses)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(dataloader)}')

    #save
    torch.save(model.state_dict(), 'repeat_unet_autoencoder.pt')

def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            pulses, lengths = batch
            pulses = pulses.to(device)
            output = model(pulses)
            loss = calculate_loss(output, pulses, lengths)
            total_loss += loss.item()
    print(f'Evaluation Loss: {total_loss / len(dataloader)}')

class FullyConnectedAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(FullyConnectedAutoencoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(True),
            nn.Linear(256, 128),
            nn.ReLU(True),
            nn.Linear(128, latent_dim)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 256),
            nn.ReLU(True),
            nn.Linear(256, input_dim),
            nn.Sigmoid()  # 用 Sigmoid 使輸出在0到1之間
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class UnetFullyConnectedAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(UnetFullyConnectedAutoencoder, self).__init__()
        
        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.ReLU(True)
            ),
            nn.Sequential(
                nn.Linear(256, 128),
                nn.ReLU(True)
            ),
            nn.Sequential(
                nn.Linear(128, latent_dim),
                nn.ReLU(True)
            )
        ])
        
        # Decoder layers
        self.decoder_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(latent_dim, 128),
                nn.ReLU(True)
            ),
            nn.Sequential(
                nn.Linear(256, 256),  # 這裡是將上一層的輸出與對應的Encoder層的輸出concat後的大小
                nn.ReLU(True)
            ),
            nn.Sequential(
                nn.Linear(512, input_dim),
                nn.Sigmoid()  # 用 Sigmoid 使輸出在0到1之間
            )
        ])

    def forward(self, x):
        encoder_outputs = []
        # print(f'input x.shape: {x.shape}')
        # Encoder forward pass
        for layer in self.encoder_layers:
            x = layer(x)
            encoder_outputs.append(x)
        # Decoder forward pass
        for i, layer in enumerate(self.decoder_layers):
            if i > 0:
                x = torch.cat([x, encoder_outputs[-i-1]], dim=1)
            x = layer(x)
        return x
def main():
    # 設置參數
    hidden_dims = [4, 8, 16, 32, 64, 128, 256]
    input_dim = 1
    latent_dim = 64    
    batch_size = 32
    lr = 1e-4

    data_folder = 'labeled_DB'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'torch device: {device}')
    json_files = get_json_files(data_folder)  # 实现一个函数来获取所有的JSON文件路径
    dataset = PulseDataset(json_files)
    train_data, test_data = train_test_split(dataset, test_size=0.1, random_state=42)
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)


    model = UnetAutoencoder(input_dim, hidden_dims, latent_dim)
    model = FullyConnectedAutoencoder(512, latent_dim)
    model = UnetFullyConnectedAutoencoder(512, latent_dim)
    model.train()
    # model = PulseTransformer(input_dim, latent_dim, num_layers, seq_length)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total number of model parameters: {trainable_params}, model:{model}') 
    train(model, train_dataloader, lr, device)
    
    
    model.eval()
    evaluate(model, test_dataloader, device)

# 測試加載並顯示數據
if __name__ == '__main__':
    main()
