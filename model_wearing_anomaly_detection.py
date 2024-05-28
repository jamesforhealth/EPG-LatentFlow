import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
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
from preprocessing import process_DB_rawdata
from model_find_peaks import detect_peaks_from_signal
import math
import coremltools as ct
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
                losses = predict_per_second(self.model, file_path, 100, self.device)
                # losses = predict_per_pulse(self.model, file_path, 100, self.device)
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

# class UNetAutoencoder(nn.Module):
#     def __init__(self, input_channels, output_channels):
#         super(UNetAutoencoder, self).__init__()
#         self.encoder = nn.Sequential(
#             nn.Conv1d(input_channels, 64, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool1d(2),
#             nn.Conv1d(64, 128, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool1d(2)
#         )
        
#         self.decoder = nn.Sequential(
#             nn.ConvTranspose1d(128, 64, kernel_size=2, stride=2),
#             nn.ReLU(),
#             nn.ConvTranspose1d(64, output_channels, kernel_size=2, stride=2),
#             nn.ReLU(),
#             nn.Conv1d(output_channels, output_channels, kernel_size=1)
#         )

#     def forward(self, x):
#         x = self.encoder(x)
#         x = self.decoder(x)
#         return x.squeeze(1)  # 確保輸出形狀與輸入匹配

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
        self.load_data(json_files)#, window_size, overlap_ratio)

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
                    
                    self.data.append(segment)
        # 對數據進行標準化處理
        self.data = self.normalize(self.data)


    def load_data(self, json_files):#pulse based
        all_signals = []
        for json_file in json_files:
            with open(json_file, 'r') as f:
                json_data = json.load(f)
                if json_data['anomaly_list'] != []: 
                    continue
                signal = json_data['smoothed_data']
                original_sample_rate = json_data.get('sample_rate', 100)
                x_points = json_data['x_points']

                if original_sample_rate != self.sample_rate:
                    num_samples = int(len(signal) * self.sample_rate / original_sample_rate)
                    signal = scipy.signal.resample(signal, num_samples)
                    x_points = [int(x * self.sample_rate / original_sample_rate) for x in x_points]

                signal = np.array(signal, dtype=np.float32)
                all_signals.extend(signal)
                
                for i in range(len(x_points) - 1):
                    segment = signal[x_points[i]:x_points[i+1]]
                    segment = self.normalize(segment)
                    if len(segment) > 0:
                        # if len(segment) < 200:
                        #     segment = np.pad(segment, (0, 200 - len(segment)), 'constant')
                        self.data.append(segment)
        
        self.data = np.array(self.data, dtype=object)

    def normalize(self, data):
        data = np.array(data)
        mean = np.mean(data)
        std = np.std(data)
        return (data - mean) / std
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def get_json_files(data_folder):
    json_files = []
    for root, dirs, files in os.walk(data_folder):
        for file in files:
            if file.endswith('.json'):
                json_files.append(os.path.join(root, file))
    return json_files

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


def train_autoencoder(model, dataloader, optimizer, criterion, device, epochs=200):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for data in dataloader:
            data = data.to(device).unsqueeze(1)
            optimizer.zero_grad()
            #outputs, l1_loss = model(data)
            # recon_loss = criterion(outputs, data)
            # loss = recon_loss + l1_loss
            outputs = model(data)
            print(f'data: {data.shape}, outputs: {outputs.shape}')
            loss = criterion(outputs, data)
            print(f'loss is {loss}')

            loss.backward()
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

class TransformerEncoder(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, num_layers, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

    def forward(self, src):
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class PulseAnomalyDetector(nn.Module):
    def __init__(self, input_size, d_model, nhead, dim_feedforward, num_layers, dropout=0.1):
        super(PulseAnomalyDetector, self).__init__()
        self.encoder = nn.Linear(input_size, d_model)
        self.transformer = TransformerEncoder(d_model, nhead, dim_feedforward, num_layers, dropout)
        self.decoder = nn.Linear(d_model, input_size)

    def forward(self, src, src_key_padding_mask):
        src = self.encoder(src)
        latent = self.transformer(src, src_key_padding_mask=src_key_padding_mask)
        output = self.decoder(latent)
        return output, latent

# def train(model, dataloader, optimizer, criterion, epochs):
#     for epoch in range(epochs):
#         running_loss = 0.0
#         for pulse in dataloader:
#             optimizer.zero_grad()
#             output, _ = model(pulse)
#             loss = criterion(output, pulse)
#             loss.backward()
#             optimizer.step()
#             running_loss += loss.item()
#         print(f"Epoch {epoch+1}, Loss: {running_loss / len(dataloader)}")

def train(model, dataloader, optimizer, criterion, device, epochs=100):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for pulses in dataloader:
            pulses = [pulse.squeeze(0) for pulse in pulses]  # 移除批次維度
            lengths = [len(pulse) for pulse in pulses]  # 記錄每個脈衝的長度
            padded_pulses = nn.utils.rnn.pad_sequence(pulses, batch_first=True)  # 填充脈衝到相同長度
            padded_pulses = padded_pulses.transpose(1, 2).to(device)  # 轉置維度並移動到設備

            optimizer.zero_grad()
            output, _ = model(padded_pulses, lengths)
            loss = criterion(output, padded_pulses)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {running_loss / len(dataloader)}")

def detect_anomalies(model, dataloader, threshold, device):
    model.eval()
    anomalies = []
    with torch.no_grad():
        for pulses in dataloader:
            pulses = [pulse.squeeze(0) for pulse in pulses]  # 移除批次維度
            lengths = [len(pulse) for pulse in pulses]  # 記錄每個脈衝的長度
            padded_pulses = nn.utils.rnn.pad_sequence(pulses, batch_first=True)  # 填充脈衝到相同長度
            padded_pulses = padded_pulses.transpose(1, 2).to(device)  # 轉置維度並移動到設備

            output, _ = model(padded_pulses, lengths)
            loss = torch.mean((output - padded_pulses) ** 2, dim=(1, 2))
            anomaly_mask = loss > threshold
            anomalies.extend(anomaly_mask.cpu().numpy())
    return anomalies

def predict_per_second(model, json_file, sample_rate, device):
    with open(json_file, 'r') as f:
        json_data = json.load(f)
        signal = process_DB_rawdata(json_data) if 'smoothed_data' not in json_data else np.array(json_data['smoothed_data'], dtype=np.float32)
        if json_data.get('sample_rate', 100) != sample_rate:
            num_samples = int(len(signal) * sample_rate / json_data.get('sample_rate', 100))
            signal = scipy.signal.resample(signal, num_samples)

    # 對測試數據進行標準化處理
    signal = (signal - np.mean(signal)) / np.std(signal)    
    # input(f'signal shape: {signal.shape}')
    model.eval()
    criterion = nn.MSELoss()
    losses = []
    for i in range(0, len(signal) - sample_rate + 1, sample_rate):
        segment = signal[i:i+sample_rate]
        segment_tensor = torch.tensor(segment).view(1, 1, -1).to(device)
        with torch.no_grad():
            output = model(segment_tensor)
            loss = criterion(output, segment_tensor)
            losses.append(loss.item())
    
    return losses

def predict_per_pulse(model, json_file, sample_rate, device):
    with open(json_file, 'r') as f:
        json_data = json.load(f)
        signal = process_DB_rawdata(json_data) if 'smoothed_data' not in json_data else np.array(json_data['smoothed_data'], dtype=np.float32)
        if json_data.get('sample_rate', 100) != sample_rate:
            num_samples = int(len(signal) * sample_rate / json_data.get('sample_rate', 100))
            signal = scipy.signal.resample(signal, num_samples)
    # print(f'predict_per_pulse signal!')


    # 检测峰值
    x_points = detect_peaks_from_signal(signal, sample_rate)
    # print(f'x_points: {x_points}')
    model.eval()
    criterion = nn.MSELoss()
    losses = []

    for i in range(len(x_points) - 1):
        segment = signal[x_points[i]:x_points[i + 1]]
        if len(segment) > 0 and len(segment) < 200:
            #standardization
            segment = np.array(segment, dtype=np.float32)
            if np.std(segment) != 0:
                segment = (segment - np.mean(segment)) / np.std(segment)
            # print(f'segment: {segment}')
            segment = np.pad(segment, (0, 200 - len(segment)), 'constant')
            segment_tensor = torch.tensor(segment, dtype=torch.float32).view(1, 1, -1).to(device)
            with torch.no_grad():
                output = model(segment_tensor)
                # print(f'output: {output}')
                loss = criterion(output, segment_tensor)
                # print(f'loss:{loss}')
                losses.append(loss.item())

    return losses


def main():
    root = tk.Tk()
    root.title("Anomaly Detection Viewer")
    data_folder = 'DB' #'labeled_DB'
    training_folder = 'labeled_DB'
    window_size = 100  # Assuming each second contains 100 sampling points
    sample_rate = 100
    pulse_size = 200
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'torch device: {device}')

    input_size = 1
    d_model = 64
    nhead = 4
    dim_feedforward = 128
    num_layers = 2
    dropout = 0.1

    model = PulseAnomalyDetector(input_size, d_model, nhead, dim_feedforward, num_layers, dropout).to(device)
    # Initialize and train the U-Net autoencoder
    # model = UNetAutoencoder(1, window_size).to(device)
    # model = DeepUNetAutoencoder(1, pulse_size).to(device) 
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total number of model parameters: {trainable_params}')    
    if os.path.exists('transformer_anomaly_detector.pt'): 
        #unet_autoencoder.pt: (UNetAutoencoder with no data standardization)
        #unet_autoencoder3.pt: (UNetAutoencoder with data standardization)  (Default)
        #unet_autoencoder4.pt: DeepUNetAutoencoder with data standardization  
        #unet_autoencoder5.pt: DeepUNetAutoencoder with padded-pulsewise input and data standardization
        model.load_state_dict(torch.load('transformer_anomaly_detector.pt'))
    # else:
    # Load data and create data loaders
    train_dataset, test_dataset, train_files, test_files = load_data(training_folder, window_size, sample_rate)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # train_autoencoder(model, train_loader, optimizer, criterion, device)
    train(model, train_loader, optimizer, criterion, device)
    # # #save model
    torch.save(model.state_dict(), 'unet_autoencoder6.pt')

    # results = {}
    # for subject_folder in os.listdir(data_folder):
    #     subject_path = os.path.join(data_folder, subject_folder)
    #     if os.path.isdir(subject_path):
    #         results[subject_folder] = {}
    #         for json_file in os.listdir(subject_path):
    #             if json_file.endswith('.json'):
    #                 json_path = os.path.join(subject_path, json_file)
                    
    #                 # 檢查訓練資料夾中是否存在相同相對路徑的檔案
    #                 training_json_path = os.path.join(training_folder, subject_folder, json_file)
    #                 if os.path.exists(training_json_path):
    #                     print(f"Skipping prediction for {subject_folder}/{json_file} (already trained)")
    #                     continue
                    
    #                 try:
    #                     # losses = predict_per_second(model, json_path, sample_rate, device)
    #                     losses = predict_per_pulse(model, json_path, sample_rate, device)
    #                     mean_loss = np.mean(losses)
    #                     std_loss = np.std(losses)
    #                     results[subject_folder][json_file] = f"{mean_loss:.7f}+-{std_loss:.7f}"
    #                     print(f"{subject_folder}/{json_file}: {mean_loss:.7f}+-{std_loss:.7f}")
    #                     # print(f'losses: {losses}')
    #                 except Exception as e:
    #                     print(f"Exception {subject_folder}/{json_file}: {e}")

    # # Save results to a JSON file
    # with open('prediction_results.json', 'w', encoding='utf-8') as f:
    #     json.dump(results, f, indent=4)

    # print("Prediction results saved to 'prediction_results.json'.")
    # train_files = []
    # test_files = []
    # app = Application(model, device, train_files, test_files,  master=root)
    # app.mainloop()

if __name__ == "__main__":
    main()