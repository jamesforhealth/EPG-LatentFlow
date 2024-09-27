
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
import matplotlib.pyplot as plt
import os
import sys
import json
import random
import scipy
from torchviz import make_dot
from preprocessing import process_DB_rawdata, get_json_files, add_noise_with_snr, add_gaussian_noise_torch, MeasurementPulseDataset, PulseDataset
from model_find_peaks import detect_peaks_from_signal
from tqdm import tqdm
import math
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import h5py
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from datautils import save_encoded_data, predict_latent_vector_list, predict_encoded_dataset
from preprocessing import split_json_files
def get_current_model_device():
    target_len = 100
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = './DisentangledAutoencoder.pth'
    model_path = './DisentangledAutoencoder_pretrain_wearing.pth'
    model_path = './DisentangledAutoencoder_pretrain_wearing2.pth' #
    model = DisentangledAutoencoder(target_len).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, device

def predict_reconstructed_signal(signal, sample_rate, peaks):
    target_len = 100
    model, device = get_current_model_device()
    # 复制一个原始信号的数组值
    origin_signal = np.copy(signal)

    # 重采样信号
    resample_ratio = 1.0
    if sample_rate != 100:
        resample_ratio = 100 / sample_rate
        signal = scipy.signal.resample(signal, int(len(signal) * resample_ratio))
        peaks = [int(p * resample_ratio) for p in peaks]  # 调整peaks索引

    # 全局标准化
    mean = np.mean(signal)
    std = np.std(signal)
    signal = (signal - mean) / std

    # 逐拍重建
    physio_vector_list = []
    wear_vector_list = []
    reconstructed_signal = np.copy(signal)
    for i in range(len(peaks) - 1):
        start_idx = peaks[i]
        end_idx = peaks[i + 1]
        pulse = signal[start_idx:end_idx]
        pulse_length = end_idx - start_idx  # 记录脉冲的原始长度

        if pulse_length > 1:
            # 插值到目标长度
            interp_func = scipy.interpolate.interp1d(np.arange(pulse_length), pulse, kind='linear', fill_value="extrapolate")
            pulse_resampled = interp_func(np.linspace(0, pulse_length - 1, target_len))
            pulse_tensor = torch.tensor(pulse_resampled, dtype=torch.float32).unsqueeze(0).to(device)

            with torch.no_grad():
                reconstructed_pulse, latent_vector = model(pulse_tensor)
                reconstructed_pulse = reconstructed_pulse.squeeze().cpu().numpy()
                # physio_vector = physio_vector.squeeze().cpu().numpy()
                # wear_vector = wear_vector.squeeze().cpu().numpy()

                # physio_vector_list.append(physio_vector)
                # wear_vector_list.append(wear_vector)

            # 将重建的脉冲还原为原始长度
            interp_func_reconstructed = scipy.interpolate.interp1d(np.linspace(0, target_len - 1, target_len), reconstructed_pulse, kind='linear', fill_value="extrapolate")
            reconstructed_pulse_resampled = interp_func_reconstructed(np.linspace(0, target_len - 1, pulse_length))
            reconstructed_signal[start_idx:end_idx] = reconstructed_pulse_resampled

    # # 计算前后physio_vector和wear_vector之间的相似程度
    # physio_similarity_list = []
    # wear_similarity_list = []
    # for i in range(len(physio_vector_list) - 1):
    #     this_physio = physio_vector_list[i]
    #     next_physio = physio_vector_list[i + 1]
    #     this_wear = wear_vector_list[i]
    #     next_wear = wear_vector_list[i + 1]

    #     physio_similarity = np.dot(this_physio, next_physio) / (np.linalg.norm(this_physio) * np.linalg.norm(next_physio))
    #     wear_similarity = np.dot(this_wear, next_wear) / (np.linalg.norm(this_wear) * np.linalg.norm(next_wear))
        
    #     physio_similarity_list.append(physio_similarity)
    #     wear_similarity_list.append(wear_similarity)

    # 反标准化
    reconstructed_signal = reconstructed_signal * std + mean

    # 根据原始采样率调整重构信号的长度
    original_length = int(len(reconstructed_signal) / resample_ratio)
    reconstructed_signal = scipy.signal.resample(reconstructed_signal, original_length)

    # 计算原始信号和重构信号的MAE
    mae = np.mean(np.abs(origin_signal - reconstructed_signal))
    print(f'MAE: {mae}')

    return reconstructed_signal#, physio_vector_list, wear_vector_list, physio_similarity_list, wear_similarity_list


def predict_corrected_reconstructed_signal(signal, sample_rate, peaks):
    wearing_mean = torch.tensor([-0.03277407,-0.16876778,-0.05552121,-0.17526882,0.09692444,-0.14301368,
    0.21773008,-0.05491834,0.23559649,0.1858692], dtype=torch.float32)
    target_len = 100
    model, device = get_current_model_device()
    wearing_mean = wearing_mean.to(device)
    # 复制一个原始信号的数组值
    origin_signal = np.copy(signal)

    # 重采样信号
    resample_ratio = 1.0
    if sample_rate != 100:
        resample_ratio = 100 / sample_rate
        signal = scipy.signal.resample(signal, int(len(signal) * resample_ratio))
        peaks = [int(p * resample_ratio) for p in peaks]  # 调整peaks索引

    # 全局标准化
    mean = np.mean(signal)
    std = np.std(signal)
    signal = (signal - mean) / std

    # 逐拍重建
    reconstructed_signal = np.copy(signal)
    for i in range(len(peaks) - 1):
        start_idx = peaks[i]
        end_idx = peaks[i + 1]
        pulse = signal[start_idx:end_idx]
        pulse_length = end_idx - start_idx  # 记录脉冲的原始长度

        if pulse_length > 1:
            # 插值到目标长度
            interp_func = scipy.interpolate.interp1d(np.arange(pulse_length), pulse, kind='linear', fill_value="extrapolate")
            pulse_resampled = interp_func(np.linspace(0, pulse_length - 1, target_len))
            pulse_tensor = torch.tensor(pulse_resampled, dtype=torch.float32).unsqueeze(0).to(device)

            with torch.no_grad():
                _, latent_vector = model(pulse_tensor)
                latent_vector[:, :10] = wearing_mean
                reconstructed_pulse = model.dec(latent_vector)
                reconstructed_pulse = reconstructed_pulse.squeeze ().cpu().numpy()

            # 将重建的脉冲还原为原始长度
            interp_func_reconstructed = scipy.interpolate.interp1d(np.linspace(0, target_len - 1, target_len), reconstructed_pulse, kind='linear', fill_value="extrapolate")
            reconstructed_pulse_resampled = interp_func_reconstructed(np.linspace(0, target_len - 1, pulse_length))
            reconstructed_signal[start_idx:end_idx] = reconstructed_pulse_resampled

    # 反标准化
    reconstructed_signal = reconstructed_signal * std + mean

    # 根据原始采样率调整重构信号的长度
    original_length = int(len(reconstructed_signal) / resample_ratio)
    reconstructed_signal = scipy.signal.resample(reconstructed_signal, original_length)

    # 计算原始信号和重构信号的MAE
    mae = np.mean(np.abs(origin_signal - reconstructed_signal))
    print(f'MAE: {mae}')

    return reconstructed_signal

class MultiResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.shortcut = nn.Conv1d(in_channels, out_channels, 1)
        self.conv1 = nn.Conv1d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, padding=1)
        self.conv3 = nn.Conv1d(out_channels, out_channels, 3, padding=1)

    def forward(self, x):
        shortcut = self.shortcut(x)
        conv1 = F.relu(self.conv1(x))
        conv2 = F.relu(self.conv2(conv1))
        conv3 = F.relu(self.conv3(conv2))
        return shortcut + conv3

class DisentangledMultiResAutoencoder(nn.Module):
    def __init__(self, input_length, input_channels, hidden_channels, physio_dim, wear_dim):
        super().__init__()
        self.physio_dim = physio_dim
        self.wear_dim = wear_dim

        # 生理信号编码器
        self.physio_enc = nn.Sequential(
            MultiResBlock(input_channels, hidden_channels),
            nn.MaxPool1d(2),
            MultiResBlock(hidden_channels, hidden_channels*2),
            nn.MaxPool1d(2),
            MultiResBlock(hidden_channels*2, hidden_channels*2),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden_channels*2, physio_dim)
        )

        # 穿戴状态编码器
        self.wear_enc = nn.Sequential(
            MultiResBlock(input_channels, hidden_channels),
            nn.MaxPool1d(2),
            MultiResBlock(hidden_channels, hidden_channels*2),
            nn.MaxPool1d(2),
            MultiResBlock(hidden_channels*2, hidden_channels*2),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden_channels*2, wear_dim)
        )

        # # 解码器
        self.dec = nn.Sequential(
            nn.Linear(physio_dim + wear_dim, hidden_channels * (input_length // 4)),
            nn.Unflatten(1, (hidden_channels, input_length // 4)),
            nn.ConvTranspose1d(hidden_channels, hidden_channels*2, 4, stride=2, padding=1),
            MultiResBlock(hidden_channels*2, hidden_channels*2),
            nn.ConvTranspose1d(hidden_channels*2, hidden_channels, 4, stride=2, padding=1),
            MultiResBlock(hidden_channels, hidden_channels),
            nn.Conv1d(hidden_channels, input_channels, 1)
        )

    def forward(self, x):
        # 确保输入是 3D 的 (batch_size, channels, sequence_length)
        # print(f'x.shape: {x.shape}')
        x = x.unsqueeze(1)
        z_physio = self.physio_enc(x)
        z_wear = self.wear_enc(x)
        # print(f'z_physio.shape: {z_physio.shape}, z_wear.shape: {z_wear.shape}')
        z = torch.cat([z_physio, z_wear], dim=1)
        # print(f'z.shape: {z.shape}')

        decoded = self.dec(z)
        # print(f'decoded.shape: {decoded.shape}')

        # 如果原始输入是 2D 的，我们需要将输出压缩回 2D
        decoded = decoded.squeeze(1)
        # input(f'decoded.shape: {decoded.shape}')
        return decoded, z_physio, z_wear

    def encode_physio(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        return self.physio_enc(x)

    def encode_wear(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        return self.wear_enc(x)
    
class DisentangledAutoencoder(nn.Module):
    def __init__(self, target_len, hidden_dim=50, physio_dim=15, wear_dim=10):
        super().__init__()
        self.physio_dim = physio_dim
        self.wear_dim = wear_dim
        
        self.physio_enc = nn.Sequential(
            nn.Linear(target_len, hidden_dim),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(hidden_dim, physio_dim),
            # nn.Tanh()  # 使用Tanh來限制輸出範圍
        )
        
        self.wear_enc = nn.Sequential(
            nn.Linear(target_len, hidden_dim),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(hidden_dim, wear_dim),
            # nn.Tanh()  # 使用Tanh來限制輸出範圍
        )
        
        self.dec = nn.Sequential(
            nn.Linear(physio_dim + wear_dim, hidden_dim),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(hidden_dim, target_len)
        )

    def forward(self, x):
        # print(f'x.shape: {x.shape}')
        z_physio = self.physio_enc(x)
        # print(f'z_physio.shape: {z_physio.shape}')
        z_wear = self.wear_enc(x)
        # print(f'z_wear.shape: {z_wear.shape}')
        z = torch.cat([z_physio, z_wear], dim=1)
        # print(f'z.shape: {z.shape}')
        pred = self.dec(z)
        # input(f'pred.shape: {pred.shape}')
        return pred, z#z_physio, z_wear

    def encode_physio(self, x):
        return self.physio_enc(x)

    def encode_wear(self, x):
        return self.wear_enc(x)

def pretrain_wear_enc(model, dataset, batch_size, device, loss_fn, optimizer):
    model.train()
    total_loss = 0
    num_batches = 0
    
    for measurement_id in dataset.measurement_indices:
        measurement_data = dataset.get_measurement(measurement_id)
        
        # Use sliding window approach
        for i in range(len(measurement_data) - batch_size + 1):
            batch = torch.stack(measurement_data[i:i+batch_size])
            batch = batch.to(device)
            
            optimizer.zero_grad()
            for param in model.wear_enc.parameters():
                param.requires_grad = True
            for param in model.physio_enc.parameters():
                param.requires_grad = False
            for param in model.dec.parameters():
                param.requires_grad = False
            
            _, latent_vector = model(batch)
            #z_wear 是 latent_vector的前10個維度的向量
            z_wear = latent_vector[:, :model.wear_dim]
            wear_consistency_loss = loss_fn(z_wear, z_wear.mean(dim=0, keepdim=True).expand_as(z_wear))
            wear_consistency_loss.backward()
            optimizer.step()
            
            total_loss += wear_consistency_loss.item()
            # input(f'i:{i},wear_consistency_loss.item(): {wear_consistency_loss.item()}')
            num_batches += 1
    # print(f'Pretraining wear encoder num_batches: {num_batches}, total_loss: {total_loss}')
    return total_loss / num_batches

def train_step(model, optimizer, batch, mode, device, loss_fn):
    optimizer.zero_grad()
    batch = batch.to(device)
    if mode == 'physio':
        # 先训练穿戴状态编码器以保持一致性
        # optimizer.zero_grad()
        # for param in model.wear_enc.parameters():
        #     param.requires_grad = True
        # for param in model.physio_enc.parameters():
        #     param.requires_grad = False
        # for param in model.dec.parameters():
        #     param.requires_grad = False
        
        # _, latent_vector = model(batch)
        # wear_consistency_loss = loss_fn(z_wear, z_wear.mean(dim=0, keepdim=True).expand_as(z_wear))
        # wear_consistency_loss.backward()
        # optimizer.step()
    
    

        # 再训练生理编码器和解码器
        optimizer.zero_grad()
        for param in model.wear_enc.parameters():
            param.requires_grad = False
        # for param in model.physio_enc.parameters():
        #     param.requires_grad = True
        # for param in model.dec.parameters():
        #     param.requires_grad = True
        
        preds, latent_vector = model(batch)
        recon_loss = loss_fn(preds, batch)
        recon_loss.backward()
        optimizer.step()

        loss = recon_loss.item() #+ wear_consistency_loss.item()
    
    elif mode == 'wear':
        optimizer.zero_grad()
        for param in model.wear_enc.parameters():
            param.requires_grad = True
        # for param in model.parameters():
        #     param.requires_grad = True
        preds, latent_vector = model(batch)
        recon_loss = loss_fn(preds, batch)
        recon_loss.backward()
        optimizer.step()
        
        loss = recon_loss.item()

    return loss

def train_epoch(model, optimizer, dataset, batch_size, device, loss_fn):
    model.train()
    total_loss = 0
    num_batches = 0
    
    # 訓練生理編碼器
    for measurement_id in dataset.measurement_indices:
        measurement_data = dataset.get_measurement(measurement_id)
        for i in range(0, len(measurement_data), batch_size):
            batch = torch.stack(measurement_data[i:i+batch_size])
            loss = train_step(model, optimizer, batch, mode='physio', device=device, loss_fn=loss_fn)
            total_loss += loss
            num_batches += 1
    print(f'train_epoch: physio_loss: {total_loss / num_batches}')#, num_batches: {num_batches}')
    
    total_loss = 0
    num_batches = 0
    # 訓練穿戴狀態編碼器
    all_pulses = list(range(len(dataset)))
    random.shuffle(all_pulses)
    for i in range(0, len(all_pulses), batch_size):
        batch_indices = all_pulses[i:i+batch_size]
        batch = torch.stack([dataset[idx] for idx in batch_indices])
        loss = train_step(model, optimizer, batch, mode='wear', device=device, loss_fn=loss_fn)
        total_loss += loss
        num_batches += 1
    # print(f' num_batches: {num_batches}')
    return total_loss / num_batches

def validate(model, dataloader, device, loss_fn):
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            preds,latent_vector = model(batch)
            loss = loss_fn(preds, batch)
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches

def train(model, train_dataset, val_dataloader, model_path, num_epochs, batch_size, device, loss_fn = F.l1_loss):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    best_val_loss = float('inf')
    

    print("Pretraining wear_enc...")
    for epoch in range(8):  # You can adjust the number of pretraining epochs
        pretrain_loss = pretrain_wear_enc(model, train_dataset, batch_size, device, loss_fn, optimizer)
        print(f"Pretrain Epoch {epoch}, Loss: {pretrain_loss:.10f}")
    input("Pretrained wear_enc...")
    for param in model.parameters():
        param.requires_grad = True
    for epoch in range(num_epochs):
        # 在前半部分使用 MAE，后半部分使用 MSE
        if epoch < num_epochs // 2:
            loss_fn = F.l1_loss 
        else:
            loss_fn = F.mse_loss
        loss_fn = F.l1_loss
        train_loss = train_epoch(model, optimizer, train_dataset, batch_size, device, loss_fn)
        val_loss = validate(model, val_dataloader, device, loss_fn)
        
        print(f"Epoch {epoch}, Train Loss: {train_loss:.10f}, Val Loss: {val_loss:.10f}")
        
        if val_loss < best_val_loss * 0.97:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_path)
            print(f"New best model saved with validation loss: {val_loss:.10f}")


def main():
    data_folder = 'labeled_DB'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'torch device: {device}')
    target_len = 100
    batch_size = 32
    json_files = get_json_files(data_folder)
    train_files, val_files = split_json_files(json_files)
    print(f'number of train files: {len(train_files)}, number of val files: {len(val_files)}')
    train_dataset  = MeasurementPulseDataset(train_files, target_len)
    val_dataset = MeasurementPulseDataset(val_files, target_len)
    print(f'train_dataset size: {len(train_dataset )}, val_dataset size: {len(val_dataset)}')
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = DisentangledAutoencoder(target_len).to(device)

    # input_channels = 1  # 如果您的输入是单通道的
    # hidden_channels = 8  # 这个值可以调整
    # physio_dim = 15  # 您可以调整这个值
    # wear_dim = 15  # 您可以调整这个值
    # model = DisentangledMultiResAutoencoder(target_len, input_channels, hidden_channels, physio_dim, wear_dim).to(device)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total number of model parameters: {trainable_params}, model:{model}') 
    model_path = './DisentangledAutoencoder.pth' #DisentangledAutoencoder
    model_path = './DisentangledAutoencoder2.pth'  #DisentangledAutoencoder without TICP data
    model_path = './DisentangledAutoencoder_pretrain_wearing.pth'
    model_path = './DisentangledAutoencoder_pretrain_wearing2.pth' #physio_dim=15, wear_dim=10
    model_path = './DisentangledAutoencoder_pretrain_wearing_test.pth' #physio_dim=15, wear_dim=10
    # model_path = './DisentangledAutoencoder3.pth'
    # if os.path.exists(model_path): 
    #     model.load_state_dict(torch.load(model_path, map_location=device))
    train(model, train_dataset, val_dataloader, model_path, num_epochs=2000, batch_size=batch_size, device=device)
    encoded_data = predict_encoded_dataset(model, json_files)
    # save_encoded_data(encoded_data, 'latent_vectors_explanable')
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
if __name__ == '__main__':
    main()
