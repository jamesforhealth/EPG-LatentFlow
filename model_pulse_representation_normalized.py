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
from preprocessing import process_DB_rawdata, get_json_files, add_noise_with_snr, add_gaussian_noise_torch, PulseDataset, PulseDatasetNormalized, baseline_correction
from model_find_peaks import detect_peaks_from_signal
from tqdm import tqdm
import math
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import h5py

from datautils import save_encoded_data, predict_latent_vector_list, predict_encoded_dataset


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


def collate_fn(batch):
    batch = [seq for seq in batch if seq is not None]
    sequences = [seq for seq in batch]
    padded_sequences = nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=0.0)
    return padded_sequences




def loss_function(recon_x, x, mu, logvar):
    BCE = nn.MSELoss()(recon_x, x)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

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

def train_autoencoder(model, train_dataloader, test_dataloader, optimizer, criterion, device, model_path, epochs=3000, save_interval=1):
    model.train()
    min_loss = float('inf')

    # 嘗試載入已有的模型參數
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print(f"Loaded model parameters from {model_path}")

    #Compute init loss
    init_loss = evaluate_model(model, test_dataloader, criterion, device)
    input(f"Init Testing Loss: {init_loss:.10f}")

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
    def __init__(self, target_len, hidden_dim=40, latent_dim=30, dropout=0.5):#, latent_dim=30, dropout=0.5):
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

class EPGBaselinePulseAutoencoderDeep(nn.Module):
    def __init__(self, target_len=100, hidden_dim=60, hidden_dim2=40, latent_dim=20, dropout=0.5):#, latent_dim=30, dropout=0.5):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(target_len, hidden_dim),
            # nn.Dropout(dropout),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim2),
            nn.BatchNorm1d(hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, latent_dim)
        )
        self.dec = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim2),
            # nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(hidden_dim2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, target_len)
        )

    def forward(self, x):
        z = self.enc(x)
        pred = self.dec(z)
        return pred, z
    
    def encode(self, x):
        return self.enc(x)


def predict_reconstructed_signal(signal, sample_rate, peaks):
    """
    使用訓練好的自編碼器重建信號，並返回每個脈衝的潛在向量
    """
    target_len = 100
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = 'pulse_interpolate_autoencoder_1125_normalized_test40.pth'
    model = EPGBaselinePulseAutoencoder(target_len=target_len, hidden_dim=50, latent_dim=40).to(device)
    
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # 複製原始信號
    origin_signal = np.copy(signal)

    # 重採樣信號
    resample_ratio = 1.0
    if sample_rate != 100:
        resample_ratio = 100 / sample_rate
        signal = scipy.signal.resample(signal, int(len(signal) * resample_ratio))
        peaks = [int(p * resample_ratio) for p in peaks]

    # 基線校正和標準化
    signal = baseline_correction(signal, 100)

    # 逐拍重建
    latent_vector_list = []
    reconstructed_signal = np.copy(signal)
    
    for i in range(len(peaks) - 1):
        start_idx = peaks[i]
        end_idx = peaks[i + 1]
        pulse = signal[start_idx:end_idx]
        pulse_length = end_idx - start_idx

        if pulse_length > 1:
            # 記錄原始幅度
            original_amplitude = pulse[0]
            
            # 歸一化脈衝
            if original_amplitude != 0:
                pulse = pulse / original_amplitude
            
            # 插值到目標長度
            interp_func = scipy.interpolate.interp1d(np.arange(pulse_length), pulse, kind='linear', fill_value="extrapolate")
            pulse_resampled = interp_func(np.linspace(0, pulse_length - 1, target_len))
            pulse_tensor = torch.tensor(pulse_resampled, dtype=torch.float32).unsqueeze(0).to(device)

            with torch.no_grad():
                reconstructed_pulse, latent_vector = model(pulse_tensor)
                reconstructed_pulse = reconstructed_pulse.squeeze().cpu().numpy()
                latent_vector = latent_vector.squeeze().cpu().numpy()
                
                # 添加時長和幅度信息
                latent_vector = np.concatenate([latent_vector, np.array([pulse_length/100, original_amplitude])])
                latent_vector_list.append(latent_vector)

            # 將重建的脈衝還原為原始長度
            interp_func_reconstructed = scipy.interpolate.interp1d(
                np.linspace(0, target_len - 1, target_len), 
                reconstructed_pulse, 
                kind='linear', 
                fill_value="extrapolate"
            )
            reconstructed_pulse_resampled = interp_func_reconstructed(
                np.linspace(0, target_len - 1, pulse_length)
            )
            
            # 還原原始幅度
            reconstructed_pulse_resampled = reconstructed_pulse_resampled * original_amplitude
            reconstructed_signal[start_idx:end_idx] = reconstructed_pulse_resampled

    # 根據原始採樣率調整重構信號的長度
    if resample_ratio != 1.0:
        original_length = int(len(reconstructed_signal) / resample_ratio)
        reconstructed_signal = scipy.signal.resample(reconstructed_signal, original_length)

    return reconstructed_signal#, latent_vector_list

def encode_pulses(model, json_files, target_len=100, sample_rate=100, device='cuda'):
    """
    將整個數據集的脈衝編碼為42維向量（40維潛在空間 + 時長 + 幅度）
    """
    model.eval()
    encoded_data = {}

    for json_file in tqdm(json_files, desc="Processing files"):
        try:
            with open(json_file, 'r') as f:
                json_data = json.load(f)
                
                # 跳過有異常標記的文件
                if json_data['anomaly_list']:
                    continue
                
                signal = json_data['raw_data']
                original_sample_rate = json_data.get('sample_rate', 100)
                x_points = json_data['x_points']

                # 重採樣
                if original_sample_rate != sample_rate:
                    num_samples = int(len(signal) * sample_rate / original_sample_rate)
                    signal = scipy.signal.resample(signal, num_samples)
                    x_points = [int(x * sample_rate / original_sample_rate) for x in x_points]
                
                # 基線校正和標準化
                signal = baseline_correction(signal, sample_rate)

                latent_vectors = []
                for j in range(len(x_points) - 1):
                    pulse_start = x_points[j]
                    pulse_end = x_points[j + 1]
                    pulse = signal[pulse_start:pulse_end]
                    
                    if len(pulse) > 1:
                        # 記錄原始幅度和時長
                        original_amplitude = pulse[0]
                        pulse_length = len(pulse)
                        
                        # 歸一化
                        if original_amplitude != 0:
                            pulse = pulse / original_amplitude
                            
                            # 插值到固定長度
                            interp_func = scipy.interpolate.interp1d(np.arange(len(pulse)), pulse, kind='linear')
                            pulse_resampled = interp_func(np.linspace(0, len(pulse) - 1, target_len))
                            pulse_tensor = torch.tensor(pulse_resampled, dtype=torch.float32).unsqueeze(0).to(device)

                            with torch.no_grad():
                                _, latent_vector = model(pulse_tensor)
                            
                            # 組合42維向量
                            latent_vector = latent_vector.squeeze().cpu().numpy()
                            latent_vector = np.concatenate([
                                latent_vector, 
                                np.array([pulse_length/100, original_amplitude])
                            ])
                            latent_vectors.append(latent_vector)

                # 獲取相對路徑
                relative_path = os.path.relpath(json_file, 'wearing_consistency')
                encoded_data[relative_path] = np.array(latent_vectors)

        except Exception as e:
            print(f'Error processing {json_file}: {e}')
            continue

    return encoded_data


def search_optimal_latent_dim(train_dataloader, test_dataloader, target_len, device, 
                            latent_dims=list(range(15, 21)), 
                            epochs=1000):
    results = {}
    
    for latent_dim in latent_dims:
        print(f"\nTesting latent_dim = {latent_dim}")
        
        # 使用與原始模型相同的配置
        model = EPGBaselinePulseAutoencoder(
            target_len=target_len,
            hidden_dim=40,  # 固定為40，與原始配置相同
            latent_dim=latent_dim,
            dropout=0.9     # 與原始配置相同
        ).to(device)
        
        # 使用相同的優化器配置
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=5e-4)
        criterion = nn.L1Loss()
        
        # 訓練追蹤變量
        best_val_loss = float('inf')
        best_model_state = None
        
        # 訓練循環
        for epoch in range(epochs):
            # 訓練階段
            model.train()
            train_loss = 0
            for batch in train_dataloader:
                pulse = batch.to(device)
                optimizer.zero_grad()
                output, _ = model(pulse)
                loss = criterion(output, pulse)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_dataloader)
            
            # 驗證階段
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in test_dataloader:
                    pulse = batch.to(device)
                    output, _ = model(pulse)
                    loss = criterion(output, pulse)
                    val_loss += loss.item()
            val_loss /= len(test_dataloader)
            
            # 更新最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict()
            
            if epoch % 50 == 0:
                print(f"Epoch [{epoch}/{epochs}], Train Loss: {train_loss:.10f}, Val Loss: {val_loss:.10f}")
        
        # 記錄結果
        results[latent_dim] = {
            'best_val_loss': best_val_loss,
            'model_state': best_model_state
        }
        
        print(f"Latent dim {latent_dim}: Best validation MAE = {best_val_loss:.10f}")
    
    # 分析結果
    print("\nResults Summary:")
    for dim in latent_dims:
        print(f"Latent dim {dim}: MAE = {results[dim]['best_val_loss']:.10f}")
    
    # 找出最佳維度
    best_dim = min(results.keys(), key=lambda k: results[k]['best_val_loss'])
    best_loss = results[best_dim]['best_val_loss']
    
    # 保存最佳模型
    torch.save(results[best_dim]['model_state'], 
              f'pulse_interpolate_autoencoder_optimal_dim_{best_dim}.pth')
    
    # 繪製維度-損失關係圖
    import matplotlib.pyplot as plt
    dims = list(results.keys())
    losses = [results[d]['best_val_loss'] for d in dims]
    
    plt.figure(figsize=(10, 6))
    plt.plot(dims, losses, 'bo-')
    plt.xlabel('Latent Dimension')
    plt.ylabel('Validation MAE')
    plt.title('Latent Dimension vs. Validation MAE')
    plt.grid(True)
    plt.savefig('latent_dim_search_results.png')
    plt.close()
    
    return best_dim, best_loss, results

def main():
    data_folder = 'wearing_consistency' #'labeled_DB'#
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'torch device: {device}')
    json_files = get_json_files(data_folder, exclude_keywords=[])  # 实现一个函数来获取所有的JSON文件路径

    # 設置參數
    batch_size = 32
    lr = 1e-4
    target_len = 100


    # 加載並劃分數據集
    dataset = PulseDatasetNormalized(json_files, target_len)
    train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    #計算訓練資料跟測試資料的長度
    print(f'train_data_len:{len(train_data)}, test_data_len:{len(test_data)}')
    # 初始化模型和優化器

    # model_path = 'pulse_interpolate_autoencoder_1107_normalized.pth'
    # model_path = 'pulse_interpolate_autoencoder_1108_normalized_test.pth' 
    # model_path = 'pulse_interpolate_autoencoder_1115_normalized_test.pth' #latent_dim=20
    # model_path = 'pulse_interpolate_autoencoder_1116_normalized_test.pth' #latent_dim=25
    # model_path = 'pulse_interpolate_autoencoder_1118_normalized_test.pth' #latent_dim=20  Training Loss: 0.0089707072, Testing Loss: 0.0108602159
    # model_path = 'pulse_interpolate_autoencoder_1118_normalized_test23.pth' #latent_dim=23 Training Loss: 0.0075240957, Testing Loss: 0.0098391391
    # model_path = 'pulse_interpolate_autoencoder_1118_normalized_test26.pth' #latent_dim=26 Training Loss: 0.0057025418, Testing Loss: 0.0078531370
    # model_path = 'pulse_interpolate_autoencoder_1118_normalized_test30.pth' #latent_dim=30 Training Loss: 0.0041844422, Testing Loss: 0.0065536414
    # model_path = 'pulse_interpolate_autoencoder_1118_normalized_test33.pth' #latent_dim=33 Training Loss: 0.0034583039, Testing Loss: 0.0056102187
    # model_path = 'pulse_interpolate_autoencoder_1118_normalized_test36.pth' #latent_dim=36 Training Loss: 0.0030284102, Testing Loss: 0.0053330789
    model_path = 'pulse_interpolate_autoencoder_1118_normalized_test40.pth' #latent_dim=40 Training Loss: 0.0027490317, Testing Loss: 0.0048130364
    model_path = 'pulse_interpolate_autoencoder_1125_normalized_test40.pth' #Default! Testing Loss: 0.0024645163
    model_path = 'pulse_interpolate_autoencoder_1126_normalized_test25.pth' #latent_dim=25 Training Loss: 0.0042386764, Testing Loss: 0.0054548768
    model_path = 'pulse_interpolate_autoencoder_1126_normalized_test30.pth' #latent_dim=30 Training Loss: 0.0028783731, Testing Loss: 0.0039177004
    model = EPGBaselinePulseAutoencoder(target_len=target_len, hidden_dim=50, latent_dim=30).to(device)
    
    # model_path = 'pulse_interpolate_autoencoder_deep_1120_normalized_test20.pth' #latent_dim=20 Training Loss: 0.0093327749, Testing Loss: 0.0088772793
    # model = EPGBaselinePulseAutoencoderDeep().to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print(f"Loaded model parameters from {model_path}")

    criterion = nn.L1Loss()  #Use L1Loss to train first
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)#5e-4   #Adam
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total number of model parameters: {trainable_params}, model:{model}') 
    train_autoencoder(model, train_dataloader, test_dataloader, optimizer, criterion, device, model_path)

    # encoded_data = encode_pulses(model, json_files, target_len, sample_rate=100, device=device)
    # save_encoded_data(encoded_data, output_dir='encoded_pulse_sequences')


    # input('Press Enter to continue...')
    # # 搜索最佳維度
    # best_dim, best_loss, results = search_optimal_latent_dim(
    #     train_dataloader, 
    #     test_dataloader,
    #     target_len=target_len,
    #     device=device,
    #     latent_dims=list(range(14, 18)),
    #     epochs=500,
    # )
    
    # print(f"\nOptimal latent dimension: {best_dim}")
    # print(f"Best validation MAE: {best_loss:.6f}")
    
    # # 繪製維度-損失關係圖
    # import matplotlib.pyplot as plt
    # dims = list(results.keys())
    # losses = [results[d]['best_val_loss'] for d in dims]
    
    # plt.figure(figsize=(10, 6))
    # plt.plot(dims, losses, 'bo-')
    # plt.xlabel('Latent Dimension')
    # plt.ylabel('Validation MAE')
    # plt.title('Latent Dimension vs. Validation MAE')
    # plt.grid(True)
    # plt.savefig('latent_dim_search_results.png')
    # plt.close()


if __name__ == '__main__':
    main()