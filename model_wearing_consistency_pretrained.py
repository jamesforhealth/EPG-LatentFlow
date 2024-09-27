from tqdm import tqdm
import math
from sklearn.model_selection import train_test_split
import h5py
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import random
import os
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from preprocessing import process_DB_rawdata, get_json_files, add_noise_with_snr, add_gaussian_noise_torch, MeasurementPulseDataset, PulseDataset
import scipy
from scipy.interpolate import interp1d
import numpy as np

class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim_physical, latent_dim_physio):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        self.fc_physical = nn.Linear(128, latent_dim_physical * 2)
        self.fc_physio = nn.Linear(128, latent_dim_physio * 2)

    def forward(self, x):
        h = self.shared(x)
        physical = self.fc_physical(h)
        physio = self.fc_physio(h)
        return physical, physio

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

class Decoder(nn.Module):
    def __init__(self, latent_dim_physical, latent_dim_physio, output_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim_physical + latent_dim_physio, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )

    def forward(self, z_physical, z_physio):
        z = torch.cat([z_physical, z_physio], dim=1)
        return self.fc(z)

class Discriminator(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, z):
        return self.net(z).squeeze(-1)  # 确保输出是 [batch_size]

class DisentangledPulseVAE(nn.Module):
    def __init__(self, input_dim, latent_dim_physical, latent_dim_physio):
        super().__init__()
        self.encoder = Encoder(input_dim, latent_dim_physical, latent_dim_physio)
        self.decoder = Decoder(latent_dim_physical, latent_dim_physio, input_dim)
        self.discriminator_physical = Discriminator(latent_dim_physical)
        self.discriminator_physio = Discriminator(latent_dim_physio)
        self.latent_dim_physical = latent_dim_physical
        self.latent_dim_physio = latent_dim_physio

    def forward(self, x):
        physical, physio = self.encoder(x)
        mu_physical, logvar_physical = physical.chunk(2, dim=1)
        mu_physio, logvar_physio = physio.chunk(2, dim=1)
        
        z_physical = self.encoder.reparameterize(mu_physical, logvar_physical)
        z_physio = self.encoder.reparameterize(mu_physio, logvar_physio)
        
        x_recon = self.decoder(z_physical, z_physio)
        
        return x_recon, mu_physical, logvar_physical, mu_physio, logvar_physio, z_physical, z_physio

    def encode(self, x):
        """
        將輸入脈衝波形編碼為潛在向量。

        Args:
            x (torch.Tensor): 輸入脈衝波形,形狀為 (batch_size, input_dim)

        Returns:
            tuple: 包含物理和生理潛在向量的元組,每個形狀為 (batch_size, latent_dim)
        """
        physical, physio = self.encoder(x)
        mu_physical, logvar_physical = physical.chunk(2, dim=1)
        mu_physio, logvar_physio = physio.chunk(2, dim=1)
        
        z_physical = self.encoder.reparameterize(mu_physical, logvar_physical)
        z_physio = self.encoder.reparameterize(mu_physio, logvar_physio)
        
        return z_physical, z_physio

    def decode(self, z_physical, z_physio):
        """
        將潛在向量解碼回脈衝波形。

        Args:
            z_physical (torch.Tensor): 物理潛在向量,形狀為 (batch_size, latent_dim_physical)
            z_physio (torch.Tensor): 生理潛在向量,形狀為 (batch_size, latent_dim_physio)

        Returns:
            torch.Tensor: 重建的脈衝波形,形狀為 (batch_size, input_dim)
        """
        return self.decoder(z_physical, z_physio)

def train_model(model, train_dataset, val_dataset, num_epochs, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

    beta = 0.5  # β-VAE parameter
    best_val_loss = float('inf')

    # 計算初始損失
    initial_train_loss = calculate_loss(model, train_dataset, device)
    initial_val_loss = calculate_loss(model, val_dataset, device)
    print(f"Initial Train Loss: {initial_train_loss:.4f}")
    print(f"Initial Validation Loss: {initial_val_loss:.4f}")

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        measurement_ids = list(train_dataset.measurement_indices.keys())
        random.shuffle(measurement_ids)

        for measurement_id in tqdm(measurement_ids, desc=f"Epoch {epoch+1}/{num_epochs}"):
            optimizer.zero_grad()
            
            pulses = train_dataset.get_measurement(measurement_id)
            x = torch.stack(pulses).to(device)
            
            x_recon, mu_physical, logvar_physical, mu_physio, logvar_physio, z_physical, z_physio = model(x)
            
            # Reconstruction loss
            recon_loss = F.mse_loss(x_recon, x, reduction='sum') / x.size(0)
            
            # KL divergence loss
            kl_physical = -0.5 * torch.sum(1 + logvar_physical - mu_physical.pow(2) - logvar_physical.exp()) / x.size(0)
            kl_physio = -0.5 * torch.sum(1 + logvar_physio - mu_physio.pow(2) - logvar_physio.exp()) / x.size(0)
            
            # Consistency loss for physical state
            consistency_loss = F.mse_loss(z_physical, z_physical.mean(dim=0).expand_as(z_physical))
            
            # Total loss
            loss = recon_loss + beta * (kl_physical + kl_physio) + 0.5 * consistency_loss
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(measurement_ids)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_loss:.4f}")
        
        # Validation
        val_loss = validate_model(model, val_dataset, device)
        print(f"Validation Loss: {val_loss:.4f}")
        
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_DisentangledPulseVAE2.pth')
            print(f"Saved best model at epoch {epoch+1} with validation loss: {val_loss:.4f}")

def calculate_loss(model, dataset, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for measurement_id in dataset.measurement_indices.keys():
            pulses = dataset.get_measurement(measurement_id)
            x = torch.stack(pulses).to(device)
            
            x_recon, mu_physical, logvar_physical, mu_physio, logvar_physio, z_physical, z_physio = model(x)
            
            recon_loss = F.mse_loss(x_recon, x, reduction='sum') / x.size(0)
            kl_physical = -0.5 * torch.sum(1 + logvar_physical - mu_physical.pow(2) - logvar_physical.exp()) / x.size(0)
            kl_physio = -0.5 * torch.sum(1 + logvar_physio - mu_physio.pow(2) - logvar_physio.exp()) / x.size(0)
            consistency_loss = F.mse_loss(z_physical, z_physical.mean(dim=0).expand_as(z_physical))
            
            loss = recon_loss + 0.5* (kl_physical + kl_physio) + 0.5 * consistency_loss
            total_loss += loss.item()
    
    return total_loss / len(dataset.measurement_indices)

def validate_model(model, val_dataset, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for measurement_id in val_dataset.measurement_indices.keys():
            pulses = val_dataset.get_measurement(measurement_id)
            x = torch.stack(pulses).to(device)
            
            x_recon, mu_physical, logvar_physical, mu_physio, logvar_physio, z_physical, z_physio = model(x)
            
            recon_loss = F.mse_loss(x_recon, x, reduction='sum') / x.size(0)
            kl_physical = -0.5 * torch.sum(1 + logvar_physical - mu_physical.pow(2) - logvar_physical.exp()) / x.size(0)
            kl_physio = -0.5 * torch.sum(1 + logvar_physio - mu_physio.pow(2) - logvar_physio.exp()) / x.size(0)
            consistency_loss = F.mse_loss(z_physical, z_physical.mean(dim=0).expand_as(z_physical))
            
            loss = recon_loss + (kl_physical + kl_physio) + 0.1 * consistency_loss
            total_loss += loss.item()
    
    return total_loss / len(val_dataset.measurement_indices)

# def train_model(model, train_dataset, val_dataset, num_epochs, device):
#     optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
#     scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

#     best_val_loss = float('inf')
#     for epoch in range(num_epochs):
#         model.train()
#         total_loss = 0
#         measurement_ids = list(train_dataset.measurement_indices.keys())
#         random.shuffle(measurement_ids)

#         for measurement_id in tqdm(measurement_ids, desc=f"Epoch {epoch+1}/{num_epochs}"):
#             optimizer.zero_grad()
            
#             # 獲取同一測量的所有脈衲
#             pulses = train_dataset.get_measurement(measurement_id)
#             x = torch.stack(pulses).to(device)
            
#             x_recon, mu_physical, logvar_physical, mu_physio, logvar_physio, z_physical, z_physio = model(x)
            
#             # Reconstruction loss
#             recon_loss = F.mse_loss(x_recon, x, reduction='sum') / x.size(0)
            
#             # KL divergence loss        
#             kl_physical = -0.5 * torch.sum(1 + logvar_physical - mu_physical.pow(2) - logvar_physical.exp()) / x.size(0)
#             kl_physio = -0.5 * torch.sum(1 + logvar_physio - mu_physio.pow(2) - logvar_physio.exp()) / x.size(0)
            
#             # Adversarial loss
#             real_physical = torch.randn(x.size(0), model.latent_dim_physical).to(device)
#             real_physio = torch.randn(x.size(0), model.latent_dim_physio).to(device)
            
#             d_loss_physical = F.binary_cross_entropy_with_logits(model.discriminator_physical(z_physical.detach()), torch.zeros(x.size(0)).to(device)) + \
#                               F.binary_cross_entropy_with_logits(model.discriminator_physical(real_physical), torch.ones(x.size(0)).to(device))
#             d_loss_physio = F.binary_cross_entropy_with_logits(model.discriminator_physio(z_physio.detach()), torch.zeros(x.size(0)).to(device)) + \
#                             F.binary_cross_entropy_with_logits(model.discriminator_physio(real_physio), torch.ones(x.size(0)).to(device))
            
#             g_loss_physical = F.binary_cross_entropy_with_logits(model.discriminator_physical(z_physical), torch.ones(x.size(0)).to(device))
#             g_loss_physio = F.binary_cross_entropy_with_logits(model.discriminator_physio(z_physio), torch.ones(x.size(0)).to(device))
            
#             # Consistency loss for physical state
#             consistency_loss = F.mse_loss(z_physical, z_physical.mean(dim=0).expand_as(z_physical))
            
#             # Total loss
#             loss = (
#                 recon_loss + 
#                 0.1 * (kl_physical + kl_physio) + 
#                 0.01 * (d_loss_physical + d_loss_physio + g_loss_physical + g_loss_physio) + 
#                 0.1 * consistency_loss
#             )
            
#             loss.backward()
#             optimizer.step()
            
#             total_loss += loss.item()
        
#         avg_loss = total_loss / len(measurement_ids)
#         print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_loss:.4f}")
        
#         # Validation
#         val_loss = validate_model(model, val_dataset, device)
#         print(f"Validation Loss: {val_loss:.4f}")
        
#         scheduler.step(val_loss)

#         # 如果validation loss有改善,就保存模型
#         if val_loss < best_val_loss * 0.95:
#             best_val_loss = val_loss
#             torch.save(model.state_dict(), 'model_final.pth')
#             print(f"Saved best model at epoch {epoch+1} with validation loss: {val_loss:.4f}")

#     # 保存最終模型
#     # torch.save(model.state_dict(), 'model_final.pth')

# def validate_model(model, val_dataset, device):
#     model.eval()
#     total_loss = 0
#     with torch.no_grad():
#         for measurement_id in val_dataset.measurement_indices.keys():
#             pulses = val_dataset.get_measurement(measurement_id)
#             x = torch.stack(pulses).to(device)
            
#             x_recon, _, _, _, _, z_physical, _ = model(x)
            
#             recon_loss = F.mse_loss(x_recon, x, reduction='sum') / x.size(0)
#             consistency_loss = F.mse_loss(z_physical, z_physical.mean(dim=0).expand_as(z_physical))
            
#             loss = recon_loss + 0.1 * consistency_loss
#             total_loss += loss.item()
    
#     return total_loss / len(val_dataset.measurement_indices)


def get_trained_model(model_path = 'best_DisentangledPulseVAE.pth', device = 'cpu'):
    input_dim = 100  # 假设每个脉冲有100个采样点
    latent_dim_physical = 5#10
    latent_dim_physio = 15#20
    model = DisentangledPulseVAE(input_dim, latent_dim_physical, latent_dim_physio).to(device)
    model.load_state_dict(torch.load(model_path))
    return model

def encode_pulse(model, pulse, device='cpu'):
    """
    將單個脈衝波形編碼為潛在向量。

    Args:
        model (DisentangledPulseVAE): 訓練好的模型
        pulse (numpy.ndarray or list): 輸入脈衝波形
        device (str): 使用的設備 ('cpu' 或 'cuda')

    Returns:
        tuple: 包含物理和生理潛在向量的元組,每個都是numpy數組
    """
    model.eval()
    model.to(device)
    with torch.no_grad():
        x = torch.FloatTensor(pulse).unsqueeze(0).to(device)  # 添加批次維度
        z_physical, z_physio = model.encode(x)
    return z_physical.cpu().numpy()[0], z_physio.cpu().numpy()[0]

def decode_latent(model, z_physical, z_physio, device='cpu'):
    """
    將潛在向量解碼回脈衝波形。

    Args:
        model (DisentangledPulseVAE): 訓練好的模型
        z_physical (numpy.ndarray): 物理潛在向量
        z_physio (numpy.ndarray): 生理潛在向量
        device (str): 使用的設備 ('cpu' 或 'cuda')

    Returns:
        numpy.ndarray: 重建的脈衝波形
    """
    model.eval()
    model.to(device)
    with torch.no_grad():
        z_physical = torch.FloatTensor(z_physical).unsqueeze(0).to(device)
        z_physio = torch.FloatTensor(z_physio).unsqueeze(0).to(device)
        reconstructed_pulse = model.decode(z_physical, z_physio)
    return reconstructed_pulse.cpu().numpy()[0]

def predict_corrected_reconstructed_signal(signal, sample_rate, peaks, target_len=100, device='cpu'):
    """
    預測修正後的重建信號。

    Args:
        model (DisentangledPulseVAE): 訓練好的模型
        signal (numpy.ndarray): 輸入信號
        sample_rate (int): 採樣率
        peaks (list): 峰值位置列表
        target_len (int): 目標脈衝長度
        device (str): 使用的設備 ('cpu' 或 'cuda')

    Returns:
        numpy.ndarray: 修正後的重建信號
    """
    model = get_trained_model()
    model.eval()
    model.to(device)

    # 複製原始信號
    signal = np.array(signal)
    reconstructed_signal = np.copy(signal)

    # 重採樣信號（如果需要）
    if sample_rate != 100:
        resampled_signal = scipy.signal.resample(signal, int(len(signal) * 100 / sample_rate))
        resampled_peaks = [int(p * 100 / sample_rate) for p in peaks]
    else:
        resampled_signal = np.array(signal)
        resampled_peaks = peaks

    # 標準化信號
    mean = np.mean(resampled_signal)
    std = np.std(resampled_signal)
    print(f"Mean: {mean}, type: {type(mean)}")
    print(f"Std: {std}, type: {type(std)}")
    normalized_signal = (resampled_signal - mean) / std

    # 處理每個脈衝
    for i in range(len(resampled_peaks) - 1):
        start_idx = resampled_peaks[i]
        end_idx = resampled_peaks[i + 1]
        pulse = normalized_signal[start_idx:end_idx]

        # 將脈衝重採樣到目標長度
        x = np.linspace(0, 1, len(pulse))
        f = interp1d(x, pulse, kind='linear')
        pulse_resampled = f(np.linspace(0, 1, target_len))

        # 編碼脈衝
        z_physical, z_physio = encode_pulse(model, pulse_resampled, device)

        # 將物理潛在向量設為0
        z_physical[:] = 0

        # 解碼修正後的潛在向量
        corrected_pulse = decode_latent(model, z_physical, z_physio, device)

        # 將修正後的脈衝重採樣回原始長度
        x_new = np.linspace(0, 1, len(pulse))
        f_new = interp1d(np.linspace(0, 1, target_len), corrected_pulse, kind='linear')
        corrected_pulse_original_length = f_new(x_new)

        # 將修正後的脈衝放回重建信號中
        resampled_signal[start_idx:end_idx] = corrected_pulse_original_length

    # 反標準化
    reconstructed_resampled = np.multiply(resampled_signal, std) + mean

    # 如果原始採樣率不是100Hz，將信號重採樣回原始採樣率
    if sample_rate != 100:
        reconstructed_signal = scipy.signal.resample(reconstructed_resampled, len(signal))
    else:
        reconstructed_signal = reconstructed_resampled

    return reconstructed_signal

def predict_reconstructed_signal(signal, sample_rate, peaks, target_len=100, device='cpu'):
    model = get_trained_model()
    model.eval()
    model.to(device)

    # 複製原始信號
    signal = np.array(signal)
    reconstructed_signal = np.copy(signal)

    # 重採樣信號（如果需要）
    if sample_rate != 100:
        resampled_signal = scipy.signal.resample(signal, int(len(signal) * 100 / sample_rate))
        resampled_peaks = [int(p * 100 / sample_rate) for p in peaks]
    else:
        resampled_signal = np.array(signal)
        resampled_peaks = peaks

    # 標準化信號
    mean = np.mean(resampled_signal)
    std = np.std(resampled_signal)
    print(f"Mean: {mean}, type: {type(mean)}")
    print(f"Std: {std}, type: {type(std)}")
    normalized_signal = (resampled_signal - mean) / std

    # 處理每個脈衝
    for i in range(len(resampled_peaks) - 1):
        start_idx = resampled_peaks[i]
        end_idx = resampled_peaks[i + 1]
        pulse = normalized_signal[start_idx:end_idx]

        # 將脈衝重採樣到目標長度
        x = np.linspace(0, 1, len(pulse))
        f = interp1d(x, pulse, kind='linear')
        pulse_resampled = f(np.linspace(0, 1, target_len))

        # 編碼脈衝
        z_physical, z_physio = encode_pulse(model, pulse_resampled, device)

        # 解碼修正後的潛在向量
        corrected_pulse = decode_latent(model, z_physical, z_physio, device)

        # 將修正後的脈衝重採樣回原始長度
        x_new = np.linspace(0, 1, len(pulse))
        f_new = interp1d(np.linspace(0, 1, target_len), corrected_pulse, kind='linear')
        corrected_pulse_original_length = f_new(x_new)

        # 將修正後的脈衝放回重建信號中
        resampled_signal[start_idx:end_idx] = corrected_pulse_original_length

    # 反標準化
    reconstructed_resampled = np.multiply(resampled_signal, std) + mean

    # 如果原始採樣率不是100Hz，將信號重採樣回原始採樣率
    if sample_rate != 100:
        reconstructed_signal = scipy.signal.resample(reconstructed_resampled, len(signal))
    else:
        reconstructed_signal = reconstructed_resampled

    return reconstructed_signal


def split_json_files(json_files, train_ratio=0.9):
    random.shuffle(json_files)
    split_point = int(len(json_files) * train_ratio)
    return json_files[:split_point], json_files[split_point:]

def main():
    input_dim = 100  # 假设每个脉冲有100个采样点
    latent_dim_physical = 5#10
    latent_dim_physio = 15#20
    data_folder = 'wearing_consistency'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'torch device: {device}')

    json_files = get_json_files(data_folder)
    train_files, val_files = split_json_files(json_files)
    print(f'number of train files: {len(train_files)}, number of val files: {len(val_files)}')
    train_dataset  = MeasurementPulseDataset(train_files, input_dim)
    val_dataset = MeasurementPulseDataset(val_files, input_dim)
    print(f'train_dataset size: {len(train_dataset )}, val_dataset size: {len(val_dataset)}')
    model = DisentangledPulseVAE(input_dim, latent_dim_physical, latent_dim_physio).to(device)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total number of model parameters: {trainable_params}, model:{model}') 
    # if os.path.exists('model_final.pth'):
    #     model.load_state_dict(torch.load('model_final.pth'))
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    train_model(model, train_dataset, val_dataset, num_epochs=2000, device=device)

if __name__ == "__main__":
    main()