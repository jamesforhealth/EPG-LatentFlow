'''
Assume that wearing status is independent of changes in physiological state
ideal latent spcae S = W (+) P
    , (+) means direct sum
    , W menas the subspace representing changes in wearing state
    , P menas the subspace representing changes in physiological state
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


class PulseEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim * 2)
        )
    
    def forward(self, x):
        params = self.encoder(x)
        mu, logvar = params.chunk(2, dim=-1)
        return mu, logvar

class SequenceEncoder(nn.Module):
    def __init__(self, pulse_latent_dim, seq_latent_dim):
        super().__init__()
        self.lstm = nn.LSTM(pulse_latent_dim, 128, batch_first=True)
        self.fc = nn.Linear(128, seq_latent_dim * 2)
    
    def forward(self, x):
        _, (h, _) = self.lstm(x)
        params = self.fc(h.squeeze(0))
        mu, logvar = params.chunk(2, dim=-1)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, physio_dim, wear_dim, output_dim):
        super().__init__()
        self.fc = nn.Linear(physio_dim + wear_dim, 128)
        self.lstm = nn.LSTM(128, 128, batch_first=True)
        self.output = nn.Linear(128, output_dim)
    
    def forward(self, z_physio, z_wear, seq_len):
        z = torch.cat([z_physio, z_wear], dim=-1)
        h = self.fc(z).unsqueeze(1).repeat(1, seq_len, 1)
        output, _ = self.lstm(h)
        return self.output(output)

class HVAE(nn.Module):
    def __init__(self, input_dim, pulse_latent_dim, physio_dim, wear_dim):
        super().__init__()
        self.pulse_encoder = PulseEncoder(input_dim, pulse_latent_dim)
        self.seq_encoder = SequenceEncoder(pulse_latent_dim, physio_dim + wear_dim)
        self.decoder = Decoder(physio_dim, wear_dim, input_dim)
        self.physio_dim = physio_dim
        self.wear_dim = wear_dim
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        
        # Encode pulses
        pulse_mu, pulse_logvar = zip(*[self.pulse_encoder(x[:, i]) for i in range(seq_len)])
        pulse_mu = torch.stack(pulse_mu, dim=1)
        pulse_logvar = torch.stack(pulse_logvar, dim=1)
        pulse_z = self.reparameterize(pulse_mu, pulse_logvar)
        
        # Encode sequence
        seq_mu, seq_logvar = self.seq_encoder(pulse_z)
        seq_z = self.reparameterize(seq_mu, seq_logvar)
        
        # Split into physiological and wearable factors
        z_physio, z_wear = seq_z.split([self.physio_dim, self.wear_dim], dim=-1)
        
        # Decode
        recon_x = self.decoder(z_physio, z_wear, seq_len)
        
        return recon_x, pulse_mu, pulse_logvar, seq_mu, seq_logvar, z_physio, z_wear
    
def train_hvae(model, train_loader, num_epochs, device):
    optimizer = torch.optim.Adam(model.parameters())
    
    for epoch in range(num_epochs):
        for batch in train_loader:
            x = batch.to(device)
            recon_x, pulse_mu, pulse_logvar, seq_mu, seq_logvar, z_physio, z_wear = model(x)
            
            # Reconstruction loss
            recon_loss = F.mse_loss(recon_x, x)
            
            # KL divergence
            pulse_kl = -0.5 * torch.sum(1 + pulse_logvar - pulse_mu.pow(2) - pulse_logvar.exp())
            seq_kl = -0.5 * torch.sum(1 + seq_logvar - seq_mu.pow(2) - seq_logvar.exp())
            
            # Physiological consistency loss
            physio_consistency = F.mse_loss(z_physio[:, None, :].expand(-1, x.size(1), -1), 
                                            z_physio[:, None, :].expand(-1, x.size(1), -1))
            
            # Wearable diversity loss
            wear_diversity = -F.mse_loss(z_wear[:, None, :].expand(-1, x.size(1), -1), 
                                         z_wear[:, None, :].expand(-1, x.size(1), -1))
            
            # Total loss
            loss = recon_loss + 0.1 * (pulse_kl + seq_kl) + 0.01 * physio_consistency + 0.01 * wear_diversity
            # 在训练循环中添加这个损失
            cycle_loss = cycle_consistency_loss(model, x, z_physio, z_wear)
            loss += 0.1 * cycle_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")

def orthogonality_loss(z_physio, z_wear):
    z_physio_norm = F.normalize(z_physio, dim=-1)
    z_wear_norm = F.normalize(z_wear, dim=-1)
    return torch.abs(torch.sum(z_physio_norm * z_wear_norm, dim=-1)).mean()

def cycle_consistency_loss(model, x, z_physio, z_wear):
    # 重构输入
    recon_x = model.decoder(z_physio, z_wear, x.size(1))
    
    # 重新编码重构的输入
    _, _, _, _, _, z_physio_recon, z_wear_recon = model(recon_x)
    
    # 计算循环一致性损失
    physio_cycle_loss = F.mse_loss(z_physio, z_physio_recon)
    wear_cycle_loss = F.mse_loss(z_wear, z_wear_recon)
    
    return physio_cycle_loss + wear_cycle_loss

def train_hvae_staged(model, train_loader, num_epochs, device):
    optimizer = torch.optim.Adam(model.parameters())
    
    # Stage 1: Train physiological subspace
    for epoch in range(num_epochs // 2):
        for batch in train_loader:
            x = batch.to(device)
            recon_x, pulse_mu, pulse_logvar, seq_mu, seq_logvar, z_physio, _ = model(x)
            
            # Only use physiological component for reconstruction
            recon_x = model.decoder(z_physio, torch.zeros_like(z_physio), x.size(1))
            
            # Compute losses
            recon_loss = F.mse_loss(recon_x, x)
            pulse_kl = -0.5 * torch.sum(1 + pulse_logvar - pulse_mu.pow(2) - pulse_logvar.exp())
            seq_kl = -0.5 * torch.sum(1 + seq_logvar - seq_mu.pow(2) - seq_logvar.exp())
            
            loss = recon_loss + 0.1 * (pulse_kl + seq_kl)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    # Stage 2: Train wearable subspace
    for epoch in range(num_epochs // 2, num_epochs):
        for batch in train_loader:
            x = batch.to(device)
            recon_x, pulse_mu, pulse_logvar, seq_mu, seq_logvar, z_physio, z_wear = model(x)
            
            # Compute losses
            recon_loss = F.mse_loss(recon_x, x)
            pulse_kl = -0.5 * torch.sum(1 + pulse_logvar - pulse_mu.pow(2) - pulse_logvar.exp())
            seq_kl = -0.5 * torch.sum(1 + seq_logvar - seq_mu.pow(2) - seq_logvar.exp())
            ortho_loss = orthogonality_loss(z_physio, z_wear)
            cycle_loss = cycle_consistency_loss(model, x, z_physio, z_wear)
            
            loss = recon_loss + 0.1 * (pulse_kl + seq_kl) + 0.1 * ortho_loss + 0.1 * cycle_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")


def main():
    data_folder = 'labeled_DB'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'torch device: {device}')
    json_files = get_json_files(data_folder)  # 实现一个函数来获取所有的JSON文件路径

    # 設置參數
    batch_size = 32
    lr = 1e-4
    target_len = 200

    # 加載並劃分數據集
    dataset = PulseDataset(json_files, target_len)
    # dataset = PulseDataset(json_files)
    train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)


if __name__ == '__main__':
    main()