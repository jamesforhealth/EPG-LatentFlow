from tqdm import tqdm
import math
from sklearn.model_selection import train_test_split
import h5py
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import random

import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from preprocessing import process_DB_rawdata, get_json_files, add_noise_with_snr, add_gaussian_noise_torch, MeasurementPulseDataset, PulseDataset

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

def train_model(model, train_loader, num_epochs, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            
            x = batch.to(device)  # 假设 batch 已经是正确的张量形状
            
            x_recon, mu_physical, logvar_physical, mu_physio, logvar_physio, z_physical, z_physio = model(x)
            
            # Reconstruction loss
            recon_loss = F.mse_loss(x_recon, x, reduction='sum')
            
            # KL divergence loss
            kl_physical = -0.5 * torch.sum(1 + logvar_physical - mu_physical.pow(2) - logvar_physical.exp())
            kl_physio = -0.5 * torch.sum(1 + logvar_physio - mu_physio.pow(2) - logvar_physio.exp())
            
            # Adversarial loss
            real_physical = torch.randn(x.size(0), model.latent_dim_physical).to(device)
            real_physio = torch.randn(x.size(0), model.latent_dim_physio).to(device)
            
            d_loss_physical = F.binary_cross_entropy_with_logits(model.discriminator_physical(z_physical), torch.zeros(x.size(0)).to(device)) + \
                              F.binary_cross_entropy_with_logits(model.discriminator_physical(real_physical), torch.ones(x.size(0)).to(device))
            d_loss_physio = F.binary_cross_entropy_with_logits(model.discriminator_physio(z_physio), torch.zeros(x.size(0)).to(device)) + \
                            F.binary_cross_entropy_with_logits(model.discriminator_physio(real_physio), torch.ones(x.size(0)).to(device))
            
            g_loss_physical = F.binary_cross_entropy_with_logits(model.discriminator_physical(z_physical), torch.ones(x.size(0)).to(device))
            g_loss_physio = F.binary_cross_entropy_with_logits(model.discriminator_physio(z_physio), torch.ones(x.size(0)).to(device))
            
            # Consistency loss for physical state
            consistency_loss = F.mse_loss(z_physical, z_physical.mean(dim=0).expand_as(z_physical))
            
            # Total loss
            loss = recon_loss + kl_physical + kl_physio + d_loss_physical + d_loss_physio + g_loss_physical + g_loss_physio + consistency_loss
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader):.4f}")



def split_json_files(json_files, train_ratio=0.9):
    random.shuffle(json_files)
    split_point = int(len(json_files) * train_ratio)
    return json_files[:split_point], json_files[split_point:]

def main():
    input_dim = 100  # 假设每个脉冲有100个采样点
    latent_dim_physical = 10
    latent_dim_physio = 20
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
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    train_model(model, train_loader, num_epochs=100, device=device)

if __name__ == "__main__":
    main()