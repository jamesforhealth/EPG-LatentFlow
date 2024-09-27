'''
Baseline model:
    target_len = 100
    model = EPGBaselinePulseAutoencoder(target_len).to(device)

'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from preprocessing import get_json_files, split_json_files, MeasurementPulseDataset
from torch.utils.data import Dataset, DataLoader, random_split



class BetaTCVAE(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, beta: float = 4.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.beta = beta

        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )

        # 使用新的类型参数语法定义类型别名
        type Tensor = torch.Tensor

        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)

        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
            nn.Sigmoid(),  # 假设输入在 [0,1] 范围内
        )

    def encode(self, x: Tensor) -> tuple[Tensor, Tensor]:
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: Tensor) -> Tensor:
        return self.decoder(z)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar, z

    def compute_loss(
        self, recon_x: Tensor, x: Tensor,
        mu: Tensor, logvar: Tensor, z: Tensor,
        batch_data: Tensor
    ) -> Tensor:
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')

        # KL 散度
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        # Total Correlation (TC) 的估计
        # 计算 q(z) 和 p(z) 的对数概率密度
        log_qz = self.log_density_gaussian(z, mu, logvar)
        log_pz = self.log_density_gaussian(z, torch.zeros_like(mu), torch.zeros_like(logvar))
        # TC = KL(q(z)||p(z))
        tc_loss = (log_qz - log_pz).sum()

        # 一致性损失
        z_physical = z[:, :self.latent_dim // 2]
        consistency_loss = F.mse_loss(
            z_physical,
            z_physical.mean(dim=0, keepdim=True),
            reduction='sum'
        )

        total_loss = recon_loss + self.beta * (kld + tc_loss) + consistency_loss
        return total_loss

    def log_density_gaussian(self, x: Tensor, mu: Tensor, logvar: Tensor) -> Tensor:
        # 计算高斯对数概率密度
        c = torch.log(torch.tensor(2 * torch.pi))
        var = torch.exp(logvar)
        log_density = -0.5 * (c + logvar + (x - mu).pow(2) / var)
        return log_density.sum(dim=1)
    

def train_model(
    model: BetaTCVAE, data_loader: torch.utils.data.DataLoader,
    device: torch.device, epochs: int = 50
):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        model.train()
        train_loss = 0

        for batch_data in data_loader:
            batch_data = batch_data.to(device)
            optimizer.zero_grad()

            recon_batch, mu, logvar, z = model(batch_data)
            loss = model.compute_loss(recon_batch, batch_data, mu, logvar, z, batch_data)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

        avg_loss = train_loss / len(data_loader.dataset)
        print(
            f"""Epoch {epoch + 1}, Average Loss: {avg_loss:.6f}"""
        )

    torch.save(model.state_dict(), 'beta_tcv_ae_pulse_disentangled_representation.pth')


def main():
    data_folder = 'wear_consistency'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'torch device: {device}')
    target_len = 100
    batch_size = 32
    json_files = get_json_files(data_folder)
    train_files, val_files = split_json_files(json_files)
    train_dataset = MeasurementPulseDataset(train_files, target_len)
    val_dataset = MeasurementPulseDataset(val_files, target_len)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = BetaTCVAE(input_dim=target_len, latent_dim=10)
    train_model(model, train_dataloader, device)

if __name__ == '__main__':
    main()
