import os
import json
import random
import numpy as np
import scipy.signal
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from preprocessing import process_DB_rawdata, PulseDataset
import math
from model_find_peaks import detect_peaks_from_signal
import csv
def save_pulses_to_csv(dataset, csv_filename):
    with open(csv_filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        
        for pulse in dataset:
            # 將張量轉換為列表並寫入CSV
            # input(f'pulse: {pulse}, len(pulse): {len(pulse)}')
            if len(pulse) < 30:
                input(f'pulse: {pulse}, len(pulse): {len(pulse)}')
                continue
            csvwriter.writerow(pulse.tolist())

    print(f"Saved {len(dataset)} pulses to {csv_filename}")




class WearingDataset(Dataset):
    def __init__(self, json_files, sample_rate=100):
        self.data = []
        self.sample_rate = sample_rate
        self.load_data(json_files)

    def load_data(self, json_files):
        all_signals = []
        x_points_list = []

        for json_file in json_files:
            with open(json_file, 'r') as f:
                json_data = json.load(f)
                if json_data['anomaly_list'] != []:
                    continue
                signal = json_data['smoothed_data']
                original_sample_rate = json_data.get('sample_rate', 100)
                x_points = json_data['x_points']

                # 确保 signal 是一个数组
                signal = np.array(signal, dtype=np.float32)
                if signal.ndim == 0:
                    print(f"Warning: signal in {json_file} is a scalar, skipping")
                    continue

                if original_sample_rate != self.sample_rate:
                    num_samples = int(len(signal) * self.sample_rate / original_sample_rate)
                    signal = scipy.signal.resample(signal, num_samples)
                    x_points = [int(x * self.sample_rate / original_sample_rate) for x in x_points]

                all_signals.append(signal)
                x_points_list.append(x_points)

        # 全局标准化
        all_signals = np.concatenate(all_signals)
        mean = np.mean(all_signals)
        std = np.std(all_signals)
        all_signals = (all_signals - mean) / std

        # 分割脉冲信号
        signal_index = 0
        for signal, x_points in zip(all_signals, x_points_list):
            signal_len = len(signal)
            for i in range(len(x_points) - 1):
                start_idx = x_points[i]
                end_idx = x_points[i + 1]

                if start_idx >= signal_len or end_idx > signal_len:
                    print(f"Skipping invalid indices in {json_file}")
                    continue

                pulse = signal[start_idx:end_idx]
                if np.isnan(pulse).any() or np.isinf(pulse).any():
                    print(f"Warning: NaN or Inf values found in {json_file}")
                    continue
                self.data.append(torch.tensor(pulse))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(batch):
    padded_batch = pad_sequence(batch, batch_first=True)
    return padded_batch

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

class TransformerAutoencoder(nn.Module):
    def __init__(self, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward=2048, dropout=0.1):
        super(TransformerAutoencoder, self).__init__()
        self.d_model = d_model
        self.encoder = nn.Linear(1, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.transformer = nn.Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout)
        self.decoder = nn.Linear(d_model, 1)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        # print(f'src shape: {src.shape}')
        src = self.encoder(src.unsqueeze(-1)) * math.sqrt(self.d_model)
        # print(f'linear shape: {src.shape}')
        src = self.pos_encoder(src)
        # print(f'pos_encoder shape: {src.shape}')
        src = src.permute(1, 0, 2)  # Transformer expects (seq_len, batch_size, d_model)
        # print(f'permute shape: {src.shape}')
        latent = self.transformer.encoder(src)
        # print(f'latent shape: {latent.shape}')
        output = self.transformer.decoder(latent, latent)
        # output = self.transformer(src, src)
        # print(f'transformer decode shape: {output.shape}')
        output = output.permute(1, 0, 2)  # Convert back to (batch_size, seq_len, d_model)
        # print(f'permute shape: {output.shape}')
        output = self.decoder(output).squeeze(-1)
        # input(f'decoder shape: {output.shape}')
        return output

class TransformerAutoencoder2(nn.Module):
    def __init__(self, d_model, nhead, num_encoder_layers, num_decoder_layers, latent_dim, lstm_hidden_dim, lstm_num_layers, dim_feedforward=2048, dropout=0.0):
        super(TransformerAutoencoder2, self).__init__()
        self.d_model = d_model
        self.encoder = nn.Linear(1, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.transformer = nn.Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout)
        self.decoder = nn.Linear(d_model, 1)
        self.lstm_encoder = nn.LSTM(d_model, lstm_hidden_dim, lstm_num_layers, batch_first=True, bidirectional=True)
        self.lstm_decoder = nn.LSTM(lstm_hidden_dim * 2, d_model, lstm_num_layers, batch_first=True)  # Adjust input dim
        self.to_latent = nn.Linear(lstm_hidden_dim * 2, latent_dim)  # Adjust input dim
        self.from_latent = nn.Linear(latent_dim, lstm_hidden_dim * 2)  # Adjust output dim

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        # print(f'src shape: {src.shape}')
        src = self.encoder(src.unsqueeze(-1)) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        src = src.permute(1, 0, 2)  # Transformer expects (seq_len, batch_size, d_model)
        latent = self.transformer.encoder(src)
        # print(f'latent shape: {latent.shape}')

        # LSTM Autoencoder
        latent = latent.permute(1, 0, 2)  # Convert to (batch_size, seq_len, d_model)
        _, (hidden, _) = self.lstm_encoder(latent)
        hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)  # Concatenate the outputs of the two directions
        latent_vector = self.to_latent(hidden)
        # print(f'latent_vector shape: {latent_vector.shape}')

        # Decoding with LSTM
        decoder_input = self.from_latent(latent_vector).unsqueeze(1).repeat(1, latent.size(1), 1)
        decoded_latent, _ = self.lstm_decoder(decoder_input)
        decoded_latent = decoded_latent.permute(1, 0, 2)  # Convert back to (seq_len, batch_size, d_model)

        output = self.transformer.decoder(decoded_latent, decoded_latent)
        output = output.permute(1, 0, 2)  # Convert back to (batch_size, seq_len, d_model)
        output = self.decoder(output).squeeze(-1)
        # print(f'decoder shape: {output.shape}')
        return output

class PulseEncoder(nn.Module):
    def __init__(self, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward=2048, dropout=0.1):
        super(PulseEncoder, self).__init__()
        self.d_model = d_model
        self.encoder = nn.Linear(1, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.transformer = nn.Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout)

    def forward(self, src):
        # print(f'PulseEncoder src shape: {src.shape}')
        src = self.encoder(src.unsqueeze(-1)) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        src = src.permute(1, 0, 2)  # Transformer expects (seq_len, batch_size, d_model)
        latent = self.transformer.encoder(src)
        return latent
        
class PulseDecoder(nn.Module):
    def __init__(self, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward=2048, dropout=0.1):
        super(PulseDecoder, self).__init__()
        self.transformer = nn.Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout)
        self.decoder = nn.Linear(d_model, 1)

    def forward(self, decoded_latent, memory):
        output = self.transformer.decoder(decoded_latent, memory)
        output = output.permute(1, 0, 2)  # Convert back to (batch_size, seq_len, d_model)
        output = self.decoder(output).squeeze(-1)
        return output
    

class TransformerAutoencoder3(nn.Module): # Very good for reducing loss, but no clear latent vector
    def __init__(self, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward=2048, dropout=0.1):
        super(TransformerAutoencoder3, self).__init__()
        self.encoder = PulseEncoder(d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout)
        self.decoder = PulseDecoder(d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout)

    def forward(self, src):
        memory = self.encoder(src)
        output = self.decoder(memory, memory)
        return output    

# class TransformerAutoencoder4(nn.Module):
#     def __init__(self, d_model, nhead, num_encoder_layers, num_decoder_layers, latent_dim, lstm_hidden_dim, lstm_num_layers, dim_feedforward=2048, dropout=0.1):
#         super(TransformerAutoencoder4, self).__init__()
#         self.encoder = PulseEncoder(d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout)
#         self.decoder = PulseDecoder(d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout)
        
#         self.lstm_encoder = nn.LSTM(d_model, lstm_hidden_dim, lstm_num_layers, batch_first=True, bidirectional=True)
#         self.lstm_decoder = nn.LSTM(lstm_hidden_dim * 2, d_model, lstm_num_layers, batch_first=True)
#         self.to_latent = nn.Linear(lstm_hidden_dim * 2, latent_dim)
#         self.from_latent = nn.Linear(latent_dim, lstm_hidden_dim * 2)

#     def forward(self, src):
#         memory = self.encoder(src)
#         memory = memory.permute(1, 0, 2)

#         # LSTM Autoencoder
#         _, (hidden, _) = self.lstm_encoder(memory)
#         hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
#         latent_vector = self.to_latent(hidden)

#         # Decoding with LSTM
#         decoder_input = self.from_latent(latent_vector).unsqueeze(1).repeat(1, memory.size(1), 1)
#         decoded_latent, _ = self.lstm_decoder(decoder_input)
#         decoded_latent = decoded_latent.permute(1, 0, 2)
#         output = self.decoder(decoded_latent, memory.permute(1, 0, 2))
#         return output 


class TransformerAutoencoder4(nn.Module):
    def __init__(self, d_model, nhead, num_encoder_layers, num_decoder_layers, latent_dim, lstm_hidden_dim, lstm_num_layers, dim_feedforward=2048, dropout=0.1):
        super(TransformerAutoencoder4, self).__init__()
        self.encoder = PulseEncoder(d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout)
        self.decoder = PulseDecoder(d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout)

        self.lstm_encoder = nn.LSTM(d_model, lstm_hidden_dim, lstm_num_layers, batch_first=True, bidirectional=True)
        self.lstm_decoder = nn.LSTM(lstm_hidden_dim * 2, d_model, lstm_num_layers, batch_first=True)
        self.to_latent = nn.Linear(lstm_hidden_dim * 2, latent_dim)
        self.from_latent = nn.Linear(latent_dim, lstm_hidden_dim * 2)

    def forward(self, src):
        # Encoder Part
        memory = self.encoder(src)
        # input(f'memory shape: {memory.shape}')
        memory = memory.permute(1, 0, 2)  # [batch_size, seq_len, d_model]
        # input(f'memory shape: {memory.shape}')
        # LSTM Encoder
        _, (hidden, _) = self.lstm_encoder(memory)
        # input(f'hidden shape: {hidden.shape}')
        hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)  # Concatenate the last two hidden states from the bidirectional LSTM
        # input(f'hidden shape: {hidden.shape}')
        latent_vector = self.to_latent(hidden)  # [batch_size, latent_dim]
        # print(f'latent_vector shape: {latent_vector.shape}')
        # LSTM Decoder
        decoder_input = self.from_latent(latent_vector).unsqueeze(1).repeat(1, memory.size(1), 1)  # [batch_size, seq_len, lstm_hidden_dim * 2]
        decoded_latent, _ = self.lstm_decoder(decoder_input)
        decoded_latent = decoded_latent.permute(1, 0, 2)  # [seq_len, batch_size, d_model]
        # print(f'decoded_latent shape: {decoded_latent.shape}')
        # Decoder Part
        output = self.decoder(decoded_latent, memory.permute(1, 0, 2))  # [seq_len, batch_size, d_model]
        # print(f'output shape: {output.shape}')
        return output  # [batch_size, seq_len]
    

class SelfAttention(nn.Module):
    def __init__(self, d_model, nhead):
        super(SelfAttention, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)

    def forward(self, x):
        attn_output, _ = self.self_attn(x, x, x)
        return attn_output
    

class TransformerAutoencoder5(nn.Module):#start to add attention in LSTM-autoencoder
    def __init__(self, d_model, nhead, num_encoder_layers, num_decoder_layers, latent_dim, lstm_hidden_dim, lstm_num_layers, dim_feedforward=2048, dropout=0.1):
        super(TransformerAutoencoder5, self).__init__()
        self.encoder = PulseEncoder(d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout)
        self.decoder = PulseDecoder(d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout)
        
        self.lstm_encoder = nn.LSTM(d_model, lstm_hidden_dim, lstm_num_layers, batch_first=True, bidirectional=True)
        self.lstm_decoder = nn.LSTM(lstm_hidden_dim * 2, d_model, lstm_num_layers, batch_first=True)
        
        self.self_attention = SelfAttention(d_model, nhead)

        self.to_latent = nn.Linear(lstm_hidden_dim * 2, latent_dim)
        self.from_latent = nn.Linear(latent_dim, lstm_hidden_dim * 2)

    def forward(self, src):
        memory = self.encoder(src)
        memory = memory.permute(1, 0, 2)

        # LSTM Autoencoder with Self-Attention
        lstm_output, (hidden, _) = self.lstm_encoder(memory)
        lstm_output = self.self_attention(lstm_output)
        hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        latent_vector = self.to_latent(hidden)

        # Decoding with LSTM
        decoder_input = self.from_latent(latent_vector).unsqueeze(1).repeat(1, memory.size(1), 1)
        decoded_latent, _ = self.lstm_decoder(decoder_input)
        decoded_latent = decoded_latent.permute(1, 0, 2)

        output = self.decoder(decoded_latent, memory.permute(1, 0, 2))
        return output    

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class DeconvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1):
        super(DeconvBlock, self).__init__()
        self.deconv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.deconv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class TransformerAutoencoder6(nn.Module):
    def __init__(self, d_model, nhead, num_encoder_layers, num_decoder_layers, latent_dim, rnn_hidden_dim, rnn_num_layers, dim_feedforward=2048, dropout=0.1):
        super(TransformerAutoencoder6, self).__init__()
        self.encoder = PulseEncoder(d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout)
        self.decoder = PulseDecoder(d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout)
        
        # 1D-CNN layers before and after LSTM
        self.conv_layers = nn.Sequential(
            ConvBlock(d_model, d_model * 2, kernel_size=3, padding=1),
            ConvBlock(d_model * 2, d_model * 4, kernel_size=3, padding=1),
            ConvBlock(d_model * 4, d_model * 8, kernel_size=3, padding=1),
            ConvBlock(d_model * 8, d_model * 16, kernel_size=3, padding=1)
        )

        self.deconv_layers = nn.Sequential(
            DeconvBlock(d_model * 16, d_model * 8, kernel_size=3, padding=1),
            DeconvBlock(d_model * 8, d_model * 4, kernel_size=3, padding=1),
            DeconvBlock(d_model * 4, d_model * 2, kernel_size=3, padding=1),
            DeconvBlock(d_model * 2, d_model, kernel_size=3, padding=1)
        )

        self.rnn_encoder = nn.LSTM(d_model * 16, rnn_hidden_dim, rnn_num_layers, batch_first=True)
        self.rnn_decoder = nn.LSTM(rnn_hidden_dim, d_model * 16, rnn_num_layers, batch_first=True)
        
        self.self_attention = SelfAttention(d_model, nhead)

        self.to_latent = nn.Linear(rnn_hidden_dim * 2, latent_dim)
        self.from_latent = nn.Linear(latent_dim, rnn_hidden_dim * 2)

    def forward(self, src):
        memory = self.encoder(src)
        memory = memory.permute(0, 2, 1)  # Change to (batch_size, d_model, seq_len)
        print(f'memory.permute.shape: {memory.shape}')
        memory = self.conv_layers(memory)  # Apply convolutional layers
        memory = memory.permute(0, 2, 1)  # Change back to (batch_size, seq_len, d_model * 16)
        print(f'memory.shape: {memory.shape}')

        # LSTM Autoencoder with Self-Attention
        rnn_output, (hidden, _) = self.rnn_encoder(memory)
        print(f'rnn_output.shape: {rnn_output.shape}')
        print(f'hidden.shape: {hidden.shape}')
        rnn_output = self.self_attention(rnn_output)
        print(f'rnn_output.shape: {rnn_output.shape}')
        print(f'hidden.shape: {hidden.shape}')

        # Concatenate the hidden states from all layers and directions
        hidden = hidden.permute(1, 0, 2).contiguous().view(rnn_output.size(0), -1)
        print(f'hidden.view.shape: {hidden.shape}')
        latent_vector = self.to_latent(hidden)
        print(f'latent_vector.shape: {latent_vector.shape}')

        # Decoding with LSTM
        decoder_input = self.from_latent(latent_vector).unsqueeze(1).repeat(1, memory.size(1), 1)
        print(f'decoder_input.shape: {decoder_input.shape}')
        decoded_latent, _ = self.rnn_decoder(decoder_input)
        decoded_latent = decoded_latent.permute(0, 2, 1)  # Change to (batch_size, d_model * 16, seq_len)
        print(f'decoded_latent.permute.shape: {decoded_latent.shape}')
        decoded_latent = self.deconv_layers(decoded_latent)  # Apply deconvolutional layers
        decoded_latent = decoded_latent.permute(0, 2, 1)  # Change to (batch_size, seq_len, d_model)
        print(f'decoded_latent.shape: {decoded_latent.shape}')
        output = self.decoder(decoded_latent, memory)
        return output

def get_json_files(data_folder):
    json_files = []
    for root, dirs, files in os.walk(data_folder):
        for file in files:
            if file.endswith('.json'):
                json_files.append(os.path.join(root, file))
    return json_files

def load_data(data_folder, sample_rate):
    json_files = get_json_files(data_folder)
    if not json_files:
        raise ValueError(f"No JSON files found in {data_folder}")

    random.shuffle(json_files)
    train_dataset = PulseDataset(json_files, sample_rate)
    print(f'Train dataset len: {len(train_dataset)}')
    return train_dataset

def train_transformer(model, dataloader, optimizer, criterion, device, model_path, epochs=20000, save_interval=1):
    
    min_loss = float('inf')


    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print(f'Model loaded:{model_path}')
    #find init loss 
    init_loss = 0
    for data in dataloader:
        data = data.to(device)
        outputs = model(data)
        mask = torch.isfinite(data)
        loss = criterion(outputs[mask], data[mask])
        # loss = criterion(outputs, data)
        init_loss += loss.item()
    min_loss = init_loss / len(dataloader)
    print(f'Initial loss: {min_loss}')
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for data in dataloader:
            data = data.to(device)
            optimizer.zero_grad()
            outputs = model(data)

            # Mask out NaN values from loss calculation
            mask = torch.isfinite(data)
            loss = criterion(outputs[mask], data[mask])
            # loss = criterion(outputs, data)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
        total_loss /= len(dataloader)
        print(f'Epoch {epoch + 1}, Loss: {total_loss}')
        if (epoch + 1) % save_interval == 0 and total_loss < min_loss * 0.95:
            min_loss = total_loss
            torch.save(model.state_dict(), model_path)
            print(f'Saved model to {model_path}, epoch loss: {total_loss}')

def predict_per_pulse(model, json_file, sample_rate, device):
    print(f'json_file:{json_file}')
    with open(json_file, 'r') as f:
        json_data = json.load(f)
        signal = process_DB_rawdata(json_data) if 'smoothed_data' not in json_data else np.array(json_data['smoothed_data'], dtype=np.float32)
        original_sample_rate = json_data.get('sample_rate', 100)
        if original_sample_rate != sample_rate:
            resample_ratio = sample_rate / original_sample_rate
            num_samples = int(len(signal) * resample_ratio)
            signal = scipy.signal.resample(signal, num_samples)
            x_points = [int(x * resample_ratio) for x in json_data['x_points']]  # 調整x_points索引
        else:
            x_points = json_data['x_points']

    if len(signal) == 0:
        print("Warning: Skipping signal due to invalid length")
        return []    

    # 全局標準化
    mean = np.mean(signal)
    std = np.std(signal)
    signal = (signal - mean) / std

    model.eval()
    criterion = nn.MSELoss()
    losses = []

    for i in range(len(x_points) - 1):
        start_idx = x_points[i]
        end_idx = x_points[i + 1]
        if start_idx >= len(signal) or end_idx > len(signal) or start_idx >= end_idx:
            print(f"Warning: Over bound index:({start_idx},{end_idx}) due to invalid indices, len(signal): {len(signal)}")
            continue

        pulse = signal[start_idx:end_idx]
        if np.isnan(pulse).any():
            print(f"Warning: Skipping pulse index:({start_idx},{end_idx}) due to invalid indices, len(pulse): {len(pulse)}")
            continue

        pulse = torch.tensor(pulse).unsqueeze(0).to(device)
        output = model(pulse)
        if torch.isnan(output).any():
            print(f"Warning: Output pulse index:({start_idx},{end_idx}) due to invalid indices, len(output): {len(output)}")
            continue

        loss = criterion(output, pulse)
        if torch.isnan(loss):
            print(f"Warning: Loss: due to pulse:{pulse}, output:{output}")
            continue

        losses.append(loss.item())
    return losses


def predict_transformer_reconstructed_signal(signal, sample_rate, peaks):
    criterion = nn.MSELoss()
    losses = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    d_model = 128
    nhead = 8
    num_encoder_layers = 2
    num_decoder_layers = 2
    latent_dim = 64
    lstm_hidden_dim, lstm_num_layers = 128, 4
    model = TransformerAutoencoder4(d_model, nhead, num_encoder_layers, num_decoder_layers, latent_dim, lstm_hidden_dim, lstm_num_layers).to(device)
    model.load_state_dict(torch.load('pulse_transformer_autoencoder4.pth'))
    model.eval()

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

    # 逐拍重建
    reconstructed_signal = np.copy(signal)
    for i in range(len(peaks) - 1):
        start_idx = peaks[i]
        end_idx = peaks[i + 1]
        pulse = signal[start_idx:end_idx]
        pulse_length = end_idx - start_idx  # 記錄脈衝的原始長度
        pulse_tensor = torch.tensor(pulse, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            reconstructed_pulse = model(pulse_tensor)
            loss = criterion(reconstructed_pulse, pulse_tensor)
            losses.append(loss.item())
            reconstructed_pulse = reconstructed_pulse.squeeze().cpu().numpy()

        # 將重建的脈衝還原為原始長度
        reconstructed_pulse = scipy.signal.resample(reconstructed_pulse, pulse_length)
        reconstructed_signal[start_idx:end_idx] = reconstructed_pulse

    print(f'Reconstructed signal loss: {losses}')

    # 反標準化
    reconstructed_signal = reconstructed_signal * std + mean

    # 根據原始採樣率調整重構信號的長度
    original_length = int(len(reconstructed_signal) / resample_ratio)
    reconstructed_signal = scipy.signal.resample(reconstructed_signal, original_length)

    return reconstructed_signal

def load_and_lock_transformer_weights(model, model_fix):
    # Load the pre-trained weights
    model.encoder.load_state_dict(model_fix.encoder.state_dict())
    model.decoder.load_state_dict(model_fix.decoder.state_dict())

    # # Lock the layers
    # for param in model.encoder.parameters():
    #     param.requires_grad = False
    # for param in model.decoder.parameters():
    #     param.requires_grad = False

    print("Loaded and locked transformer weights.")


class TransformerVAE(nn.Module):
    def __init__(self, d_model, nhead, num_encoder_layers, num_decoder_layers, latent_dim, lstm_hidden_dim, lstm_num_layers, dim_feedforward=2048, dropout=0.1):
        super(TransformerVAE, self).__init__()
        self.encoder = PulseEncoder(d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout)
        self.decoder = PulseDecoder(d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout)

        self.lstm_encoder = nn.LSTM(d_model, lstm_hidden_dim, lstm_num_layers, batch_first=True, bidirectional=True)
        self.lstm_decoder = nn.LSTM(lstm_hidden_dim * 2, d_model, lstm_num_layers, batch_first=True)
        self.to_mu = nn.Linear(lstm_hidden_dim * 2, latent_dim)
        self.to_logvar = nn.Linear(lstm_hidden_dim * 2, latent_dim)
        self.from_latent = nn.Linear(latent_dim, lstm_hidden_dim * 2)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, src):
        # Encoder Part
        memory = self.encoder(src)
        memory = memory.permute(1, 0, 2)  # [batch_size, seq_len, d_model]
        _, (hidden, _) = self.lstm_encoder(memory)
        hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)  # Concatenate the last two hidden states from the bidirectional LSTM
        mu = self.to_mu(hidden)  # [batch_size, latent_dim]
        logvar = self.to_logvar(hidden)  # [batch_size, latent_dim]
        z = self.reparameterize(mu, logvar)  # [batch_size, latent_dim]

        # Decoder Part
        decoder_input = self.from_latent(z).unsqueeze(1).repeat(1, memory.size(1), 1)  # [batch_size, seq_len, lstm_hidden_dim * 2]
        decoded_latent, _ = self.lstm_decoder(decoder_input)
        decoded_latent = decoded_latent.permute(1, 0, 2)  # [seq_len, batch_size, d_model]
        output = self.decoder(decoded_latent, memory.permute(1, 0, 2))  # [seq_len, batch_size, d_model]

        return output, mu, logvar

def loss_function(recon_x, x, mu, logvar):
    BCE = nn.MSELoss()(recon_x, x)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

def trainVAE(model, dataloader, optimizer, device, epochs=1000):
    model.to(device)
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in dataloader:
            pulses = batch.to(device)
            optimizer.zero_grad()
            output, mu, logvar = model(pulses)
            loss = loss_function(output, pulses, mu, logvar)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(dataloader)}')

def main():
    training_folder = 'labeled_DB'
    data_folder = 'DB'
    sample_rate = 100
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'torch device: {device}')

    train_dataset = load_data(training_folder, sample_rate)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    # save_pulses_to_csv(train_dataset, 'pulse_data.csv')
    input()
    latent_dim = 64#32  # or any other desired value
    d_model = 128
    nhead = 8
    num_encoder_layers = 2
    num_decoder_layers = 2

    # model = TransformerAutoencoder(d_model, nhead, num_encoder_layers, num_decoder_layers).to(device)
    # # if os.path.exists('transformer_autoencoder2.pth'):
    # #     model.load_state_dict(torch.load('transformer_autoencoder2.pth'))

    # trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(f'Total number of model parameters: {trainable_params}, model:{model}') 
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    # criterion = nn.MSELoss()

    # train_transformer(model, train_dataloader, optimizer, criterion, device, epochs=300)
    # # torch.save(model.state_dict(), 'transformer_autoencoder2.pth')
    # input()
    lstm_hidden_dim, lstm_num_layers = 128, 4
    model = TransformerAutoencoder4(d_model, nhead, num_encoder_layers, num_decoder_layers, latent_dim, lstm_hidden_dim, lstm_num_layers).to(device)
    # model_path = 'pulse_transformer_autoencoder2_fix_outer.pth' #作為預訓練知識
    model_path = 'pulse_transformer_autoencoder4.pth'
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
    # model_path = 'pulse_transformer_autoencoder4_1.pth'
    # #

    
    # model_fix = TransformerAutoencoder3(d_model, nhead, num_encoder_layers, num_decoder_layers).to(device)
    # model_fix.load_state_dict(torch.load('pulse_transformer_autoencoder3.pth'))
    # load_and_lock_transformer_weights(model, model_fix)
    #
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    input(f'Total number of model parameters: {trainable_params}, model:{model}') 
    # # # Define optimizer and criterion
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-6)#, lr=1e-5)
    criterion = nn.MSELoss()
    # Train the new model
    
    train_transformer(model, train_dataloader, optimizer, criterion, device, model_path)


    # num_epochs = 200
    # for epoch in range(num_epochs):
    #     for i, x in enumerate(train_dataloader):
    #         x = x.to(device)
    #         # print(f'x.shape: {x.shape}')
    #         optimizer.zero_grad()
    #         outputs = model(x)
    #         # input(f'outputs.shape: {outputs.shape}')
    #         loss = criterion(outputs, x)
    #         loss.backward()
    #         optimizer.step()
    #     print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')

    # Save the new model
    # torch.save(model.state_dict(), 'new_transformer_autoencoder.pth')

    input()

    # 初始化PulseEncoder
    # pulse_encoder = PulseEncoder(d_model, nhead, num_encoder_layers).to(device)

    # # 將預訓練的TransformerAutoencoder編碼器部分的參數複制到PulseEncoder
    # pulse_encoder.encoder.load_state_dict(model.encoder.state_dict())
    # pulse_encoder.pos_encoder.load_state_dict(model.pos_encoder.state_dict())
    # pulse_encoder.transformer_encoder.load_state_dict(model.transformer.encoder.state_dict())


    # # 初始化PulseDecoder
    # pulse_decoder = PulseDecoder(d_model, nhead, num_decoder_layers).to(device)
    # # 將預訓練的TransformerAutoencoder解碼器部分的參數複制到PulseDecoder
    # pulse_decoder.decoder.load_state_dict(model.decoder.state_dict())
    # pulse_decoder.transformer_decoder.load_state_dict(model.transformer.decoder.state_dict())


    # seq_to_vec_encoder = SeqToVecEncoder(128, 96, 64).to(device)
    # vec_to_seq_decoder = VecToSeqDecoder(64, 96, 128).to(device)
    # integrated_model = PulseAutoencoder(pulse_encoder, seq_to_vec_encoder, vec_to_seq_decoder, pulse_decoder).to(device)
    # if os.path.exists('integrated_pulse2latent_autoencoder.pth'):
    #     integrated_model.load_state_dict(torch.load('integrated_pulse2latent_autoencoder.pth'))
    #     # 凍結PulseEncoder的參數
    # for param in pulse_encoder.parameters():
    #     param.requires_grad = False

    # for param in pulse_decoder.parameters():
    #     param.requires_grad = False
    # # 定義損失函數和優化器
    # criterion = nn.MSELoss()
    # optimizer = torch.optim.Adam(integrated_model.parameters(), lr=1e-4)

    # # 訓練循環
    # torch.autograd.set_detect_anomaly(True)

    # num_epochs = 500
    # for epoch in range(num_epochs):
    #     for i, x in enumerate(train_dataloader):
    #         x = x.to(device)
            
    #         optimizer.zero_grad()
    #         reconstructed = integrated_model(x)
    #         loss = criterion(reconstructed, x)
    #         loss.backward()
    #         optimizer.step()
            
    #     print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
    # #save model
    # torch.save(integrated_model.state_dict(), 'integrated_pulse2latent_autoencoder.pth')

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
                        losses = predict_per_pulse(model, json_path, sample_rate, device)
                        # print(f'losss: {losses}')
                        mean_loss = np.mean(losses)
                        std_loss = np.std(losses)
                        # print(f'Mean loss: {mean_loss}, std loss: {std_loss}')
                        results[subject_folder][is_trainingset+json_file] = f"{mean_loss:.7f}+-{std_loss:.7f}"
                        print(f"{subject_folder}/{json_file}: {mean_loss:.7f}+-{std_loss:.7f}")
                        # print(f'losses: {losses}')
                    except Exception as e:
                        print(f"Exception {subject_folder}/{json_file}: {e}")

    # Save results to a JSON file
    with open('prediction_results_per_pulse.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)


if __name__ == '__main__':
    main()
