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
                
                file_name = os.path.basename(file_path)
                print(f'load file_name:{file_name}')
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

class UNetAutoencoder(nn.Module):
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
        return x.squeeze(1)  # 確保輸出形狀與輸入匹配

class WearingDataset(Dataset):
    def __init__(self, json_files, window_size, sample_rate=100, overlap_ratio=0.99):
        self.data = []
        self.sample_rate = sample_rate
        self.load_data(json_files, window_size, overlap_ratio)

    def load_data(self, json_files, window_size, overlap_ratio):
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
    print(f'Train dataset shape: {train_dataset.data[0].shape}')
    return train_dataset, test_dataset, [os.path.basename(f) for f in train_files], [os.path.basename(f) for f in test_files]


def train_autoencoder(model, dataloader, optimizer, criterion, device, epochs=300):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for data in dataloader:
            data = data.to(device).unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, data)
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
        segment_tensor = torch.tensor(segment).view(1, 1, -1).to(device)
        with torch.no_grad():
            output = model(segment_tensor)
            loss = criterion(output, segment_tensor)
            losses.append(loss.item())
    
    return losses

def main():
    root = tk.Tk()
    root.title("Anomaly Detection Viewer")
    data_folder = 'DB' #'labeled_DB'
    training_folder = 'labeled_DB'
    window_size = 100  # Assuming each second contains 100 sampling points
    sample_rate = 100
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'torch device: {device}')


    # Initialize and train the U-Net autoencoder
    model = UNetAutoencoder(1, window_size).to(device)

    if os.path.exists('unet_autoencoder.pt'):
        model.load_state_dict(torch.load('unet_autoencoder.pt'))
    # else:
    json_files = get_json_files(training_folder)
    # Load data and create data loaders
    train_dataset, test_dataset, train_files, test_files = load_data(training_folder, window_size, sample_rate)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train_autoencoder(model, train_loader, optimizer, criterion, device)
    # test_autoencoder(model, test_loader, criterion, device)


    # #save model
    torch.save(model.state_dict(), 'unet_autoencoder.pt')

    results = {}
    for subject_folder in os.listdir(data_folder):
        subject_path = os.path.join(data_folder, subject_folder)
        if os.path.isdir(subject_path):
            results[subject_folder] = {}
            for json_file in os.listdir(subject_path):
                if json_file.endswith('.json'):
                    json_path = os.path.join(subject_path, json_file)
                    
                    # 檢查訓練資料夾中是否存在相同相對路徑的檔案
                    training_json_path = os.path.join(training_folder, subject_folder, json_file)
                    if os.path.exists(training_json_path):
                        print(f"Skipping prediction for {subject_folder}/{json_file} (already trained)")
                        continue
                    
                    try:
                        losses = predict_per_second(model, json_path, sample_rate, device)
                        mean_loss = np.mean(losses)
                        std_loss = np.std(losses)
                        results[subject_folder][json_file] = f"{mean_loss:.7f}+-{std_loss:.7f}"
                        print(f"{subject_folder}/{json_file}: {mean_loss:.7f}+-{std_loss:.7f}")
                    except Exception as e:
                        print(f"Exception {subject_folder}/{json_file}: {e}")

    # Save results to a JSON file
    with open('prediction_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)

    # print("Prediction results saved to 'prediction_results.json'.")
    # train_files = []
    # test_files = []
    # app = Application(model, device, train_files, test_files,  master=root)
    # app.mainloop()

if __name__ == "__main__":
    main()