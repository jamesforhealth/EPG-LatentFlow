import os
import json
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch import nn, optim
import scipy
import sys
# import matplotlib.pyplot as plt
from preprocessing import process_DB_rawdata
# import coremltools as ct
import time
import torch.nn.functional as F
def save_preprocessed_data(json_files, window_size, sample_rate, output_dir):
    data = []
    labels = []

    for json_file in json_files:
        print(f'Processing file: {json_file}')
        with open(json_file, 'r') as f:
            try:
                data_json = json.load(f)
                signal = data_json['smoothed_data']
                peaks = set(data_json['x_points'])
                original_sample_rate = data_json.get('sample_rate', 100)

                if original_sample_rate != sample_rate:
                    num_samples = int(len(signal) * sample_rate / original_sample_rate)
                    signal = scipy.signal.resample(signal, num_samples)
                    peaks = {int(peak * sample_rate / original_sample_rate) for peak in peaks}

                num_samples = len(signal)
                for start in range(0, num_samples - window_size + 1, 100):
                    end = start + window_size
                    segment = signal[start:end]
                    label = np.zeros(window_size, dtype=np.float32)
                    for peak in peaks:
                        if start <= peak < end:
                            peak_pos = peak - start
                            sigma = 3
                            for i in range(max(0, peak_pos - 3 * sigma), min(window_size, peak_pos + 3 * sigma + 1)):
                                label[i] += np.exp(-0.5 * ((i - peak_pos) / sigma) ** 2)
                    label = np.clip(label, 0, 1)
                    data.append(segment)
                    labels.append(label)
            except Exception as e:
                print(f'Error processing file {json_file}: {e}')

    data = np.array(data, dtype=np.float32)
    labels = np.array(labels, dtype=np.float32)

    np.save(os.path.join(output_dir, 'data.npy'), data)
    np.save(os.path.join(output_dir, 'labels.npy'), labels)
    print(f'Data and labels saved to {output_dir}')


class PreprocessedDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        segment = self.data[idx]
        label = self.labels[idx]
        segment = segment.reshape(1, -1)  # 添加通道維度
        return torch.tensor(segment, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

# def load_datasets(data_path, labels_path, split_ratio=0.9):
#     data = np.load(data_path)
#     labels = np.load(labels_path)

#     total_samples = len(data)
#     indices = np.arange(total_samples)
#     np.random.shuffle(indices)

#     split_point = int(total_samples * split_ratio)
#     train_indices = indices[:split_point]
#     val_indices = indices[split_point:]

#     train_data, train_labels = data[train_indices], labels[train_indices]
#     val_data, val_labels = data[val_indices], labels[val_indices]

#     train_dataset = PreprocessedDataset(train_data, train_labels)
#     val_dataset = PreprocessedDataset(val_data, val_labels)

#     return train_dataset, val_dataset

############################################################################################


class EPGDataset(Dataset):
    def __init__(self, data_path, labels_path):
        self.data = np.load(data_path, mmap_mode='r')
        self.labels = np.load(labels_path, mmap_mode='r')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        segment = self.data[idx]
        label = self.labels[idx]
        segment = segment.reshape(1, -1)  # 添加通道维度
        return torch.from_numpy(segment).float(), torch.from_numpy(label).float()

def load_datasets(data_path, labels_path, split_ratio=0.9):
    dataset = EPGDataset(data_path, labels_path)
    train_size = int(split_ratio * len(dataset))
    val_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(42)  # 设置随机种子
    return random_split(dataset, [train_size, val_size], generator=generator)

def create_dataloader(dataset, batch_size, shuffle, num_workers=4):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)

############################################################################################

class PeakDetectionDataset(Dataset):
    def __init__(self, json_files, window_size, sample_rate=100):
        self.data = []
        self.labels = []
        self.sample_rate = sample_rate
        self.load_data(json_files, window_size)

    def load_data(self, json_files, window_size):
        for json_file in json_files:
            print(f'json_file:{json_file}')
            with open(json_file, 'r') as f:
                
                try:
                    data = json.load(f)
                    signal = data['smoothed_data']
                    peaks = set(data['x_points'])
                    original_sample_rate = data.get('sample_rate', 100)
                    
                    if original_sample_rate != self.sample_rate:
                        num_samples = int(len(signal) * self.sample_rate / original_sample_rate)
                        signal = scipy.signal.resample(signal, num_samples)
                        # 重新計算峰值的位置
                        peaks = {int(peak * self.sample_rate / original_sample_rate) for peak in peaks}
                    
                    num_samples = len(signal)
                    # print(f'num_samples:{num_samples}')
                    for start in range(0, num_samples - window_size + 1):#self.sample_rate):
                        end = start + window_size
                        segment = signal[start:end]
                        label = np.zeros(window_size, dtype=np.float32)
                        for peak in peaks:
                            if start <= peak < end:
                                # 在峰值位置使用高斯窗函數
                                peak_pos = peak - start
                                sigma = 3  # 控制高斯分佈的寬度
                                for i in range(max(0, peak_pos - 3 * sigma), min(window_size, peak_pos + 3 * sigma + 1)):
                                    label[i] += np.exp(-0.5 * ((i - peak_pos) / sigma) ** 2)
                        label = np.clip(label, 0, 1)  # 確保標籤值在0和1之間
                        self.data.append(segment)
                        self.labels.append(label)
                except:
                    print(f'json_file:{json_file} is not valid')


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        segment = np.array(self.data[idx], dtype=np.float32)
        segment = segment.reshape(1, -1)
        label = np.array(self.labels[idx], dtype=np.float32)
        return torch.tensor(segment), torch.tensor(label)

class WindowAttention1D(nn.Module):
    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1), num_heads))
        coords = torch.arange(window_size)
        coords_table = torch.stack(torch.meshgrid([coords])).flatten(1)
        relative_coords = coords_table[:, :, None] - coords_table[:, None, :]
        relative_coords += window_size - 1
        self.register_buffer("relative_position_index", relative_coords.squeeze())

    def forward(self, x):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size, self.window_size, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        attn = F.softmax(attn, dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        return x

class SwinTransformerBlock1D(nn.Module):
    def __init__(self, dim, num_heads, window_size, shift_size=0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention1D(dim, window_size, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim)
        )

    def forward(self, x):
        H = x.shape[1]
        x = self.norm1(x)
        
        # Cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=-self.shift_size, dims=1)
        else:
            shifted_x = x

        # Partition windows
        x_windows = shifted_x.view(-1, self.window_size, self.dim)

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows)

        # Merge windows
        shifted_x = attn_windows.view(-1, H, self.dim)

        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=self.shift_size, dims=1)
        else:
            x = shifted_x

        # FFN
        x = x + self.mlp(self.norm2(x))
        return x

class SwinTransformer1D(nn.Module):
    def __init__(self, input_size, patch_size, dim, depth, num_heads, window_size):
        super().__init__()
        self.patch_embed = nn.Conv1d(1, dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, input_size // patch_size, dim))
        
        self.layers = nn.ModuleList([])
        for i in range(depth):
            self.layers.append(
                SwinTransformerBlock1D(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if (i % 2 == 0) else window_size // 2
                )
            )

        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, input_size)

    def forward(self, x):
        x = self.patch_embed(x).transpose(1, 2)
        x = x + self.pos_embed
        
        for layer in self.layers:
            x = layer(x)
        
        x = self.norm(x)
        x = x.mean(dim=1)
        x = self.head(x)
        return torch.sigmoid(x)


class PeakDetectionDCNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(PeakDetectionDCNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 128, kernel_size=3, dilation=1, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(128, 64, kernel_size=3, dilation=2, padding=2)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, dilation=4, padding=4)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv1d(128, 128, kernel_size=3, dilation=8, padding=8)
        self.relu4 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.conv5 = nn.Conv1d(128, 256, kernel_size=3, dilation=16, padding=16)
        self.relu5 = nn.ReLU(inplace=True)
        self.conv6 = nn.Conv1d(256, 256, kernel_size=3, dilation=32, padding=32)
        self.relu6 = nn.ReLU(inplace=True)
        self.pool3 = nn.AdaptiveAvgPool1d(1)
        
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(256, input_size)  # 修改全連接層的輸出尺寸為input_size
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool1(x)
        
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.pool2(x)
        
        x = self.conv5(x)
        x = self.relu5(x)
        x = self.conv6(x)
        x = self.relu6(x)
        x = self.pool3(x)
        
        x = self.flatten(x)
        x = self.fc(x)
        x = self.sigmoid(x)
        
        return x

def test_model(model, dataloader, device, threshold=0.5):
    model.eval()
    predictions = []
    with torch.no_grad():
        for data, _ in dataloader:
            data = data.to(device)
            outputs = model(data)
            probabilities = outputs.squeeze(0)  # 移除批次維度
            predicted = (probabilities > threshold).cpu().numpy().astype(int)
            predictions.append(predicted)
    
    return predictions

# def train_model(model, train_dataloader, val_dataloader, criterion, optimizer, device, epochs=100):
#     for epoch in range(epochs):
#         model.train()
#         running_loss = 0.0
#         for data, labels in train_dataloader:
#             data, labels = data.to(device), labels.to(device)
#             optimizer.zero_grad()
#             outputs = model(data)
#             loss = criterion(outputs, labels)  # 確保標籤的形狀為(batch_size, window_size)
#             loss.backward()
#             optimizer.step()
#             running_loss += loss.item()

#         val_loss = validate_model(model, val_dataloader, criterion, device)
#         print(f'Epoch {epoch+1}, Training Loss: {running_loss / len(train_dataloader)}, Validation Loss: {val_loss}')
def train_model(model, train_dataloader, val_dataloader, criterion, optimizer, device, epochs=2000, patience=100):
    best_val_loss = float('inf')
    best_model = None
    patience_counter = 0
    start_time = time.time()
    # 計算初始loss
    initial_train_loss = validate_model(model, train_dataloader, criterion, device)
    initial_val_loss = validate_model(model, val_dataloader, criterion, device)
    print(f'Initial Training Loss: {initial_train_loss:.4f}, Initial Validation Loss: {initial_val_loss:.4f}')

    for epoch in range(epochs):
        epoch_start_time = time.time()
        model.train()
        running_loss = 0.0
        for data, labels in train_dataloader:
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_loss = running_loss / len(train_dataloader)
        val_loss = validate_model(model, val_dataloader, criterion, device)
        epoch_time = time.time() - epoch_start_time

        print(f'Epoch {epoch+1}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Time: {epoch_time:.2f}s')

        # 檢查是否需要保存模型
        if val_loss < best_val_loss * 0.95:
            best_val_loss = val_loss
            best_model = model.state_dict()
            torch.save(best_model, 'peak_detection_swin_transformer.pth')
            print(f'New best model saved at epoch {epoch+1}')
            patience_counter = 0
        else:
            patience_counter += 1

        # 提前停止檢查
        if patience_counter >= patience:
            print(f'Early stopping triggered after {epoch+1} epochs')
            break

    total_time = time.time() - start_time
    print(f'Total training time: {total_time:.2f}s')

    return best_model

def validate_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data, labels in dataloader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            loss = criterion(outputs, labels)  # 確保標籤的形狀為(batch_size, window_size)
            total_loss += loss.item()
    return total_loss / len(dataloader)



def get_json_files(data_folder):
    json_files = []
    for root, dirs, files in os.walk(data_folder):
        for file in files:
            if file.endswith('.json'):
                json_files.append(os.path.join(root, file))
    return json_files

def add_padding(signal, window_size):
    remainder = len(signal) % window_size
    if remainder != 0:
        # Calculate how much padding is needed to make the signal length a multiple of the window size
        padding_length = window_size - remainder
        # Pad the signal at the end with the mean of the signal or zero
        padded_signal = np.pad(signal, (0, padding_length), 'constant', constant_values=(0, 0))
    else:
        padded_signal = signal
    return list(padded_signal)

def predict_window(model, device, segment, start):
    segment = np.array(segment, dtype=np.float32).reshape(1, 1, -1)  # 調整為模型的輸入格式
    segment_tensor = torch.tensor(segment).to(device)

    with torch.no_grad():
        output = model(segment_tensor)
        predicted = (output > 0.5).cpu().numpy().astype(int)[0]
        peak_positions = np.where(predicted == 1)[0] + start  # 計算峰值的絕對位置    
    return peak_positions

def predict_peaks_json(model, device, json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
        signal = process_DB_rawdata(data) if 'smoothed_data' not in data else data['smoothed_data']
        sample_rate = data.get('sample_rate', 100)
    return predict_peaks(model, device, signal, sample_rate)

def predict_peaks(model, device, signal, sample_rate):
    signal = np.asarray(signal, dtype=np.float32)
    num_samples = len(signal)
    downsampled_signal = scipy.signal.resample(signal, int(num_samples * 100 / sample_rate))
    real_peaks = predict_peaks_core(model, device, downsampled_signal)
    resampled_peaks = resample_peaks(signal, real_peaks, sample_rate)
    resampled_peaks = sorted(list(set(resampled_peaks)))
    return resampled_peaks

def predict_peaks_core(model, device, signal):
    sample_rate = 100
    num_samples = len(signal)
    window_size = 200
    model.eval()
    peaks_indices = []
    
    # 計算完整的窗口數量
    num_windows = (num_samples - window_size) // int(sample_rate) + 1
    
    # 遍歷每個完整的窗口
    for i in range(num_windows):
        start = i * int(sample_rate)
        end = start + window_size
        segment = signal[start:end]
        peak_positions = predict_window(model, device, segment, start)
        peaks_indices = list(set(peaks_indices).union(set(peak_positions)))
    
    # 處理最後一個可能不完整的窗口
    if end < num_samples:
        start = num_samples - window_size
        end = num_samples
        segment = signal[start:end]
        peak_positions = predict_window(model, device, segment, start)
        peaks_indices = list(set(peaks_indices).union(set(peak_positions)))
    
    peak_ranges = sorted(list(set(peaks_indices)))
    # print(f'Detected: {peak_ranges}')
    if len(peak_ranges) == 0:
        return []
    real_peaks = []
    last_i = peak_ranges[0]
    argmax_signal = last_i
    for i in peak_ranges[1:]:
        if i != last_i + 1:
            real_peaks.append(argmax_signal)
            argmax_signal = i            
        else:
            if signal[i] > signal[argmax_signal]:
                argmax_signal = i
        last_i = i
    
    # 將最後一個 argmax_signal 加入到 real_peaks 中
    if argmax_signal not in real_peaks:
        real_peaks.append(argmax_signal)

    return real_peaks
def resample_peaks(signal, peak_indices, original_sample_rate, window_size=0.1):
    true_peaks = []
    for peak_index in peak_indices:
        # 計算峰值點在原始採樣率下的時間戳記
        timestamp = peak_index / 100

        # 計算時間窗口的起始和結束位置
        window_start = int(max(0, (timestamp - window_size / 2) * original_sample_rate))
        window_end = int(min(len(signal), (timestamp + window_size / 2) * original_sample_rate))

        # 在時間窗口內找到訊號值最大的點
        window_signal = signal[window_start:window_end]
        true_peak_index = window_start + np.argmax(window_signal)
        true_peaks.append(true_peak_index)

    return true_peaks


def detect_peaks_in_window(signal, sample_rate, window_size=200, model_path='peak_detection_model3.pt'):
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = PeakDetectionDCNN(window_size, num_classes=1).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        
        all_window_peaks = []
        num_samples = len(signal)

        # 遍历整个信号，每次移动一个 sample_rate（即1秒）
        for start in range(0, num_samples - window_size + 1, sample_rate):
            end = start + window_size
            segment = signal[start:end]
            segment_peaks = predict_window(model, device, segment, start)
            if len(segment_peaks) > 1:  # 如果该段包含完整的脉冲波（即多于一个 peak）
                all_window_peaks.extend(segment_peaks)

        # 后处理以找到真正的 peak
        peak_ranges = sorted(all_window_peaks)
        if len(peak_ranges) == 0:
            return []
        
        real_peaks = []
        last_i = peak_ranges[0]
        argmax_signal = last_i
        
        for i in peak_ranges[1:]:
            if i != last_i + 1:
                real_peaks.append(argmax_signal)
                argmax_signal = i            
            else:
                if signal[i] > signal[argmax_signal]:
                    argmax_signal = i
            last_i = i
        
        if argmax_signal not in real_peaks:
            real_peaks.append(argmax_signal)
        
        return real_peaks
    except TypeError as e:
        print(f'Type error in detect_peaks_in_window: {e}')
        return []
    except Exception as e:
        print(f'Error in detect_peaks_in_window: {e}')
        return []

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    data_folder = 'labeled_DB'
    json_files = get_json_files(data_folder)

    if not json_files:
        print(f"No JSON files found in {data_folder} and its subfolders. Please check the directory.")
        return
    split_ratio = 0.9
    random.shuffle(json_files)
    split_point = int(split_ratio * len(json_files))
    train_files = json_files[:split_point]
    val_files = json_files[split_point:]

    # val_files = ['labeled_DB/149/(2024-04-22 16-06-22),(EPG_R - 96_70_62 - 公司).json', 'labeled_DB/18/(2023-11-04 07-42-57),(General log - EPG).json', 'labeled_DB/88/(2024-05-16 11-24-54),(EPG -  - 公司).json', 'labeled_DB/142/(2024-04-19 15-47-32),(EPG_R - 97_57_72 - 公司).json', 'labeled_DB/4/(2024-02-16 15-48-04),(EPG - 115_76 - 公司).json', 'labeled_DB/4/(2024-03-21 14-56-20),(EPG - radial cuff - 公司).json', 'labeled_DB/68/(2023-11-16 22-30-29),( - EPG).json', 'labeled_DB/102/(2023-08-16 12-51-17),().json', 'labeled_DB/4/(2023-07-19 13-37-29),(0203 飯後).json', 'labeled_DB/42/(2024-04-12 11-10-42),(EPG - 95_69_78  - 公司).json', 'labeled_DB/19/(2024-02-15 09-36-27),(128_76mmHg - EPG).json', 'labeled_DB/129/(2024-03-12 14-41-19),(EPG_L - test1 - 北榮心臟___).json', 'labeled_DB/19/(2024-02-15 09-35-35),(110_76 - EPG).json', 'labeled_DB/42/(2024-04-11 09-35-10),(EPG_L - 94_65_76 飯前 - 公司).json', 'labeled_DB/111/(2024-04-11 09-32-28),(EPG_L - 早上餐前節水 125_91mmHg).json', 'labeled_DB/118/(2024-02-12 22-55-49),(Covid day +6 103_73mmHg - EPG).json', 'labeled_DB/50/(2023-07-14 15-31-50),().json', 'labeled_DB/111/(2024-03-28 15-35-38),(EPG_L - 奶茶118_93_80 - 公司(儀器可能異常)).json', 'labeled_DB/129/(2024-03-12 14-41-19),(EPG_L - test1 - 北榮心臟 - 正式同步___).json', 'labeled_DB/144/(2024-04-22 11-52-58),(EPG_L - 86_64_77 - 公司).json', 'labeled_DB/23/(2023-05-08 00-23-38),().json', 'labeled_DB/4/(2024-04-11 13-10-18),(EPG_R 午餐前 122_81mmHg).json', 'labeled_DB/42/(2024-02-02 10-57-11),(100hz 93_59_83 - EPG).json', 'labeled_DB/125/(2024-04-15 10-50-56),(EPG_L - 131_99mmHg - 公司 ).json', 'labeled_DB/68/(2023-12-13 18-19-36),( - EPG).json']
    # val_files = [os.path.join(p) for p in val_files]
    # print(f'validation files: {val_files}')
    window_size = 200  # Assuming a fixed window size for simplicity
    sample_rate = 100
    save_preprocessed_data(json_files, window_size, sample_rate, './')
    input()
    data_path = 'data.npy'
    labels_path = 'labels.npy'
    
    train_dataset, val_dataset = load_datasets(data_path, labels_path, split_ratio=0.9)
    train_dataloader = DataLoader(train_dataset, batch_size=64, num_workers=4, pin_memory=True, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=64, num_workers=4, pin_memory=True, shuffle=False)
    # train_dataset = PeakDetectionDataset(train_files, window_size)
    # val_dataset = PeakDetectionDataset(val_files, window_size)
    # train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    # val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    #印出訓練跟測試資料的長度
    print(f'Number of training samples: {len(train_dataset)}')
    print(f'Number of validation samples: {len(val_dataset)}')


    # 使用示例
    model = SwinTransformer1D(
        input_size=window_size,  # 假设输入长度为1250
        patch_size=10,   # 每10個點作為一個patch
        dim=16,          # 特徵維度
        depth=4,         # Transformer 層數
        num_heads=4,     # 注意力頭數
        window_size=20   # 窗口大小
    ).to(device)

    # input_size = window_size
    # num_classes = 1
    # model = PeakDetectionDCNN(window_size, 1).to(device)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total number of model parameters: {trainable_params}')
    #如果模型存在
    # if os.path.exists('peak_detection_model3.pt'):
    #     model.load_state_dict(torch.load('peak_detection_model3.pt', map_location=device))
    if os.path.exists('peak_detection_swin_transformer.pth'):
        model.load_state_dict(torch.load('peak_detection_swin_transformer.pth', map_location=device))
    else:
        print('No model found. Training from scratch.')

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_model(model, train_dataloader, val_dataloader, criterion, optimizer, device)

    #儲存模型
    # torch.save(model.state_dict(), 'peak_detection_model3.pt')

    # plt.ion()
    # for json_file in val_files:
    #     try:
    #         print(f'Predicting peaks for {json_file}')
    #         with open(json_file, 'r') as f:
    #             data = json.load(f)
    #             signal = data['smoothed_data']
    #             sample_rate = data.get('sample_rate', 100)

    #         predicted_peaks = predict_peaks(model, device, signal, sample_rate)

    #         # 創建一個新的圖形
    #         fig, ax = plt.subplots(figsize=(12, 6))

    #         # 繪製平滑訊號
    #         ax.plot(signal, color='blue', label='Smoothed Signal')

    #         # 繪製預測的 peak 點
    #         peak_positions = [signal[peak] for peak in predicted_peaks]
    #         ax.scatter(predicted_peaks, peak_positions, color='yellow', label='Predicted Peaks')

    #         ax.set_title(f'Signal and Predicted Peaks - {os.path.basename(json_file)}')
    #         ax.set_xlabel('Sample Index')
    #         ax.set_ylabel('Amplitude')
    #         ax.legend()

    #         # 顯示圖形
    #         plt.show()

    #         # 等待用戶關閉圖形窗口
    #         input("Press Enter to continue...")
    #         plt.close(fig)
    #     except Exception as e:
    #         print(f'Error in predicting peaks: {e}')

    # json_file = sys.argv[1]
    # predicted_peaks = predict_peaks_json(model, device, json_file)
    # print(predicted_peaks)

def detect_peaks_from_json(json_file, model_path='peak_detection_model3.pt'):
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        window_size = 200
        num_classes = 1
        model = PeakDetectionDCNN(window_size, num_classes).to(device)
        model.load_state_dict(torch.load(model_path))

        predicted_peaks = predict_peaks_json(model, device, json_file)
        return predicted_peaks
    except TypeError as e:
        print(f'Type error in detect_peaks_from_json: {e}')
        return []
    except Exception as e:
        print(f'Error in detect_peaks_from_json: {e}')
        return []

def detect_peaks_from_signal(signal, sample_rate, model_path='peak_detection_swin_transformer.pth'):
    if len(signal) == 0:
        print("Warning: Skipping signal due to invalid length")
        return []
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        window_size = 200
        model = SwinTransformer1D(
            input_size=window_size,  # 假設輸入長度為200
            patch_size=10,           # 每10個點作為一個patch
            dim=16,                  # 特徵維度
            depth=4,                 # Transformer層數
            num_heads=4,             # 注意力頭數
            window_size=20           # 窗口大小
        ).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))

        predicted_peaks = predict_peaks(model, device, signal, sample_rate)
        return predicted_peaks
    except TypeError as e:
        print(f'Type error in detect_peaks_from_signal: {e}')
        return []
    except Exception as e:
        print(f'Error in detect_peaks_from_signal: {e}')
        return []
    
# def detect_peaks_from_signal(signal, sample_rate, model_path='peak_detection_model3.pt'):
#     if len(signal) == 0:
#         print("Warning: Skipping signal due to invalid length")
#         return []
#     try:
#         # print(f'signal : {signal}')
#         # print(f"Input signal type: {type(signal)}")
#         # print(f"Input signal dtype: {np.array(signal).dtype}")
        
#         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         window_size = 200
#         num_classes = 1
#         model = PeakDetectionDCNN(window_size, num_classes).to(device)
#         model.load_state_dict(torch.load(model_path))

#         predicted_peaks = predict_peaks(model, device, signal, sample_rate)
#         # predicted_peaks = list(set(predicted_peaks)).sort()
#         return predicted_peaks
#     except TypeError as e:
#         print(f'Type error in detect_peaks_from_signal: {e}')
#         return []
#     except Exception as e:
#         print(f'Error in detect_peaks_from_signal: {e}')
#         return []

if __name__ == '__main__':
    main()
    # json_file = sys.argv[1]
    # predicted_peaks = detect_peaks_from_json(json_file)
    # print(predicted_peaks)