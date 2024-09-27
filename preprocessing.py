
import numpy as np
import os 
import torch
import json
import scipy
import random
from torch.utils.data import Dataset
def split_json_files(json_files, train_ratio=0.9):
    random.shuffle(json_files)
    split_point = int(len(json_files) * train_ratio)
    return json_files[:split_point], json_files[split_point:]
# def get_json_files(data_folder):
#     json_files = []
#     for root, dirs, files in os.walk(data_folder):
#         for file in files:
#             if file.endswith('.json'):
#                 json_files.append(os.path.join(root, file))
#     return json_files
def get_json_files(data_folder, exclude_keywords = ["TCCP", "ICP", "TICP"]):
    json_files = []
    for root, dirs, files in os.walk(data_folder):
        for file in files:
            if file.endswith('.json') and not any(keyword.lower() in file.lower() for keyword in exclude_keywords):
                json_files.append(os.path.join(root, file))
    return json_files

def gaussian_smooth(input, window_size, sigma):
    if window_size == 0.0:
        return input
    half_window = window_size // 2
    output = np.zeros_like(input)
    weights = np.zeros(window_size)
    weight_sum = 0

    # Calculate Gaussian weights
    for i in range(-half_window, half_window + 1):
        weights[i + half_window] = np.exp(-0.5 * (i / sigma) ** 2)
        weight_sum += weights[i + half_window]

    # Normalize weights
    weights /= weight_sum

    # Apply Gaussian smoothing
    for i in range(len(input)):
        smoothed_value = 0
        for j in range(-half_window, half_window + 1):
            index = i + j
            if 0 <= index < len(input):
                smoothed_value += input[index] * weights[j + half_window]
        output[i] = smoothed_value

    # Copy border values from the input
    output[:window_size] = input[:window_size]
    output[-window_size:] = input[-window_size:]

    return output

def process_DB_rawdata(data):
    raw_data = [-value for packet in data['raw_data'] for value in packet['datas']]
    sample_rate = data['sample_rate']
    print(f'Sample_rate: {sample_rate}')
    scale = int(3 * sample_rate / 100)
    return  np.array(gaussian_smooth(raw_data, scale, scale/4), dtype=np.float32)

def add_noise_with_snr(data, target_snr_db):
    signal_power = torch.mean(data ** 2)
    signal_power_db = 10 * torch.log10(signal_power)

    noise_power_db = signal_power_db - target_snr_db
    noise_power = 10 ** (noise_power_db / 10)
    
    noise = torch.sqrt(noise_power) * torch.randn_like(data)
    noisy_data = data + noise
    return noisy_data

def add_gaussian_noise_torch(data, std=0.001):
    """
    Add Gaussian noise to the input data using PyTorch.
    
    Args:
        data (torch.Tensor): Input data.
        mean (float): Mean of the Gaussian distribution (default is 0).
        std (float): Standard deviation of the Gaussian distribution (default is 0.1).
        
    Returns:
        torch.Tensor: Noisy data.
    """
    noise = torch.randn_like(data) * std
    noisy_data = data + noise
    return noisy_data

class PulseDataset(Dataset):  
    def __init__(self, json_files, sample_rate=100):
        self.data = []
        self.sample_rate = sample_rate
        self.load_data(json_files)

    def load_data(self, json_files):
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    json_data = json.load(f)
                    if json_data['anomaly_list']:
                        continue
                    signal = json_data['smoothed_data']
                    original_sample_rate = json_data.get('sample_rate', 100)
                    x_points = json_data['x_points']
                    # print(f'x_points:{x_points}, json_file:{json_file}')
                    if original_sample_rate != self.sample_rate:
                        num_samples = int(len(signal) * self.sample_rate / original_sample_rate)
                        signal = scipy.signal.resample(signal, num_samples)
                        x_points = [int(x * self.sample_rate / original_sample_rate) for x in x_points]
                    signal = self.normalize(signal)
                    for j in range(len(x_points) - 1):
                        pulse_start = x_points[j]
                        pulse_end = x_points[j + 1]
                        pulse = signal[pulse_start:pulse_end]
                        if len(pulse) > 40 : 
                            self.data.append(pulse)
                            # if len(pulse) <= 50 or len(pulse) >=  130 : input(f'pulse:{pulse}, j:{j}, pulse_start:{pulse_start}, pulse_end:{pulse_end}')
            except Exception as e:
                print(f'Error in loading {json_file}: {e}')

    def normalize(self, data):
        return (data - np.mean(data)) / (np.std(data) + 1e-8)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pulse = self.data[idx]
        return torch.tensor(pulse, dtype=torch.float32)
    

class PulseDataset(Dataset):
    def __init__(self, json_files, target_len, sample_rate=100):
        self.data = []
        self.target_len = target_len
        self.sample_rate = sample_rate
        self.load_data(json_files)

    def load_data(self, json_files):
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    json_data = json.load(f)
                    if json_data['anomaly_list']:
                        continue
                    signal = json_data['smoothed_data']
                    original_sample_rate = json_data.get('sample_rate', 100)
                    x_points = json_data['x_points']

                    if original_sample_rate != self.sample_rate:
                        num_samples = int(len(signal) * self.sample_rate / original_sample_rate)
                        signal = scipy.signal.resample(signal, num_samples)
                        x_points = [int(x * self.sample_rate / original_sample_rate) for x in x_points]
                    
                    signal = self.normalize(signal)
                    
                    for j in range(len(x_points) - 1):
                        pulse_start = x_points[j]
                        pulse_end = x_points[j + 1]
                        pulse = signal[pulse_start:pulse_end]
                        if len(pulse) > 40:
                            interp_func = scipy.interpolate.interp1d(np.arange(len(pulse)), pulse, kind='linear')
                            pulse_resampled = interp_func(np.linspace(0, len(pulse) - 1, self.target_len))
                            self.data.append(pulse_resampled)
            except Exception as e:
                print(f'Error in loading {json_file}: {e}')

    def normalize(self, data):
        return (data - np.mean(data)) / (np.std(data) + 1e-8)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pulse = self.data[idx]
        return torch.tensor(pulse, dtype=torch.float32)


class MeasurementPulseDataset(Dataset):
    def __init__(self, json_files, target_len, sample_rate=100):
        self.data = []
        self.target_len = target_len
        self.sample_rate = sample_rate
        self.measurement_indices = {}
        self.load_data(json_files)

    def load_data(self, json_files):
        for measurement_id, json_file in enumerate(json_files):
            try:
                with open(json_file, 'r') as f:
                    json_data = json.load(f)
                    if json_data['anomaly_list']:
                        continue
                    measurement_pulses = []
                    signal = json_data['smoothed_data']
                    original_sample_rate = json_data.get('sample_rate', 100)
                    x_points = json_data['x_points']

                    if original_sample_rate != self.sample_rate:
                        num_samples = int(len(signal) * self.sample_rate / original_sample_rate)
                        signal = scipy.signal.resample(signal, num_samples)
                        x_points = [int(x * self.sample_rate / original_sample_rate) for x in x_points]
                    
                    signal = self.normalize(signal)                   

                    for j in range(len(x_points) - 1):
                        pulse_start = x_points[j]
                        pulse_end = x_points[j + 1]
                        pulse = signal[pulse_start:pulse_end]
                        if len(pulse) > 40:
                            interp_func = scipy.interpolate.interp1d(np.arange(len(pulse)), pulse, kind='linear')
                            pulse_resampled = interp_func(np.linspace(0, len(pulse) - 1, self.target_len))
                            self.data.append(pulse_resampled)
                            measurement_pulses.append(len(self.data) - 1)
                    
                    self.measurement_indices[measurement_id] = measurement_pulses
            except Exception as e:
                print(f'Error in loading {json_file}: {e}')

    def normalize(self, data):
        return (data - np.mean(data)) / (np.std(data) + 1e-8)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pulse = self.data[idx]
        return torch.tensor(pulse, dtype=torch.float32)

    def get_measurement(self, measurement_id):
        return [self.__getitem__(idx) for idx in self.measurement_indices[measurement_id]]