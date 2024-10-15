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
from scipy.stats import norm
from torchviz import make_dot
from preprocessing import process_DB_rawdata, get_json_files, add_noise_with_snr, add_gaussian_noise_torch, PulseDataset
from model_find_peaks import detect_peaks_from_signal
from tqdm import tqdm
import math
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import h5py

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import pandas as pd
import plotly.express as px
from dash.dependencies import Input, Output, State


def combinations(arr, k):
    result = []
    def generate_combinations(start, path):
        if len(path) == k:
            result.append(path)
            return
        for i in range(start, len(arr)):
            generate_combinations(i + 1, path + [arr[i]])
    
    generate_combinations(0, [])
    return result

class EPGBaselinePulseAutoencoder(nn.Module):
    def __init__(self, target_len, hidden_dim=50, latent_dim=30, dropout=0.5):
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

def encode(data):
    signal = data['smoothed_data']
    sample_rate = data['sample_rate']
    peaks = data['x_points']

    target_len = 100
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = 'pulse_interpolate_autoencoder2.pth' # target_len = 200
    model_path = 'pulse_interpolate_autoencoder.pth' # target_len = 100
    model_path = 'pulse_interpolate_autoencoder_test.pth' # target_len = 100 
    model = EPGBaselinePulseAutoencoder(target_len).to(device)
    model.load_state_dict(torch.load(model_path,map_location = device))
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

    latent_vector_list = []
    for i in range(len(peaks) - 1):
        start_idx = peaks[i]
        end_idx = peaks[i + 1]
        pulse = signal[start_idx:end_idx]
        pulse_length = end_idx - start_idx  # 記錄脈衝的原始長度

        if pulse_length > 1:
            # 插值到目標長度
            interp_func = scipy.interpolate.interp1d(np.arange(pulse_length), pulse, kind='linear', fill_value="extrapolate")
            pulse_resampled = interp_func(np.linspace(0, pulse_length - 1, target_len))
            pulse_tensor = torch.tensor(pulse_resampled, dtype=torch.float32).unsqueeze(0).to(device)

            with torch.no_grad():
                reconstructed_pulse, latent_vector = model(pulse_tensor)
                reconstructed_pulse = reconstructed_pulse.squeeze().cpu().numpy()
                latent_vector = latent_vector.squeeze().cpu().numpy()
                latent_vector = np.concatenate([latent_vector, np.array([pulse_length/100])], axis=0)

                # print(f'i:{i}, start_idx:{start_idx}, latent_vector:{latent_vector}')
                latent_vector_list.append(latent_vector)
    
    return latent_vector_list

    








def main():
    file_paths = ['labeled_DB/113/(2024-06-17 14-13-54),(EPG - 111 - 公司2F).json', 'labeled_DB/113/(2024-06-17 14-15-30),(EPG - 111 - 公司34).json','labeled_DB/113/(2024-06-17 14-18-58),(EPG - 111 - 公司3B).json','labeled_DB/113/(2024-06-17 14-25-48),(EPG - 111 - 公司29).json','labeled_DB/113/(2024-06-17 14-28-36),(EPG - 111 - 公司55).json']
    #file_paths = ['labeled_DB/20/(2023-08-05 22-41-23),(sit, after eat fruit).json']
    array = []
    tensors = []

    for file_path in file_paths:
        with open(file_path, 'r') as f:
            data = json.load(f)
        print(data['smoothed_data'][data['x_points'][0]])
        encode_data = encode(data)
        encode_data = np.array(encode_data)
        print("np_encode_data:",encode_data[:1])
        encode_data = encode_data[:30]
        encode_data = encode_data.T
        encode_data = encode_data[:30]
        array.append(encode_data)
        #print("np_encode_data:",len(encode_data))
        
        encode_data_tensor = torch.from_numpy(encode_data)
        #print("encode_data_tensor:",encode_data_tensor)
        tensors.append(encode_data_tensor)

    # ##==========畫圖===========
    # #subplot_width = 5
    # #subplot_height =5
    # #fig, axes = plt.subplots(len(array), len(array[0]), figsize=(subplot_width * len(array[0]), subplot_height * len(array)))
    # ##fig, axes = plt.subplots(10, 6, figsize=(12, 8))
    # #擬合高斯分佈，計算均值和標準差
    # for i in range(len(array)):
    #     for j in range(len(array[i])):
    #         mu, std = norm.fit(array[i][j])

    #         # 創建範圍的 x 值，用於繪製曲線
    #         x = np.linspace(min(array[i][j]), max(array[i][j]), 100)

    #         # 計算高斯分佈的 y 值
    #         y = norm.pdf(x, mu, std)

    #         # # 繪製數據的直方圖和高斯分佈曲線
    #         plt.hist(array[i][j], bins=10, density=True, alpha=0.6, color='g', label='Data')
    #         plt.plot(x, y, 'r-', label=f'Gaussian Fit (mean={mu:.2f}, std={std:.2f})')

    #         # 設置圖的標題和顯示
    #         plt.title('Gaussian Distribution Fit')
    #         plt.xlabel('Value')
    #         plt.ylabel('Density')
    #         plt.legend()
    #         plt.show()

    #         # # 绘制数据的直方图和高斯分布曲线
    #         # axes[i, j].hist(array[i][j], bins=10, density=True, alpha=0.6, color='g', label='Data')
    #         # axes[i, j].plot(x, y, 'r-', label=f'Gaussian Fit (mean={mu:.2f}, std={std:.2f})')

    #         # # 设置图的标题和显示
    #         # axes[i, j].set_title(f'Plot ({i+1},{j+1})')
    #         # axes[i, j].set_xlabel('Value')
    #         # axes[i, j].set_ylabel('Density')
    #         # axes[i, j].legend()

    # #plt.tight_layout()
    # #plt.show()
    # ##==========畫圖===========



    #print("tensors:",tensors)
    
    
    #kl_div_results = []
    # for i in range(len(tensors[0])): 

    #     # 需要將數據轉換為對數概率分佈 (log_softmax)
    #     log_subarray1 = F.log_softmax(tensors[0][i], dim=0)
    
    #     # 目標是概率分佈，使用 softmax 處理
    #     subarray2 = F.softmax(tensors[1][i], dim=0)
    
    #     # 計算 KL 散度
    #     kl_div = F.kl_div(log_subarray1, subarray2, reduction='batchmean')
    
    #     kl_div_results.append(kl_div.item())
    
    
    # print("kl_div_results: ",kl_div_results)
    # kl_div_sum = sum(kl_div_results)
    # print("kl_div_sum",kl_div_sum)

    for i in range(len(tensors)):
        for j in range(i + 1, len(tensors)):
            kl_div_results = []
            for k in range(len(tensors[0])): 
                # 需要將數據轉換為對數概率分佈 (log_softmax)
                log_subarray1 = F.log_softmax(tensors[i][k], dim=0)
            
                # 目標是概率分佈，使用 softmax 處理
                subarray2 = F.softmax(tensors[j][k], dim=0)
            
                # 計算 KL 散度
                kl_div = F.kl_div(log_subarray1, subarray2, reduction='batchmean')
            
                kl_div_results.append(kl_div.item())
                #result.append([arr[i], arr[j]])
            print("i,j:",i,j)
            #print("kl_div_results: ",kl_div_results)
            kl_div_sum = sum(kl_div_results)
            print("kl_div_sum",kl_div_sum)



        



if __name__ == '__main__':
    main()





# # 計算每一列的平均值和標準差
# encode_mean = np.mean(encode_data, axis=0)
# encode_std = np.std(encode_data, axis=0)
# print("encode_mean",encode_mean)
# print("encode_std",encode_std)
