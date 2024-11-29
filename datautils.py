import os
import h5py
import torch
import scipy
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import json

def predict_encoded_dataset(model, json_files, basedir='labeled_DB'):
    model.eval()
    encoded_data = {} 
    for json_file in json_files:
        try:
            # 簡化路徑處理
            relative_path = os.path.relpath(json_file, basedir)
            relative_path = relative_path.replace('\\', '/').replace('.json', '')
            
            with open(json_file, 'r') as f:
                json_data = json.load(f)
                signal = json_data['smoothed_data']
                original_sample_rate = json_data.get('sample_rate', 100)
                x_points = json_data['x_points']
                
                if not signal or not x_points:
                    print(f'Warning: Empty signal or x_points in {json_file}')
                    continue
                    
                latent_vector_list = predict_latent_vector_list(
                    model, signal, original_sample_rate, x_points, target_len=100
                )
                
                if latent_vector_list:  # 確保有結果才保存
                    encoded_data[relative_path] = np.array(latent_vector_list)
                
        except Exception as e:
            print(f'Error in processing {json_file}: {e}')
            continue
            
    return encoded_data

def predict_latent_vector_list(model, signal, sample_rate, peaks, target_len=100):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # input(f'signal: {signal}')
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
    # print(f'peaks: {peaks}')
    # 逐拍重建
    latent_vector_list = []
    for i in range(len(peaks) - 1):
        start_idx = peaks[i]
        end_idx = peaks[i + 1]
        
        # 添加邊界檢查
        if start_idx >= len(signal) or end_idx > len(signal):
            print(f'Warning: Invalid peak indices: {start_idx}, {end_idx} for signal length {len(signal)}')
            continue
            
        pulse = signal[start_idx:end_idx]
        pulse_length = end_idx - start_idx

        if pulse_length > 1:
            try:
                # 插值到目標長度
                interp_func = scipy.interpolate.interp1d(
                    np.arange(pulse_length), 
                    pulse, 
                    kind='linear', 
                    fill_value="extrapolate"
                )
                pulse_resampled = interp_func(np.linspace(0, pulse_length - 1, target_len))
                
                # 確保數據類型正確
                pulse_tensor = torch.tensor(pulse_resampled, dtype=torch.float32).unsqueeze(0).to(device)

                with torch.no_grad():
                    _, latent_vector = model(pulse_tensor)
                    latent_vector = latent_vector.squeeze().cpu().numpy()
                    
                    # 確保向量維度正確
                    if len(latent_vector.shape) > 1:
                        print(f'Warning: Unexpected latent vector shape: {latent_vector.shape}')
                        continue
                        
                    latent_vector = np.concatenate([
                        latent_vector, 
                        np.array([pulse_length/100])
                    ])
                    latent_vector_list.append(latent_vector)
                    
            except Exception as e:
                print(f'Error processing pulse {i}: {e}')
                continue
    
    # 確保有足夠的向量進行差異分析
    if len(latent_vector_list) < 2:
        print('Warning: Not enough vectors for difference analysis')
        return latent_vector_list
        
    #計算前後latent_vector之間的相似程度
    similarity_list = []
    distance_list = []
    diff_vectors = []
    for i in range(len(latent_vector_list) - 1):
        this_vec = latent_vector_list[i]
        next_vec = latent_vector_list[i + 1]

        similarity = np.dot(this_vec, next_vec) / (np.linalg.norm(this_vec) * np.linalg.norm(next_vec))
        distance = np.linalg.norm(this_vec - next_vec)
        diff_vector = next_vec - this_vec

        similarity_list.append(similarity)
        distance_list.append(distance)
        diff_vectors.append(diff_vector)

    # for i, (sim, dist, diff) in enumerate(zip(similarity_list, distance_list, diff_vectors)):
    #     print(f'Pulse {i} to Pulse {i+1} Diff Vector:{diff}, Norm: {np.linalg.norm(diff):.4f}')
    norms = [np.linalg.norm(l) for l in diff_vectors]
    print(f'norm of diff vectors: {norms}')
    #print norm of latent vectors
    norms = [np.linalg.norm(l) for l in latent_vector_list]
    print(f'norm of latent_vector_list: {norms}')
    return latent_vector_list

def save_encoded_data(encoded_data, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for path, vectors in encoded_data.items():
        # 創建合法的文件名
        safe_path = path.replace('/', '_').replace('\\', '_')
        output_path = os.path.join(output_dir, f'{safe_path}.h5')
        
        try:
            with h5py.File(output_path, 'w') as f:
                f.create_dataset('data', data=vectors)
                f.attrs['original_path'] = path
                
        except Exception as e:
            print(f'Error saving file {output_path}: {e}')

def analyze_diff_vectors(encoded_data):
    all_diff_vectors = []
    for vectors in encoded_data.values():
        diff_vectors = np.diff(vectors, axis=0)
        all_diff_vectors.extend(diff_vectors)
    
    all_diff_vectors = np.array(all_diff_vectors)
    
    # 执行PCA
    pca = PCA()
    pca.fit(all_diff_vectors)
    
    # 计算累积解释方差比
    cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
    
    # 绘制累积解释方差比
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(cumulative_variance_ratio) + 1), cumulative_variance_ratio, 'bo-')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance Ratio')
    plt.title('PCA Analysis of Diff Vectors')
    plt.grid(True)
    plt.show()
    
    # 找出解释95%方差所需的维度数
    n_components_95 = np.argmax(cumulative_variance_ratio >= 0.99) + 1
    print(f"Number of components needed to explain 95% of the variance: {n_components_95}")
    
    # 分析每个主成分的贡献
    print("\nTop 10 principal components and their explained variance ratio:")
    for i in range(20):
        print(f"PC{i+1}: {pca.explained_variance_ratio_[i]:.4f}")
    
    return pca, n_components_95, all_diff_vectors

def analyze_principal_components(pca, n_components):
    print("\nTop 5 principal components:")
    for i in range(5):
        print(f"PC{i+1}:")
        print(pca.components_[i])
        print("---")

def visualize_projections(all_diff_vectors, pca):
    projected = pca.transform(all_diff_vectors)
    plt.figure(figsize=(12, 4))
    
    plt.subplot(131)
    plt.scatter(projected[:, 0], projected[:, 1], alpha=0.1)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('Projection on PC1 and PC2')
    
    plt.subplot(132)
    plt.scatter(projected[:, 0], projected[:, 2], alpha=0.1)
    plt.xlabel('PC1')
    plt.ylabel('PC3')
    plt.title('Projection on PC1 and PC3')
    
    plt.subplot(133)
    plt.scatter(projected[:, 1], projected[:, 2], alpha=0.1)
    plt.xlabel('PC2')
    plt.ylabel('PC3')
    plt.title('Projection on PC2 and PC3')
    
    plt.tight_layout()
    plt.show()

def analyze_reconstruction_error(all_diff_vectors, pca, n_components_range):
    errors = []
    for n in n_components_range:
        pca_n = PCA(n_components=n)
        projected = pca_n.fit_transform(all_diff_vectors)
        reconstructed = pca_n.inverse_transform(projected)
        error = np.mean(np.sum((all_diff_vectors - reconstructed)**2, axis=1))
        errors.append(error)
    
    plt.figure(figsize=(10, 6))
    plt.plot(n_components_range, errors, 'bo-')
    plt.xlabel('Number of Components')
    plt.ylabel('Mean Squared Reconstruction Error')
    plt.title('Reconstruction Error vs. Number of Components')
    plt.grid(True)
    plt.show()