import os
import json
import shutil
import numpy as np
import sys
def analyze_signal(data, sample_rate):
    window_size = int(2 * sample_rate)  # 將窗口大小轉換為整數
    amplitudes = []
    for i in range(0, len(data)-sample_rate, sample_rate):
        window = data[i:i+window_size]
        amplitude = max(window) - min(window)
        amplitudes.append(amplitude)
    return amplitudes

def is_stable(amplitudes):
    if not amplitudes:
        return False
    max_amplitude = max(amplitudes)
    min_amplitude = min(amplitudes)
    return max_amplitude <= 1.5 * min_amplitude

def copy_stable_files(src_dir, dest_dir):
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    for root, dirs, files in os.walk(src_dir):
        for file in files:
            if file.endswith('.json'):
                src_path = os.path.join(root, file)
                with open(src_path, 'r') as f:
                    try:
                        data = json.load(f)
                        if 'sample_rate' in data and 'smoothed_data' in data:
                            sample_rate = int(data['sample_rate'])  # 將 sample_rate 轉換為整數
                            smoothed_data = data['smoothed_data']
                            amplitudes = analyze_signal(smoothed_data, sample_rate)
                            if is_stable(amplitudes):
                                rel_path = os.path.relpath(src_path, src_dir)
                                dest_path = os.path.join(dest_dir, rel_path)
                                os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                                shutil.copy2(src_path, dest_path)
                                print(f"Copied: {rel_path}")
                    except json.JSONDecodeError:
                        print(f"Error decoding JSON: {src_path}")
                    except Exception as e:
                        print(f"Error processing file {src_path}: {str(e)}")

def analyze_file(path):
    with open(path, 'r') as f:
        data = json.load(f)
        if 'sample_rate' in data and 'smoothed_data' in data:
            sample_rate = int(data['sample_rate'])  # 將 sample_rate 轉換為整數
            smoothed_data = data['smoothed_data']
            amplitudes = analyze_signal(smoothed_data, sample_rate)
            print(amplitudes)
            return is_stable(amplitudes)
    return False

# 使用示例
src_dir = 'labeled_DB'
dest_dir = 'wearing_consistency'
# copy_stable_files(src_dir, dest_dir)

analyze_file(sys.argv[1])