import torch
import numpy as np
import json
import os
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from modelBPclassify import LSTMClassifier
import sys
# 假設 label_map 和 LSTMClassifier 已經定義好
label_map = {'high': 0, 'middle': 1, 'low': 2}
reverse_label_map = {v: k for k, v in label_map.items()}

class PulseDataset(Dataset):
    def __init__(self, json_file):
        self.data = []
        self.load_data(json_file)

    def load_data(self, json_file):
        with open(json_file, 'r') as f:
            data = json.load(f)
            smoothed_data = data['smoothed_data']
            x_points = data['x_points']
            
            for i in range(len(x_points) - 1):
                start, end = x_points[i], x_points[i+1]
                pulse = smoothed_data[start:end]
                signal_type = np.zeros_like(pulse, dtype=int)  # 假設全部類型都是0
                pulse_data = np.stack([pulse, signal_type], axis=1)
                self.data.append(torch.tensor(pulse_data, dtype=torch.float32))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def predict(model, dataset, device):
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)  # 每次載入一拍
    model.eval()
    predictions = []
    with torch.no_grad():
        for data in dataloader:
            lengths = [len(data[0])]
            data_padded = pad_sequence(data, batch_first=True).to(device)
            output = model(data_padded, lengths)
            _, predicted = torch.max(output, 1)
            predictions.append(reverse_label_map[predicted.item()])
    return predictions

def main():
    json_file = sys.argv[1]
    model_path = 'bp_classifier2.pth'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    dataset = PulseDataset(json_file)
    model = LSTMClassifier(input_dim=2, hidden_dim=128, num_layers=2, num_classes=len(label_map))
    model.to(device)
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path,map_location = self.device))
        print("Model loaded successfully.")
    else:
        print("Model file not found.")
        return
    
    results = predict(model, dataset, device)
    print("Prediction results per pulse:", results)

if __name__ == '__main__':
    main()
