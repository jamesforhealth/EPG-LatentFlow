import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import json
import numpy as np
import math

map_dir = {
    '111' : 'high',
    '4' : 'middle',
    '42' : 'low',
}
label_map = {
    'high': 0,
    'middle': 1,
    'low': 2,
}

class PulseDataset(Dataset):
    def __init__(self, data_folder):
        self.data = []
        self.labels = []
        self.load_data(data_folder)

    def load_data(self, data_folder):
        for dir_name, label  in map_dir.items():
            dir_path = os.path.join(data_folder, dir_name)
            for file_name in os.listdir(dir_path):
                if file_name.endswith('.json'):
                    with open(os.path.join(dir_path, file_name), 'r') as f:
                        data = json.load(f)
                        smoothed_data = data['smoothed_data']
                        x_points = data['x_points']

                        # 切割 pulses
                        for i in range(len(x_points) - 1):
                            start, end = x_points[i], x_points[i+1]
                            pulse = smoothed_data[start:end]
                            signal_type = np.zeros_like(pulse, dtype=int)  # 這裡假設全部類型都是0，需要根據實際情況調整
                            pulse_data = np.stack([pulse, signal_type], axis=1)
                            self.data.append(torch.tensor(pulse_data, dtype=torch.float32))
                            self.labels.append(label_map[label])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, lengths):
        # 打包序列
        x_packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(x_packed)
        # 使用最後一個隱藏狀態
        output = self.fc(h_n[-1])
        return output
    
def collate_fn(batch):
    data, labels = zip(*batch)
    lengths = [len(x) for x in data]
    data_padded = nn.utils.rnn.pad_sequence(data, batch_first=True)
    labels = torch.tensor(labels, dtype=torch.long)
    return data_padded, labels, lengths

def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for data_padded, labels, lengths in dataloader:
        data_padded = data_padded.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(data_padded, lengths)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(dataloader)

def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for data_padded, labels, lengths in dataloader:
            data_padded = data_padded.to(device)
            labels = labels.to(device)
            outputs = model(data_padded, lengths)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    return running_loss / len(dataloader), accuracy


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    data_folder = 'point_labelled_DB'
    batch_size = 32
    num_epochs = 250
    learning_rate = 0.001
    input_size = 2  # 每個時間點有兩個特徵:訊號值和訊號點種類
    num_classes = len(label_map)
    d_model = 128
    nhead = 8
    num_layers = 6

    # model = TransformerClassifier(input_size, num_classes, d_model, nhead, num_layers).to(device)
    model = LSTMClassifier(input_size, d_model, 2, num_classes).to(device)
    # output the amount of model trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total number of model parameters: {trainable_params}')


    dataset = PulseDataset(data_folder)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)


    #load model
    if os.path.exists('bp_classifier2.pth'):
        model.load_state_dict(torch.load('bp_classifier2.pth'))
        print(f'load model from bp_classifier2.pth')
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        train_loss = train(model, train_dataloader, criterion, optimizer, device)
        test_loss, test_accuracy = evaluate(model, test_dataloader, criterion, device)
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')

    torch.save(model.state_dict(), 'bp_classifier2.pth')

if __name__ == '__main__':
    main()