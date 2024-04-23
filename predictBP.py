import os 
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json
import numpy as np
import sys
def load_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)

    smoothed_data = data['smoothed_data']
    x_points = data['x_points']
    y_points = data['y_points']
    z_points = data['z_points']
    a_points = data['a_points']
    b_points = data['b_points']
    c_points = data['c_points']

    seq_length = len(smoothed_data)
    signal_values = np.array(smoothed_data).reshape(1, seq_length, 1)
    signal_classes = np.zeros((1, seq_length), dtype=int)

    for point in x_points:
        signal_classes[0, point] = 1
    for point in y_points:
        signal_classes[0, point] = 2
    for point in z_points:
        signal_classes[0, point] = 3
    for point in a_points:
        signal_classes[0, point] = 4
    for point in b_points:
        signal_classes[0, point] = 5
    for point in c_points:
        signal_classes[0, point] = 6
    
    return signal_values, signal_classes


json_file = sys.argv[1]
signal_value, signal_class = load_json(json_file)


model = tf.keras.models.load_model('./model.keras')
# 根據資料來進行血壓預測
predicted_sbp, predicted_dbp = model.predict([signal_value, signal_class])
print(f"Predicted SBP: {predicted_sbp[0][0]}, DBP: {predicted_dbp[0][0]}")
