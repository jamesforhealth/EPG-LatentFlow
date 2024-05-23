
import numpy as np
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