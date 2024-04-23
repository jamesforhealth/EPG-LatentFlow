import os 
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, GlobalAveragePooling1D, Embedding, concatenate, Dense, BatchNormalization, Activation, Add, MaxPooling1D, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber
import json
import numpy as np
#定義一個dict對應檔名跟(SBP, DBP)標籤值
label_dict = {
    '(2024-02-16 15-42-29),(EPG - 120_77mmHg - 公司).json' : (120, 77),
    '(2024-02-16 15-48-04),(EPG - 115_76 - 公司).json' : (115, 76),
    '(2024-02-16 22-46-35),(108_67mmHg 臥姿 - EPG).json' : (108, 67),
    '(2024-04-11 13-10-18),(EPG_R 午餐前 122_81mmHg).json' : (122, 81),
    '(2024-01-19 16-25-11),(sys85 dia62 hr83 - EPG).json' : (85, 62),
    '(2024-01-25 16-34-11),( - EPG_left [103_75_78]).json' : (103, 75),
    '(2024-02-02 10-57-11),(100hz 93_59_83 - EPG).json' : (93, 59),
    '(2024-04-11 09-35-10),(EPG_L - 94_65_76 飯前 - 公司).json' : (94, 65),
    '(2024-04-11 10-01-59),(EPG_L - 早餐歺後 94_67mmHg).json' : (94, 67),
    '(2024-04-11 12-58-12),(EPG - 93_70_68 飯前 - 公司).json' : (93, 70),
    '(2024-04-12 11-10-42),(EPG - 95_69_78  - 公司).json' : (95, 69),
    '(2024-04-12 12-30-39),(EPG -  - 公司 91_76).json' : (91, 76),
    '(2024-04-12 16-13-10),(EPG -  - 公司 105_85).json' : (105, 85),
    '(2024-04-15 12-09-46),(EPG - 96_67_79 - 公司).json' : (96, 67),
    '(2024-02-02 16-13-28),(飯後3小時 - EPG 121_97).json' : (121, 97),
    '(2024-02-15 17-21-11),(EPG -  - 公司 131_98).json' : (131, 98),
    '(2024-02-16 16-01-09),(EPG -  - 公司 128_92).json' : (128, 92),
    '(2024-04-11 09-32-28),(EPG_L - 早上餐前節水 125_91mmHg).json' : (125, 91),
    '(2024-04-11 12-59-52),(EPG - 午餐前126_91_81 - 公司).json' : (126, 91)
}

def create_model(sample_rate, time_range, num_classes, embedding_dim):
    input_signal = Input(shape=(None, 1))
    input_class = Input(shape=(None,))

    kernel_size = int(sample_rate * time_range)

    # 訊號分支
    x = Conv1D(filters=64, kernel_size=kernel_size, padding='same')(input_signal)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv1D(filters=128, kernel_size=kernel_size, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(pool_size=2)(x)

    for filters in [128, 128, 128]:
        x = resnet_block(x, filters=filters, kernel_size=kernel_size)

    x = GlobalAveragePooling1D()(x)

    # 類別分支
    y = Embedding(num_classes, embedding_dim)(input_class)
    y = Conv1D(filters=128, kernel_size=kernel_size, padding='same')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = GlobalAveragePooling1D()(y)

    # 合併分支
    merged = concatenate([x, y])
    merged = Dense(512, activation='relu')(merged)
    merged = Dropout(0.5)(merged)
    merged = Dense(256, activation='relu')(merged)
    merged = Dropout(0.5)(merged)
    output_sbp = Dense(1, activation='linear', name='sbp')(merged)
    output_dbp = Dense(1, activation='linear', name='dbp')(merged)

    model = tf.keras.models.Model(inputs=[input_signal, input_class], outputs=[output_sbp, output_dbp])
    return model
def resnet_block(input_tensor, filters, kernel_size):
    x = Conv1D(filters=filters, kernel_size=kernel_size, padding='same')(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv1D(filters=filters, kernel_size=kernel_size, padding='same')(x)
    x = BatchNormalization()(x)
    x = Add()([x, input_tensor])
    x = Activation('relu')(x)
    return x

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
    
    return signal_values[0], signal_classes[0]

def load_data(data_folder):
    signal_values_list = []
    signal_classes_list = []
    sbp_values = []
    dbp_values = []
    file_names = []  # 新增一個列表來存儲檔名

    for root, dirs, files in os.walk(data_folder):
        for file in files:
            if file.endswith(".json"):
                file_path = os.path.join(root, file)
                signal_value, signal_class = load_json(file_path)
                signal_values_list.append(signal_value)  # 添加一維陣列而不是二維陣列
                signal_classes_list.append(signal_class)  # 添加一維陣列而不是二維陣列
                sbp_values.append(label_dict[file][0])
                dbp_values.append(label_dict[file][1])
                file_names.append(file)

    # 找到最長的訊號長度
    max_length = max(len(signal) for signal in signal_values_list)

    # 將所有訊號值padding到相同的長度
    signal_values_array = pad_sequences(signal_values_list, maxlen=max_length, dtype='float32')
    signal_classes_array = pad_sequences(signal_classes_list, maxlen=max_length, dtype='int32')

    # 調整訊號值的維度為(num_samples, max_length, 1)
    signal_values_array = signal_values_array.reshape(signal_values_array.shape[0], max_length, 1)

    sbp_values_array = np.array(sbp_values)
    dbp_values_array = np.array(dbp_values)

    return signal_values_array, signal_classes_array, sbp_values_array, dbp_values_array, file_names

# 設定參數
data_folder = "point_labelled_DB"
sample_rate = 100  # 假設採樣率為100 Hz
time_range = 0.1  # 考慮前後0.1秒的資訊
num_classes = 7  # 類別數量(normal,x,y,z,a,b,c)
embedding_dim = 16  # 嵌入維度

# 載入資料
signal_values, signal_classes, sbp_values, dbp_values, file_names = load_data(data_folder)

# 創建模型
model = create_model(sample_rate, time_range, num_classes, embedding_dim)

# 如果model.h5存在就載入模型
if os.path.exists('./model.keras'):
    model = tf.keras.models.load_model('./model.keras')

#確認GPU狀態以及確認模型跑在GPU上
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
print(tf.test.is_built_with_cuda())
print(tf.config.list_physical_devices('GPU'))
model.compile(optimizer=Adam(learning_rate=0.001), loss=Huber(delta=1.0))
    # 訓練模型
model.fit([signal_values, signal_classes], [sbp_values, dbp_values], epochs=300, batch_size=32, validation_split=0.1)

# 模型評估以及預測結果
loss = model.evaluate([signal_values, signal_classes], [sbp_values, dbp_values])
print(f"Loss: {loss}")

predictions = model.predict([signal_values, signal_classes])
print("Predictions:")
for i in range(len(predictions[0])):
    print(f"File: {file_names[i]}")
    print(f"SBP: {predictions[0][i][0]}, DBP: {predictions[1][i][0]}")
    print(f"True SBP: {sbp_values[i]}, True DBP: {dbp_values[i]}")
    print("---")
#保存模型
model.save('./model.keras')