'''
from ios app json format to DB format

''' 

import os
import json
import matplotlib.pyplot as plt

def merge_json_files(input_folder, output_file):
    merged_data = []

    # 遍歷資料夾中的所有 JSON 檔案
    for filename in os.listdir(input_folder):
        if filename.endswith('.json'):
            file_path = os.path.join(input_folder, filename)
            # 讀取每個 JSON 檔案
            with open(file_path, 'r') as f:
                data = json.load(f)
                # 將讀取到的資料加入合併的列表中
                merged_data.extend(data)

    # 將合併後的資料寫入新的 JSON 檔案
    with open(output_file, 'w') as f:
        json.dump(merged_data, f, indent=4)

# 使用範例
# input_folder = '25_EPG-test _2024-10-21 11:37:21'  # 替換成你的資料夾路徑
# output_file = 'merged_data.json'  # 合併後的 JSON 檔案名稱
# merge_json_files(input_folder, output_file)

# print(f'所有 JSON 檔案已成功合併並儲存至 {output_file}')



def check_json_count(file_path):
    # 讀取 JSON 檔案
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # 檢查資料數量
    count = len(data)
    print(f'合併後的 JSON 檔案中共有 {count} 個項目')

# 使用範例
# file_path = 'merged_data.json'  # 替換成你的 JSON 檔案路徑
# check_json_count(file_path)

def merge_and_combine_json(input_folder, session_file, output_file):
    merged_data = []

    # 遍歷資料夾中的所有 JSON 檔案
    for filename in os.listdir(input_folder):
        if filename.endswith('.json') and filename != session_file:
            file_path = os.path.join(input_folder, filename)
            # 讀取每個 timestamp 的 JSON 檔案
            with open(file_path, 'r') as f:
                data = json.load(f)
                # 合併 timestamp 的資料
                merged_data.extend(data)
    
    # 依照 timestamp 排序 merged_data
    merged_data = sorted(merged_data, key=lambda x: x['timestamp'])

    # 讀取 session_data.json
    session_file_path = os.path.join(input_folder, session_file)
    with open(session_file_path, 'r') as f:
        session_data = json.load(f)

    # 刪除 'system_info'
    session_data.pop('system_info', None)
    # 將合併後的資料加到 session_data.json 中
    session_data["raw_data"] = merged_data

    # 將結果寫入新的 JSON 檔案
    with open(output_file, 'w') as f:
        json.dump(session_data, f, indent=4)

    print(f'合併完成，結果已儲存至 {output_file}')


def process_all_folders(parent_folder, session_file, output_suffix='.json'):
    # 創建儲存輸出檔案的資料夾，如果不存在的話
    # if not os.path.exists(output_folder):
    #     os.makedirs(output_folder)
    


    # 遍歷大資料夾中的每個子資料夾
    for folder_name in os.listdir(parent_folder):
        folder_path = os.path.join(parent_folder, folder_name)
        # 確保是子資料夾而不是其他檔案
        if os.path.isdir(folder_path):
            # 設定輸出檔案名稱，並將其儲存到新的輸出資料夾
            output_folder = f'DB/{folder_name.split("_")[0]}'
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
                print(f'已創建資料夾：{output_folder}')



            output_file = os.path.join(output_folder,folder_name.split('_', 1)[1].replace(':', '-') + output_suffix)
            # 執行合併和合併的函式
            merge_and_combine_json(folder_path, session_file, output_file)
            print(f'{folder_name} 資料夾已處理完畢，結果儲存至 {output_file}')

# 使用範例
parent_folder = 'ios_data'  # 替換成大資料夾的路徑
session_file = 'session_data.json'  # session_data.json 的檔名
process_all_folders(parent_folder, session_file)



#檢查時間差
#从文件路径读取JSON数据

# file_path = 'DB/42/(2024-11-05 10-25-29),(EPG_L - 13mask 12).json'  # 替换为你的文件路径
# with open(file_path, 'r') as f:
#     data = json.load(f)

# # 提取时间戳
# timestamps = [entry["timestamp"] for entry in data["raw_data"]]

# # 计算时间差
# time_diffs = [(timestamps[i] - timestamps[i - 1])*1e-9 for i in range(1, len(timestamps))]
# time_diffs = time_diffs 

# # 绘制线图
# plt.plot(time_diffs[:40], marker='o', linestyle='-', color='b')
# plt.xlabel("Packet Index")
# plt.ylabel("Time Difference (s)")
# plt.title("Time Difference Between Consecutive Packets")
# plt.grid()
# plt.show()