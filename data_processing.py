'''
用於處理頭＋手的資料整理



'''

import numpy as np
import os 
import torch
import json
import scipy
import random
import shutil
import re
import matplotlib.pyplot as plt


import os
import shutil

def find_and_copy_json_files(src_folder, dest_folder):
    print("in find_and_copy_json_files")
    # 遍歷 src_folder 中的子資料夾
    for subfolder_name in os.listdir(src_folder):
        print(subfolder_name)
        subfolder_path = os.path.join(src_folder, subfolder_name)
        
        # 確認是否為資料夾
        if not os.path.isdir(subfolder_path):
            continue

        # 建立時間的計數器
        time_count = {}
        for file_name in os.listdir(subfolder_path):
            if file_name.endswith('.json'):
                # 使用 split 擷取時間部分
                time_key = file_name.split(',')[0].strip()
                print("time_key:",time_key)
                if time_key in time_count:
                    time_count[time_key] += 1
                else:
                    time_count[time_key] = 1

        # 找出重複兩次的時間
        repeated_times = [time for time, count in time_count.items() if count == 2]
        
        # 如果有符合的重複時間，將其對應的 JSON 檔案複製到新的資料夾
        if repeated_times:
            #print("有符合的重複時間:",dest_subfolder) 
            dest_subfolder = os.path.join(dest_folder, subfolder_name)
            os.makedirs(dest_subfolder, exist_ok=True)

            for file_name in os.listdir(subfolder_path):
                # 使用 split 擷取時間部分
                time_key = file_name.split(',')[0].strip()
                if time_key in repeated_times:
                    # 確認目標資料夾中是否已存在該檔案
                    dest_file_path = os.path.join(dest_subfolder, file_name)
                    if not os.path.exists(dest_file_path):
                        # 複製檔案到目的資料夾
                        shutil.copy(os.path.join(subfolder_path, file_name), dest_subfolder)
                        print(f"Copied {file_name} to {dest_subfolder}")
                    else:
                        print(f"{file_name} already exists in {dest_subfolder}, skipping.")


def read_smoothed_data_and_x_points(dest_folder):
    # 建立一個用來存放結果的字典
    data_dict = {}

    # 遍歷 head_hand_DB 資料夾中的每個子資料夾
    for subfolder_name in os.listdir(dest_folder):
        subfolder_path = os.path.join(dest_folder, subfolder_name)

        # 確認是否為資料夾
        if not os.path.isdir(subfolder_path):
            continue

        # 建立時間的對應列表
        time_to_files = {}
        for file_name in os.listdir(subfolder_path):
            if file_name.endswith('.json'):
                # 使用 split 擷取時間部分
                time_key = file_name.split(',')[0].strip()
                if time_key not in time_to_files:
                    time_to_files[time_key] = []
                time_to_files[time_key].append(file_name)

        # 遍歷時間對應列表
        for time_key, files in time_to_files.items():
            if len(files) == 2:  # 確保有兩個檔案
                smoothed_data_list = []
                x_points_list = []

                # 讀取每個檔案中的 smoothed_data 和 x_points
                for file_name in files:
                    file_path = os.path.join(subfolder_path, file_name)
                    try:
                        with open(file_path, 'r') as json_file:
                            data = json.load(json_file)
                            smoothed_data = data.get('smoothed_data', [])
                            x_points = data.get('x_points', [])

                            smoothed_data_list.append(smoothed_data)
                            x_points_list.append(x_points)
                    except json.JSONDecodeError:
                        print(f"Skipping {file_name}: Invalid JSON format.")
                        continue

                # 如果成功讀取到資料，則將結果儲存到 data_dict
                if smoothed_data_list and x_points_list:
                    data_dict[(subfolder_name, time_key)] = {
                        'smoothed_data': smoothed_data_list,
                        'x_points': x_points_list
                    }

    return data_dict






if __name__ == '__main__':
    src_folder = 'labeled_DB'
    dest_folder = 'head_hand_DB' 

    # 將頭手資料從labeled_DB 抓出，並且複製到head_hand_DB
    # find_and_copy_json_files(src_folder, dest_folder)


    # 呼叫函式並獲取資料
    result = read_smoothed_data_and_x_points(dest_folder)
    
    # 打印結果
    for key, value in result.items():
        subfolder, time_key = key
        if subfolder == '4':
            print(f"Subfolder: {subfolder}, Time: {time_key}")
            #print("Smoothed Data:", value['smoothed_data'])
            if len(value['x_points']) > 1:
                print("X Points:", len(value['x_points'][0]),len(value['x_points'][1]))
                #print(value['x_points'])
                # xx interval


                # 檢查 smoothed_data 和 x_points 是否有資料
                if value['smoothed_data'] and len(value['smoothed_data']) > 1:
                    # 取得 smoothed_data 和 x_points
                    smoothed_data0 = value['smoothed_data'][0]
                    smoothed_data1 = value['smoothed_data'][1]
                    x_points0 = value['x_points'][0]
                    x_points1 = value['x_points'][1]

                    #XX interval
                    if len(x_points0) > 1:
                        result_array0 = np.array([x_points0[n+1] - x_points0[n]
                                                for n in range(len(x_points0) - 1)])
                        print("xx interval 0:", result_array0)
                        result_array1 = np.array([x_points1[n+1] - x_points1[n]
                                                for n in range(len(x_points1) - 1)])
                        print("xx interval 1:", result_array1)

                        #difference_array = result_array0 - result_array1
                        #print("difference_array:",difference_array)

                        # 繪製圖表
                        plt.figure(figsize=(12, 6))
                        plt.plot(result_array0, label='xx interval 0', marker='o')
                        plt.plot(result_array1, label='xx interval 1', marker='x')
                        #plt.plot(difference_array,label='difference', marker='')
                        plt.title(f'xx interval - {subfolder}, {time_key}')
                        plt.xlabel('Index')
                        plt.ylabel('Difference')
                        plt.legend()
                        plt.grid()
                        plt.show()
                    else:
                        print("Not enough x_points to perform the calculation.")
                    

                    #頭手相減 
                    #確保 x_points 的索引在 smoothed_data 的範圍內
                    # if all(x < len(smoothed_data0) for x in x_points0) and all(x < len(smoothed_data1) for x in x_points1):
                    #     # 計算 smoothed_data 的差值
                    #     result_array = np.array([smoothed_data0[x] - smoothed_data1[y] for x, y in zip(x_points0, x_points1)])
                    #     print("Result array:", result_array)
                    # else:
                    #     print("Some x_points are out of range for smoothed_data.")
                else:
                    print("Insufficient smoothed_data.")



            print("-" * 40)


