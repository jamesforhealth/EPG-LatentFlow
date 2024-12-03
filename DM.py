import os
import re
import numpy as np
import h5py
import pandas as pd
import math
import matplotlib.pyplot as pyplot
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score,calinski_harabasz_score
from scipy.spatial.distance import cdist
from scipy.sparse import diags
from scipy.linalg import svd  
from collections import defaultdict
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import json
import dash 
from dash import dcc
from dash import html
import plotly.graph_objs as go
import pandas as pd
import plotly.express as px
from dash.dependencies import Input, Output, State 
from dash import callback_context

from preprocessing import get_json_files

# Reduced diffusion map via Roseland method without knn
def DMapRoseland_redo(X, Dim, ep=None, m=None):
    """
    X: Data matrix, n-by-p where n is the number of points, and p is the dimension of each point
    Dim: The number of eigenvectors, reducing R^p to R^Dim
    ep: Bandwidth for the kernel (optional)
    m: Number of landmark points (default: sqrt(n))
    """
    n, p = X.shape

    print('Step 1: Use k-means to find landmark.')
    if m is None:
        m = int(np.floor(np.sqrt(n)))
        print(f'(info) The number of landmarks is chosen: {m}.')
    
    # Step 1: K-means to find landmark points
    kmeans = KMeans(n_clusters=m, random_state=0).fit(X)
    refdex = kmeans.cluster_centers_
    
    # Step 2: Calculate Euclidean distance between all points and landmarks
    print('Step 2: Construct bandwidth of kernel function.')
    dist_ext = cdist(X, refdex)
    
    if ep is None:
        ep = 10  # Default scaling parameter if not provided
    
    # Gaussian kernel computation
    W_ext = np.exp(-10 * (dist_ext / ep) ** 2)
    print(np.matrix(np.sum(W_ext, axis=0)).shape)
    # Construct the diagonal normalization matrix D
    D = W_ext @ np.array(np.sum(W_ext, axis=0)).T
    V2 = D ** (-0.5)
    
    # Sparse diagonal matrix for normalization
    V2 = diags(V2, 0)
    
    # Compute the normalized transition matrix
    A_ext = V2 @ W_ext
    
    # Singular Value Decomposition (SVD)
    U_ext, S_ext, _ = np.linalg.svd(A_ext, full_matrices=False)
    
    # Normalize eigenvectors
    V = V2 @ U_ext
    # print(S_ext)
    S_ext = np.diag(S_ext)
    # Square the singular values to get eigenvalues
    S = np.diag(S_ext) ** 2
    print(U_ext.shape)
    # Sort the eigenvalues and eigenvectors in descending order
    idx = np.argsort(S)[::-1]
    S = S[idx]
    U = V[:, idx]
    
    # Keep only the top Dim dimensions
    U = U[:, 1:Dim+1]
    S = S[1:Dim+1]

    return U, S

def handle_DM():
    # 指定資料夾路徑
    #folder_path = 'latent_vectors_0727_outlier'
    folder_path = 'encoded_pulse_sequences_40'

    # 獲取資料夾中所有 h5 檔案的列表
    file_list = [f for f in os.listdir(folder_path) if f.endswith('.h5')]


    # 初始化一個字典來儲存所有檔案的資料
    all_data = {}
    num_pattern = r'x(\d+)_'

    # 遍歷所有檔案並讀取資料 
    for file_name in file_list:
        file_path = os.path.join(folder_path, file_name)
        file_data = {}
        with h5py.File(file_path, 'r') as f:
            for dataset_name in f.keys():
                data = f[dataset_name][:]
                file_data[dataset_name] = data
        clean_file_name = re.sub(r'\W|^(?=\d)', '_', file_name)
        all_data[clean_file_name] = file_data
        match = re.search(num_pattern, clean_file_name)
        if match:
            all_data[clean_file_name]['uniqueNumber'] = int(match.group(1))

    # 初始化 list 儲存名稱
    names = [re.sub(r'\W|^(?=\d)', '_', file_name) for file_name in file_list]
    s = []

    # 提取資料的大小
    for name in names:
        s.append(all_data[name]['data'].shape[0])

    # 提取名字中的數字部分
    numbers = []
    for name in names:
        match = re.search(num_pattern, name)
        if match:
            numbers.append(int(match.group(1)))  # 只有在找到匹配時才調用 group(1)

    unique_numbers = np.unique(numbers)

    # 使用預設的色彩映射來分配顏色
    colors = pyplot.get_cmap('tab10', len(unique_numbers)).colors
    color_map = {num: colors[i] for i, num in enumerate(unique_numbers)}

    # 初始化儲存資料的列表
    data_array = [all_data[name]['data'] for name in names]
    print("data_array",data_array[:5])

    # 合併資料
    n = 128 # 這個要換
    # for i, data in enumerate(data_array):
    #     print(f"Data at index {i} has shape: {data.shape}")
    data = np.vstack(data_array)
    # 檢查堆疊後的結果形狀
    print(f"堆疊後的數據形狀: {data.shape}")

    # 確認資料大小正確
    if data.shape[1] != 42:
        raise ValueError('尺寸錯誤！')

    # 擴散映射計算，替換為 DMapRoseland_redo
    Dim = 4
    U, S = DMapRoseland_redo(data, Dim, ep=math.floor(data.shape[1] / 2))
    #print("U,S:",U,S)

    # 建立 comments 列
    total_rows = sum(s)
    comments = np.empty(total_rows, dtype=object)
    current_row = 0
    for i, file_name in enumerate(file_list):
        num_rows = s[i]
        comments[current_row:current_row + num_rows] = file_name
        current_row += num_rows

    # 將註解和數據組合
    output_matrix = np.column_stack((U, comments))

    # 保存為 CSV 文件
    output_df = pd.DataFrame(output_matrix)
    output_df.to_csv('U_large_name_40.csv', index=False, header=False)

    # 檢查檔案是否存在
    if os.path.exists('U_large_name_40.csv'):
        print("檔案已成功儲存。")
        return True
        
        # # 再次讀取檔案以確認內容
        # saved_df = pd.read_csv('U_large_name.csv', header=None)
        # print(saved_df.head())  # 顯示檔案的前幾行
    else:
        print("檔案儲存失敗。")
        return False



def show_data():

    data_folder = 'labeled_DB'
    json_files = get_json_files(data_folder, exclude_keywords=[])

    # 讀取 CSV 檔案
    #df = pd.read_csv('U_large_name_with_outlier.csv', header=None, names=['X', 'Y', 'Z', 'W', 'id'])
    df = pd.read_csv('U_large_name_40.csv', header=None, names=['X', 'Y', 'Z', 'W', 'id'])

    # 計算 Silhouette Score and Calinski-Harabasz Score
    coordinates = df.iloc[:, :3].values  # 前三維當作空間中的座標 
    labels = df.iloc[:, 4].values  # 第五列當作群集標籤
    sil_score = silhouette_score(coordinates, labels)
    ch_score = calinski_harabasz_score(coordinates, labels)
    print(f'Silhouette Score: {sil_score}')
    print(f'Calinski-Harabasz Score: {ch_score}')


    # 確保 id 欄位是字符串類型
    df['id'] = df['id'].astype(str)

    #移除離群值
    # files_to_delete = ['20  (2023-11-28 10-08-28).h5', '68  (2023-11-27 10-14-33).h5', '75  (2023-12-18 15-26-11).h5', '82  (2024-07-23 12-48-14).h5', '91  (2024-07-10 15-11-05).h5', '91  (2024-07-11 15-44-18).h5', '91  (2024-07-11 15-35-30).h5', '91  (2024-07-11 16-00-18).h5', '91  (2024-07-11 15-48-46).h5', '92  (2024-07-11 16-16-21).h5', '93  (2024-07-11 15-44-18).h5', '111  (2024-01-22 10-11-23).h5', '113  (2024-06-18 16-55-31).h5', '117  (2024-01-10 10-48-17).h5', '114  (2024-03-18 10-54-56).h5', '146  (2024-04-19 16-11-32).h5', '146  (2024-04-17 17-20-42).h5', '146  (2024-04-19 16-14-12).h5']
    # df = df[~df['id'].isin(files_to_delete)]

    # 創建一個字典，用來將每個 id 對應到一個顏色
    unique_ids = df['id'].unique()
    unique_ids = sorted(unique_ids, key=lambda x: int(x.split('_')[0]))
    colors = px.colors.qualitative.G10
    color_map = {id: colors[i % len(colors)] for i, id in enumerate(unique_ids)}

    # 為每個數據點分配對應的顏色
    df['color'] = df['id'].map(color_map)
    df['index'] = df.groupby('id').cumcount() + 1
    print("df['index']",df['index'])


    # 標準化數據
    # pd.set_option('display.float_format', '{:.20f}'.format)
    # print("raw data:",df.head())
    # print("X min and max",df['X'].min(), df['X'].max(),df['id'][df['X'].idxmin()],df['id'][df['X'].idxmax()])
    # print("Y min and max",df['Y'].min(), df['Y'].max(),df['id'][df['Y'].idxmin()],df['id'][df['Y'].idxmax()])
    # print("Z min and max",df['Z'].min(), df['Z'].max(),df['id'][df['Z'].idxmin()],df['id'][df['Z'].idxmax()])

    # scaler = StandardScaler()
    # columns_to_standardize = ['X', 'Y', 'Z', 'W']
    # df[columns_to_standardize] = scaler.fit_transform(df[columns_to_standardize])


    #df[['X', 'Y', 'Z', 'W']] = df[['X', 'Y', 'Z', 'W']] * 10**20
    
    # print("standar:",df[:10])

    #camera setup
    set_camera = [1.25,1.25,1.25]















    app = dash.Dash(__name__)

    app.layout = html.Div([
        html.Div([
            html.Label('Search ID:()'),
            dcc.Input(
                id='id-text-input',
                type='text',
                placeholder='Enter text to search ID',
                style={'width': '300px'}
            ),
            html.Div([
                html.Label('Select IDs to Hide:'),
                dcc.Checklist(
                id='id-checkbox',
                options=[{'label': str(id), 'value': id} for id in unique_ids],
                value=[],
                labelStyle={'display': 'block'}),
            ],style={'height': '90vh','overflowY': 'scroll'}),

        ], style={'width': '20%', 'float': 'left', 'height': '95vh'}),

        html.Div([
            dcc.Graph(
                id='3d-scatter-plot',
                style={'height': '50vh'}
            ),
            dcc.Graph(
                id='2d-line-plot',
                style={'height': '50vh'},
            ),
            html.Button('set camera posiotion', id='camera-button'),  # 添加按钮
            html.P(
                id='explanation-text',
                style={'margin-x': '60px'} 
            )  
        ], style={'width': '75%', 'display': 'inline-block', 'height': '100vh', 'verticalAlign': 'top'}),
        #html.Button('set camera posiotion', id='camera-button'),  # 添加按钮
    ])


    @app.callback(
        Output('id-checkbox', 'value'),
        Input('id-text-input', 'value'),
        State('id-checkbox', 'options'),
        State('3d-scatter-plot', 'figure')
    )
    def update_checklist(input_value, options,figure):
        print("in update_checklist")
       
        if input_value:
            #print(len(input_value))
            if ',' in input_value:
                input_values = input_value.split(',')
                selected_ids = [option['value'] for option in options if option['value'].split('_')[0] not in input_values]
            else:
                if '(' in input_value:

                    selected_ids = [option['value'] for option in options if option['value'] not in input_value]
                    print("options",options[1]['value'])
                    print("selected_ids",selected_ids)
                
                else:
                    selected_ids = [option['value'] for option in options if option['value'].split('_')[0] not in input_value]
        else: 
            selected_ids = []
        
        if figure is not None:
            # 检查并获取相机设置
            if 'scene' in figure['layout'] and 'camera' in figure['layout']['scene']:
                camera = figure['layout']['scene']['camera']
                eye = camera.get('eye', {})  # 只获取eye参数
                #print("当前相机eye设置:", eye)  # 打印eye参数
                #print("当前相机设置:", camera)  # 打印相机设置

                set_camera[0] = eye['x']
                set_camera[1] = eye['y']
                set_camera[2] = eye['z']
            else:
                print("未找到相机设置")
        

        return selected_ids


    @app.callback(
        Output('3d-scatter-plot', 'figure'),
        Input('id-checkbox', 'value'),
        State('3d-scatter-plot', 'figure')  # 通过 State 获取当前图表的相机状态
    )
    def update_graph(hidden_ids, existing_figure):
        print("in update_graph")
    
        # 确保 hidden_ids 是字符串类型的列表
        hidden_ids = [str(id) for id in hidden_ids]

        # 过滤数据
        filtered_df = df[~df['id'].isin(hidden_ids)]    
        hidden_df = df[df['id'].isin(hidden_ids)]    


        # 获取当前相机位置，如果没有就用默认值
        existing_camera = existing_figure['layout']['scene']['camera'] if existing_figure else dict(
            eye=dict(x=set_camera[0], y=set_camera[1], z=set_camera[2]),
            center=dict(x=0, y=0, z=0),
            up=dict(x=0, y=0, z=1)
        )

        # 定义3D散点图
        figure = {
            'data': [
                go.Scatter3d(
                    x=filtered_df['X'],
                    y=filtered_df['Y'],
                    z=filtered_df['Z'],
                    mode='markers',
                    marker=dict(
                        size=5,
                        color=filtered_df['color'],
                        opacity=1
                    ),
                    #text=filtered_df['id'],
                    text=filtered_df.apply(lambda row: f"id: {row['id']}, index: {row['index']}", axis=1),  # 显示 id 和序号
                    hoverinfo='x+y+z+text',
                ),
                go.Scatter3d(
                    x=hidden_df['X'],
                    y=hidden_df['Y'],
                    z=hidden_df['Z'],
                    mode='markers',
                    marker=dict(
                        size=5,
                        color=hidden_df['color'],
                        opacity=0.008
                    ),
                    #text=hidden_df.apply(lambda row: f"id: {row['id']}, index: {row['index']}", axis=1),  # 显示 id 和序号
                    #hoverinfo='x+y+z+text'
                    hoverinfo='none'
                )
            ],
            'layout': go.Layout(
                title='3D Scatter Plot',
                scene=dict(
                    xaxis=dict(title='X Axis', range=[df['X'].min(), df['X'].max()]),
                    yaxis=dict(title='Y Axis', range=[df['Y'].min(), df['Y'].max()]),
                    zaxis=dict(title='Z Axis', range=[df['Z'].min(), df['Z'].max()]),
                    camera=existing_camera  # 使用现有的相机视角
                ),
                margin=dict(l=0, r=0, b=0, t=0)
            )
        }

        camera_info = figure['layout']['scene']['camera']
        print("Camera Position (eye):", camera_info['eye'])
        print("Camera Target (center):", camera_info['center'])
        print("Camera Up:", camera_info['up'])
        return figure

    # 捕捉点击事件
    @app.callback(
        Output('2d-line-plot', 'figure'),
        Input('3d-scatter-plot', 'clickData')  # 监听 clickData
    )
    def display_click_data(clickData):
        print("clickData",clickData)
        clicked_id = clickData['points'][0]['text']
        print("clicked_id",clicked_id)
        target_id = clicked_id.split(" ")[1]  
        print("target_id",target_id)
        target_timestamp = clicked_id.split("(")[1].split(")")[0]
        print("target_timestamp",target_timestamp)
        found_indices = [i for i, file in enumerate(json_files) if target_id in file and target_timestamp in file]
        print("found_indices",found_indices)
        print(str(json_files[found_indices[0]]))
        file_path = str(json_files[found_indices[0]])

        with open(file_path, 'r') as f:
            data = json.load(f)

        smoothed_data = data.get("smoothed_data", [])
        x_point = data.get("x_points",[])
        peak_x = np.array(x_point) * 0.01
        peak_y = np.array([smoothed_data[i] for i in x_point])
        peak_x_index = [f"Index: {i}" for i in range(len(peak_x))]

        fig = {
            'data': [
                go.Scatter(
                    x = np.linspace(0, len(smoothed_data) * 0.01, len(smoothed_data)),
                    y = np.array(smoothed_data),
                    mode='lines',
                    name='smooth data'
                ),
                go.Scatter(
                    x= peak_x,
                    y= peak_y,
                    mode='markers',
                    name='xpoint',
                    marker=dict(size=5, color='red'),
                    text=peak_x_index,
                    hoverinfo='x+y+text' 
                )
            ],
            'layout': go.Layout(
                title=file_path,
                xaxis=dict(title='X Axis'),
                yaxis=dict(title='Y Axis'),
            )
        }

        return fig
    
    @app.callback(
        Output('explanation-text', 'children'),  # 输出到 <p> 的内容
        Input('2d-line-plot', 'figure')  # 输入是图表的 figure
    )
    def update_explanation(figure):
        title = figure['layout']['title']['text']  # 获取图表的标题
        new_filename = f"{title.split('/')[1]}  {title.split('/')[2].split(',')[0]}.h5"
        return new_filename
   

            
            
            

         
        

    #button to set camera pos
    @app.callback(
        Output('camera-button', 'n_clicks'),  # 为了触发事件，但不实际使用输出
        Input('camera-button', 'n_clicks'),
        State('3d-scatter-plot', 'figure')
    )
    def print_camera_settings(n_clicks, figure):
        if n_clicks is not None and figure is not None:
            # 检查并获取相机设置
            if 'scene' in figure['layout'] and 'camera' in figure['layout']['scene']:
                camera = figure['layout']['scene']['camera']
                eye = camera.get('eye', {})  # 只获取eye参数
                print("当前相机eye设置:", eye)  # 打印eye参数
                #print("当前相机设置:", camera)  # 打印相机设置

                set_camera[0] = eye['x']
                set_camera[1] = eye['y']
                set_camera[2] = eye['z']
            else:
                print("未找到相机设置")
        return n_clicks  # 无实际输出，只是为了完成回调

    app.run_server(debug=True)



def delete_file():
    # 指定要删除的文件列表
    files_to_delete = ['20  (2023-11-28 10-08-28).h5', '68  (2023-11-27 10-14-33).h5', '75  (2023-12-18 15-26-11).h5', '82  (2024-07-23 12-48-14).h5', '91  (2024-07-10 15-11-05).h5', '91  (2024-07-11 15-44-18).h5', '91  (2024-07-11 15-35-30).h5', '91  (2024-07-11 16-00-18).h5', '91  (2024-07-11 15-48-46).h5', '92  (2024-07-11 16-16-21).h5', '93  (2024-07-11 15-44-18).h5', '111  (2024-01-22 10-11-23).h5', '113  (2024-06-18 16-55-31).h5', '117  (2024-01-10 10-48-17).h5', '114  (2024-03-18 10-54-56).h5', '146  (2024-04-19 16-11-32).h5', '146  (2024-04-17 17-20-42).h5', '146  (2024-04-19 16-14-12).h5']
    #files_to_delete = ['68  (2023-11-27 10-14-33).h5','114  (2024-03-18 10-54-56).h5']
    
    # 指定文件夹路径
    folder_path = 'latent_vectors_0727'
    # 遍历文件列表并删除文件
    for filename in files_to_delete:
        file_path = os.path.join(folder_path, filename)
        try:
            os.remove(file_path)
            print(f"已删除: {file_path}")
        except FileNotFoundError:
            print(f"文件未找到: {file_path}")
        except Exception as e:
            print(f"删除文件时出错: {file_path}, 错误: {e}")



if __name__ == '__main__':
    #res = handle_DM()
    

    # data_folder = 'labeled_DB'
    # json_files = get_json_files(data_folder, exclude_keywords=[])
    # json_id = [file.split("/")[1] for file in json_files]
    # json_note = [file.split("/")[2].split(",")[0].strip('()') for file in json_files]
    # combined = [f"{id_}  {note}" for id_, note in zip(json_id, json_note)]
    # print("json_id",json_id[:5])
    # print("json_note",json_note[:5])
    # print(combined[:5])
    # print(len(combined))

    show_data()
    
    
    #delete_file()

    



    

# 4  (2023-07-14 09-36-00).h5
# 1_(2024-10-16 16_36_20),(1_EPG - resting  - 公司).json.h5

        