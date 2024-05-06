import pandas as pd
# import pymc3 as pm
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
# 從Google Sheets讀取資料
url = '參數集合v2測試 - systolic_120.csv'
url = '參數集合v2_test - systolic_120.csv'
data = pd.read_csv(url)

#考慮帝82列之前的data就好然後沒有資料的列也不需要
# data = data.iloc[:50, :]
# data = data.dropna()
#print data shape

# 提取所需的特徵和目標變數
# features = ['xy interval', 'xz interval', 'xa interval', 'xb interval', 'xc interval', 
#             'yz interval', 'za interval', 'ab interval', 'bc interval', 
#             'y amplitude', 'z amplitude', 'a amplitude', 'b amplitude', 'c amplitude', 
#             'yz amplitude', 'za amplitude', 'ab amplitude', 'bc amplitude']

features = ['xy interval', 'xz interval', 'xa interval', 'xb interval', 'xc interval', 'y amplitude', 'z amplitude', 'a amplitude', 'b amplitude', 'c amplitude']
X = data[features].values
y_sbp = data['systolic'].values
y_dbp = data['diastolic'].values
print(data.shape)
print(X.shape)
print(y_sbp.shape)
print(y_dbp.shape)

# 將缺失值替換為0
X = np.nan_to_num(X)
y_sbp = np.nan_to_num(y_sbp)
y_dbp = np.nan_to_num(y_dbp)

# input(f'X:{X}')
# input(f'y:{y}')
# 創建多項式特徵
poly = PolynomialFeatures(degree=1)  # 可以調整degree的值來改變多項式的次數
X_poly = poly.fit_transform(X)

# 將資料分為訓練集和測試集
X_train, X_test, y_sbp_train, y_sbp_test, y_dbp_train, y_dbp_test, indices_train, indices_test = train_test_split(X_poly, y_sbp, y_dbp, data.index, test_size=0.2, random_state=42)

# 創建並訓練收縮壓模型
model_sbp = LinearRegression()
model_sbp.fit(X_train, y_sbp_train)

# 創建並訓練舒張壓模型
model_dbp = LinearRegression()
model_dbp.fit(X_train, y_dbp_train)

# 在訓練集和測試集上進行預測
y_sbp_train_pred = model_sbp.predict(X_train)
y_sbp_test_pred = model_sbp.predict(X_test)
y_dbp_train_pred = model_dbp.predict(X_train)
y_dbp_test_pred = model_dbp.predict(X_test)

# 計算訓練集和測試集上的MAD和誤差平均值
sbp_train_errors = y_sbp_train_pred - y_sbp_train
dbp_train_errors = y_dbp_train_pred - y_dbp_train
sbp_test_errors = y_sbp_test_pred - y_sbp_test
dbp_test_errors = y_dbp_test_pred - y_dbp_test
print(f'np.mean(sbp_train_errors):{np.mean(sbp_train_errors)}, np.mean(dbp_train_errors):{np.mean(dbp_train_errors)}, np.mean(sbp_test_errors):{np.mean(sbp_test_errors)}, np.mean(dbp_test_errors):{np.mean(dbp_test_errors)}')
print("Training Set MAD:")
print(f"SBP: {np.mean(np.abs(sbp_train_errors - np.mean(sbp_train_errors))):.4f}")
print(f"DBP: {np.mean(np.abs(dbp_train_errors - np.mean(dbp_train_errors))):.4f}")

print("Test Set MAD:")
print(f"SBP: {np.mean(np.abs(sbp_test_errors - np.mean(sbp_test_errors))):.4f}")
print(f"DBP: {np.mean(np.abs(dbp_test_errors - np.mean(dbp_test_errors))):.4f}")

print("Training Set Mean Error:")
print(f"SBP: {np.mean(sbp_train_errors):.4f}")
print(f"DBP: {np.mean(dbp_train_errors):.4f}")

print("Test Set Mean Error:")
print(f"SBP: {np.mean(sbp_test_errors):.4f}")
print(f"DBP: {np.mean(dbp_test_errors):.4f}")

# 輸出每個人的MAD
print("=" * 50)
personalized_models_sbp = {}
personalized_models_dbp = {}
MAD_map = {}

# 列出訓練集的預測值、實際值以及誤差
print("Training Set Predictions, Actual Values, and Errors:")
print(f"indices_train: {indices_train}")
for i, idx in enumerate(indices_train):
    # print(f"d = {data.iloc[idx]}")
    name = data.iloc[idx]['name']
    if name not in MAD_map:
        MAD_map[name] = {'sbp_error': [], 'dbp_error': []}
    print(f"Name: {name}, Row: {idx+2}")
    print(f"SBP: Predicted = {y_sbp_train_pred[i]:.4f}, Actual = {y_sbp_train[i]:.4f}, Error = {y_sbp_train_pred[i] - y_sbp_train[i]:.4f}")
    print(f"DBP: Predicted = {y_dbp_train_pred[i]:.4f}, Actual = {y_dbp_train[i]:.4f}, Error = {y_dbp_train_pred[i] - y_dbp_train[i]:.4f}")
    MAD_map[name]['sbp_error'].append(np.abs(y_sbp_train_pred[i] - y_sbp_train[i]))
    MAD_map[name]['dbp_error'].append(np.abs(y_dbp_train_pred[i] - y_dbp_train[i]))
    
print("=" * 50)

# 列出測試集的預測值、實際值以及誤差
print("Test Set Predictions, Actual Values, and Errors:")
for i, idx in enumerate(indices_test):
    name = data.iloc[idx]['name']
    if name not in MAD_map:
        MAD_map[name] = {'sbp_error': [], 'dbp_error': []}
    print(f"Name: {name}, Row: {idx+2}")
    print(f"SBP: Predicted = {y_sbp_test_pred[i]:.4f}, Actual = {y_sbp_test[i]:.4f}, Error = {y_sbp_test_pred[i] - y_sbp_test[i]:.4f}")
    print(f"DBP: Predicted = {y_dbp_test_pred[i]:.4f}, Actual = {y_dbp_test[i]:.4f}, Error = {y_dbp_test_pred[i] - y_dbp_test[i]:.4f}")
    MAD_map[name]['sbp_error'].append(np.abs(y_sbp_test_pred[i] - y_sbp_test[i]))
    MAD_map[name]['dbp_error'].append(np.abs(y_dbp_test_pred[i] - y_dbp_test[i]))

# 計算每個人的MAD並排序
for name in MAD_map:
    mean_sbp_error = np.mean(MAD_map[name]['sbp_error'])
    mean_dbp_error = np.mean(MAD_map[name]['dbp_error'])
    # print(f'Name: {name}, Mean SBP Error: {mean_sbp_error:.4f}, Mean DBP Error: {mean_dbp_error:.4f}, sbp_error: {MAD_map[name]["sbp_error"]}, dbp_error: {MAD_map[name]["dbp_error"]}')
    MAD_map[name]['sbp_error'] = [np.abs(e - mean_sbp_error) for e in MAD_map[name]['sbp_error']]
    MAD_map[name]['dbp_error'] = [np.abs(e - mean_dbp_error) for e in MAD_map[name]['dbp_error']]
    MAD_sbp = np.mean(MAD_map[name]['sbp_error'])
    MAD_dbp = np.mean(MAD_map[name]['dbp_error'])

    print(f"Name: {name}, MAD SBP: {MAD_sbp:.4f}, MAD DBP: {MAD_dbp:.4f}")


# 印出回歸模型的計算公式和係數
print("\nRegression Formula:")
feature_names = poly.get_feature_names_out(features)
formula = "Systolic = "
for i in range(len(model_sbp.coef_)):
    if i == 0:
        formula += f"{model_sbp.intercept_:.4f}"
    else:
        if model_sbp.coef_[i] >= 0:
            formula += f" + {model_sbp.coef_[i]:.4f} * {feature_names[i]}"
        else:
            formula += f" - {abs(model_sbp.coef_[i]):.4f} * {feature_names[i]}"
print(formula)

formula = "Diastolic = "
for i in range(len(model_dbp.coef_)):
    if i == 0:
        formula += f"{model_dbp.intercept_:.4f}"
    else:
        if model_dbp.coef_[i] >= 0:
            formula += f" + {model_dbp.coef_[i]:.4f} * {feature_names[i]}"
        else:
            formula += f" - {abs(model_dbp.coef_[i]):.4f} * {feature_names[i]}"
print(formula)


# personalized_models_sbp = {}
# personalized_models_dbp = {}
# MAD_map = {}
# for name in data['name'].unique():
#     if name not in MAD_map:
#         MAD_map[name] = {
#             'sbp_error' : 0.0,
#             'dbp_error' : 0.0,
#             'count' : 0
#         }
#     person_data = data[data['name'] == name]
#     X_person = person_data[features].values
#     y_sbp_person = person_data['systolic'].values
#     y_dbp_person = person_data['diastolic'].values
    
#     # 將缺失值替換為0
#     X_person = np.nan_to_num(X_person)
#     y_sbp_person = np.nan_to_num(y_sbp_person)
#     y_dbp_person = np.nan_to_num(y_dbp_person)
    
#     # 創建多項式特徵
#     X_poly_person = poly.transform(X_person)
    
#     # 對每個人的每次量測進行增量訓練和測試
#     for i in range(len(X_poly_person)):
#         # 使用第i次量測的資料進行增量訓練
#         X_train_person = X_poly_person[i].reshape(1, -1)
#         y_sbp_train_person = y_sbp_person[i]
#         y_dbp_train_person = y_dbp_person[i]
        
#         # 使用通用模型進行預測
#         sbp_pred = model_sbp.predict(X_train_person)
#         dbp_pred = model_dbp.predict(X_train_person)
        
#         # 計算預測誤差
#         sbp_error = y_sbp_train_person - sbp_pred
#         dbp_error = y_dbp_train_person - dbp_pred
        
#         # 創建個人的修正模型
#         model_diff_sbp = LinearRegression()
#         model_diff_dbp = LinearRegression()
        
#         model_diff_sbp.fit(X_train_person, [sbp_error])
#         model_diff_dbp.fit(X_train_person, [dbp_error])
        
#         # 儲存個人的修正模型
#         personalized_models_sbp[name] = model_diff_sbp
#         personalized_models_dbp[name] = model_diff_dbp
        
#         # 使用其他量測資料進行測試
#         X_test_person = np.delete(X_poly_person, i, axis=0)
#         y_sbp_test_person = np.delete(y_sbp_person, i)
#         y_dbp_test_person = np.delete(y_dbp_person, i)
        
#         # 使用通用模型和個人修正模型進行預測
#         sbp_pred_test = model_sbp.predict(X_test_person) + model_diff_sbp.predict(X_test_person)
#         dbp_pred_test = model_dbp.predict(X_test_person) + model_diff_dbp.predict(X_test_person)
        
#         # 計算MAD
#         sbp_mad = np.mean(np.abs(sbp_pred_test - y_sbp_test_person))
#         dbp_mad = np.mean(np.abs(dbp_pred_test - y_dbp_test_person))
        
#         print(f"Person: {name}, Measurement: {i+1}")
#         print(f"SBP MAD: {sbp_mad:.4f}")
#         print(f"DBP MAD: {dbp_mad:.4f}")
#         print("---")
        
#         MAD_map[name]['sbp_error'] += sbp_mad
#         MAD_map[name]['dbp_error'] += dbp_mad
#         MAD_map[name]['count'] += 1

# # 輸出個人的平均MAD
# for name in MAD_map:
#     MAD_map[name]['sbp_error'] /= MAD_map[name]['count']
#     MAD_map[name]['dbp_error'] /= MAD_map[name]['count']
#     print(f"Name: {name}, Avg_MAD_SBP: {MAD_map[name]['sbp_error']:.4f}, Avg_MAD_DBP: {MAD_map[name]['dbp_error']:.4f}")
