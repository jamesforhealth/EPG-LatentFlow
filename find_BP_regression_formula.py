import pandas as pd
# import pymc3 as pm
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
# 從Google Sheets讀取資料
url = f'參數集合v2測試 - systolic_120.csv'
data = pd.read_csv(url)

#考慮帝82列之前的data就好然後沒有資料的列也不需要
data = data.iloc[:75, :]
data = data.dropna()

# 提取所需的特徵和目標變數
# features = ['xy interval', 'xz interval', 'xa interval', 'xb interval', 'xc interval', 
#             'yz interval', 'za interval', 'ab interval', 'bc interval', 
#             'y amplitude', 'z amplitude', 'a amplitude', 'b amplitude', 'c amplitude', 
#             'yz amplitude', 'za amplitude', 'ab amplitude', 'bc amplitude']

features = ['xy interval', 'xz interval', 'xa interval', 'xb interval', 'xc interval', 'y amplitude', 'z amplitude', 'a amplitude', 'b amplitude', 'c amplitude']
X = data[features].values
y_sbp = data['systolic'].values
y_dbp = data['diastolic'].values


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

print("Training Set MAD:")
print(f"SBP: {np.mean(np.abs(sbp_train_errors)):.2f}")
print(f"DBP: {np.mean(np.abs(dbp_train_errors)):.2f}")

print("Test Set MAD:")
print(f"SBP: {np.mean(np.abs(sbp_test_errors)):.2f}")
print(f"DBP: {np.mean(np.abs(dbp_test_errors)):.2f}")

print("Training Set Mean Error:")
print(f"SBP: {np.mean(sbp_train_errors):.2f}")
print(f"DBP: {np.mean(dbp_train_errors):.2f}")

print("Test Set Mean Error:")
print(f"SBP: {np.mean(sbp_test_errors):.2f}")
input(f"DBP: {np.mean(dbp_test_errors):.2f}")

# 對每個人進行個性化校正和增量訓練
personalized_models_sbp = {}
personalized_models_dbp = {}

for name in data['name'].unique():
    person_data = data[data['name'] == name].sort_values('Unnamed: 0')  # 按照原始順序排序
    X_person = person_data[features].values
    y_sbp_person = person_data['systolic'].values
    y_dbp_person = person_data['diastolic'].values
    
    # 將缺失值替換為0
    X_person = np.nan_to_num(X_person)
    y_sbp_person = np.nan_to_num(y_sbp_person)
    y_dbp_person = np.nan_to_num(y_dbp_person)
    
    # 創建多項式特徵
    X_poly_person = poly.transform(X_person)
    
    # 使用第一次量測的血壓值作為基準
    sbp_base = y_sbp_person[0]
    dbp_base = y_dbp_person[0]
    
    # 創建個人的收縮壓和舒張壓模型
    model_sbp_person = LinearRegression()
    model_dbp_person = LinearRegression()
    
    # 增量訓練模型
    for i in range(1, len(X_poly_person)):
        model_sbp_person.fit(X_poly_person[:i], y_sbp_person[:i])
        model_dbp_person.fit(X_poly_person[:i], y_dbp_person[:i])
        
        # 進行預測並校正
        sbp_pred = model_sbp_person.predict(X_poly_person[i].reshape(1, -1))[0]
        dbp_pred = model_dbp_person.predict(X_poly_person[i].reshape(1, -1))[0]
        
        sbp_corrected = sbp_pred + (sbp_base - model_sbp_person.predict(X_poly_person[0].reshape(1, -1))[0])
        dbp_corrected = dbp_pred + (dbp_base - model_dbp_person.predict(X_poly_person[0].reshape(1, -1))[0])
        
        print(f"Name: {name}, Measurement: {i}")
        print(f"SBP: Predicted = {sbp_pred:.2f}, Corrected = {sbp_corrected:.2f}, Actual = {y_sbp_person[i]:.2f}")
        print(f"DBP: Predicted = {dbp_pred:.2f}, Corrected = {dbp_corrected:.2f}, Actual = {y_dbp_person[i]:.2f}")
        print("---")
    
    personalized_models_sbp[name] = model_sbp_person
    personalized_models_dbp[name] = model_dbp_person
# # 列出訓練集的預測值、實際值以及誤差
# print("Training Set Predictions, Actual Values, and Errors:")
# for i, idx in enumerate(indices_train):
#     print(f"Name: {data.iloc[idx]['name']}, Row: {idx+2}")
#     print(f"SBP: Predicted = {y_sbp_train_pred[i]:.2f}, Actual = {y_sbp_train[i]:.2f}, Error = {y_sbp_train_pred[i] - y_sbp_train[i]:.2f}")
#     print(f"DBP: Predicted = {y_dbp_train_pred[i]:.2f}, Actual = {y_dbp_train[i]:.2f}, Error = {y_dbp_train_pred[i] - y_dbp_train[i]:.2f}")

# print("=" * 50)

# # 列出測試集的預測值、實際值以及誤差
# print("Test Set Predictions, Actual Values, and Errors:")
# for i, idx in enumerate(indices_test):
#     print(f"Name: {data.iloc[idx]['name']}, Row: {idx+2}")
#     print(f"SBP: Predicted = {y_sbp_test_pred[i]:.2f}, Actual = {y_sbp_test[i]:.2f}, Error = {y_sbp_test_pred[i] - y_sbp_test[i]:.2f}")
#     print(f"DBP: Predicted = {y_dbp_test_pred[i]:.2f}, Actual = {y_dbp_test[i]:.2f}, Error = {y_dbp_test_pred[i] - y_dbp_test[i]:.2f}")
# # 印出回歸模型的計算公式和係數
# print("\nRegression Formula:")
# feature_names = poly.get_feature_names_out(features)
# formula = "Systolic = "
# for i in range(len(model_sbp.coef_)):
#     if i == 0:
#         formula += f"{model_sbp.intercept_:.2f}"
#     else:
#         if model_sbp.coef_[i] >= 0:
#             formula += f" + {model_sbp.coef_[i]:.2f} * {feature_names[i]}"
#         else:
#             formula += f" - {abs(model_sbp.coef_[i]):.2f} * {feature_names[i]}"
# print(formula)

# formula = "Diastolic = "
# for i in range(len(model_dbp.coef_)):
#     if i == 0:
#         formula += f"{model_dbp.intercept_:.2f}"
#     else:
#         if model_dbp.coef_[i] >= 0:
#             formula += f" + {model_dbp.coef_[i]:.2f} * {feature_names[i]}"
#         else:
#             formula += f" - {abs(model_dbp.coef_[i]):.2f} * {feature_names[i]}"
# print(formula)

# individual_ids = data['name'].astype('category').cat.codes.values

# num_individuals = len(np.unique(individual_ids))
# num_features = len(features)

# # 定義層次貝葉斯模型
# with pm.Model() as hierarchical_model:
#     # 全局參數的先驗
#     mu_a = pm.Normal('mu_a', mu=0, sigma=1)
#     sigma_a = pm.HalfNormal('sigma_a', sigma=1)
    
#     mu_b = pm.Normal('mu_b', mu=0, sigma=1, shape=num_features)
#     sigma_b = pm.HalfNormal('sigma_b', sigma=1, shape=num_features)
    
#     # 個體參數的先驗
#     a = pm.Normal('a', mu=mu_a, sigma=sigma_a, shape=num_individuals)
#     b = pm.Normal('b', mu=mu_b, sigma=sigma_b, shape=(num_individuals, num_features))
    
#     # 觀測資料的似然
#     sigma = pm.HalfNormal('sigma', sigma=1)
#     y_pred = a[individual_ids] + pm.math.dot(b[individual_ids], X.T)
#     y_obs = pm.Normal('y_obs', mu=y_pred, sigma=sigma, observed=y)

#     # 推斷
#     trace = pm.sample(2000, tune=2000, cores=1)

# # 預測並印出每個個體的預測值和實際值
# for individual_id in np.unique(individual_ids):
#     individual_name = data['name'][data['name'].astype('category').cat.codes == individual_id].iloc[0]
#     individual_data = data[data['name'] == individual_name]
    
#     for _, row in individual_data.iterrows():
#         SBPo = row['diastolic']
#         Ao = row[features].values
        
#         Ae = row[features].values  # 假設 Ae 與 Ao 相同,你可以根據實際情況修改
        
#         # 提取此個體的參數後驗均值
#         a_mean = trace['a'][:, individual_id].mean()
#         b_mean = trace['b'][:, individual_id, :].mean(axis=0)
        
#         # 進行預測
#         SBPe_pred = a_mean + np.dot(b_mean, Ae)
        
#         print(f"Individual: {individual_name}")
#         print(f"Actual SBPe: {SBPo:.2f}")
#         print(f"Predicted SBPe: {SBPe_pred:.2f}")
#         print(f"Error: {SBPe_pred - SBPo:.2f}")
#         print("------------------------")