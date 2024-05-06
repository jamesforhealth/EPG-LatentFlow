import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

url = '參數集合v2_test - systolic_120.csv'
data = pd.read_csv(url)

# 提取所需的特徵和目標變數
sbp_features = ['y amplitude', 'z amplitude', 'a amplitude'] #'xa interval', 
dbp_features = ['xa interval', 'xb interval', 'xc interval'] #'xy interval', 'xz interval', , 'y amplitude', 'z amplitude', 'a amplitude', 'b amplitude', 'c amplitude']
MAD_map = {}
# 對每個人進行個人化校正和預測
formula_map = {}
for name in data['name'].unique():
    person_data = data[data['name'] == name].reset_index(drop=True)
    
    # 取第一筆量測作為基準量測
    base_data = person_data.iloc[0]
    sbp_base = base_data['systolic']
    dbp_base = base_data['diastolic']
    
    # 計算特徵差值
    X_sbp_diff = person_data[sbp_features].diff().dropna()
    X_dbp_diff = person_data[dbp_features].diff().dropna()
    
    # 計算目標變數差值
    y_sbp_diff = person_data['systolic'].diff().dropna()
    y_dbp_diff = person_data['diastolic'].diff().dropna()
    
    # 創建線性回歸模型
    sbp_model = LinearRegression()
    dbp_model = LinearRegression()
    
    # 訓練模型
    sbp_model.fit(X_sbp_diff, y_sbp_diff)
    dbp_model.fit(X_dbp_diff, y_dbp_diff)
    
    # 儲存校正公式
    sbp_formula = f"SBPe - SBPo = {sbp_model.intercept_:.2f}"
    for i, feature in enumerate(sbp_features):
        sbp_formula += f" + {sbp_model.coef_[i]:.2f}({feature} diff)"
    
    dbp_formula = f"DBPe - DBPo = {dbp_model.intercept_:.2f}"
    for i, feature in enumerate(dbp_features):
        dbp_formula += f" + {dbp_model.coef_[i]:.2f}({feature} diff)"
    
    formula_map[name] = {
        'sbp_formula': sbp_formula,
        'dbp_formula': dbp_formula
    }

     # 進行預測
    sbp_pred_diff = sbp_model.predict(X_sbp_diff)
    dbp_pred_diff = dbp_model.predict(X_dbp_diff)
    
    # 還原預測值
    sbp_pred = sbp_base + np.cumsum(sbp_pred_diff)
    dbp_pred = dbp_base + np.cumsum(dbp_pred_diff)
    
    # 計算誤差和MAD
    sbp_errors = person_data['systolic'].iloc[1:] - sbp_pred
    dbp_errors = person_data['diastolic'].iloc[1:] - dbp_pred
    sbp_mad = np.mean(np.abs(sbp_errors))
    dbp_mad = np.mean(np.abs(dbp_errors))

        # 儲存結果
    MAD_map[name] = {
        'sbp_errors': sbp_errors,
        'dbp_errors': dbp_errors,
        'sbp_mad': sbp_mad,
        'dbp_mad': dbp_mad
    }
    
    # 印出每次量測的推測誤差
    print(f"Name: {name}")
    for i in range(len(sbp_errors)):
        print(f"Measurement {i+1}: SBP Error = {sbp_errors.iloc[i]:.2f}, DBP Error = {dbp_errors.iloc[i]:.2f}")
    print(f"MAD: SBP = {sbp_mad:.2f}, DBP = {dbp_mad:.2f}")
    print("---")

# 印出每個人的校正公式
print("Individual Correction Formulas:")
for name in formula_map:
    print(f"Name: {name}")
    print(formula_map[name]['sbp_formula'])
    print(formula_map[name]['dbp_formula'])
    print("---")

# 印出每個人的MAD
print("Individual MAD:")
for name in MAD_map:
    print(f"Name: {name}, SBP MAD = {MAD_map[name]['sbp_mad']:.2f}, DBP MAD = {MAD_map[name]['dbp_mad']:.2f}")
