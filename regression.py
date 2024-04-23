import pandas as pd
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
def calculate_correlations(data, feature, target):
    try:
        corr = data[[feature, target]].astype(float).corr(numeric_only=True).iloc[0, 1]
        print(f"Correlation between {feature} and {target}: {corr:.3f}")
    except (ValueError, IndexError):
        print(f"Failed to calculate correlation between {feature} and {target}")
data = pd.read_csv('參數集合 - 工作表1.csv')
selected_names = ['becca', 'shaleen', 'Andy']
data = data[data['name'].isin(selected_names)]

# 選擇特徵和目標變量
features = ['xz / xx interval', 'xz interval', 'xy /xx interval', 'xy interval', 'za amplitude', 'integral area',
            'Systolic volume Avg', 'Diastolic volume Avg', 'XX', 'inte height', 'ya amplitude', 'ya/xx interval',
            'z amplitude', 'yz amplitude', 'yz interval', 'yz R', 'area/xx', 's/xx', 'd/xx']
target_sys = 'systolic'
target_dia = 'diastolic'

# 檢查每一列是否包含缺失值
missing_values = data[features + [target_sys, target_dia]].isnull().any(axis=1)

# 選擇不包含缺失值的行
data_clean = data[~missing_values]

becca_data = data_clean[data_clean['name'] == 'becca']
shaleen_data = data_clean[data_clean['name'] == 'shaleen']
andy_data = data_clean[data_clean['name'] == 'Andy']

# 對每個子集應用每種回歸方法
for name, subset_data in [('becca', becca_data), ('shaleen', shaleen_data), ('andy', andy_data)]:
    subset_data['pulse_pressure'] = subset_data[target_sys].astype(float) - subset_data[target_dia].astype(float)
    
    print(f"\nResults for {name}:")
    
    for feature in features:
        print(f"\nFeature: {feature}")
        calculate_correlations(subset_data, feature, target_sys)
        calculate_correlations(subset_data, feature, target_dia)
        calculate_correlations(subset_data, feature, 'pulse_pressure')
    
    feature_stats = subset_data[features].agg(['mean', 'std'])
    # print("\nFeature - Mean and Standard Deviation:")
    # print(feature_stats)
    
    sys_stats = subset_data[target_sys].agg(['mean', 'std'])
    print("\nSystolic Pressure - Mean and Standard Deviation:")
    print(sys_stats)
    
    dia_stats = subset_data[target_dia].astype(float).agg(['mean', 'std'])
    print("Diastolic Pressure - Mean and Standard Deviation:")
    print(dia_stats)
    
    pp_stats = subset_data['pulse_pressure'].agg(['mean', 'std'])
    print("Pulse Pressure - Mean and Standard Deviation:")
    print(pp_stats)
    # X_subset = subset_data[features]
    # y_sys_subset = subset_data[target_sys]
    # y_dia_subset = subset_data[target_dia].astype(float)
    # y_pp_subset = y_sys_subset - y_dia_subset
    
    # X_subset_scaled = scaler.transform(X_subset)
    
    # for method_name, sys_model, dia_model, pp_model in regression_methods:
    #     print(f"\n{method_name}:")
        
    #     y_sys_pred = []
    #     y_dia_pred = []
    #     y_pp_pred = []
        
    #     for train_index, test_index in loocv.split(X_subset_scaled):
    #         X_train, X_test = X_subset_scaled[train_index], X_subset_scaled[test_index]
    #         y_sys_train, y_sys_test = y_sys_subset.iloc[train_index], y_sys_subset.iloc[test_index]
    #         y_dia_train, y_dia_test = y_dia_subset.iloc[train_index], y_dia_subset.iloc[test_index]
    #         y_pp_train, y_pp_test = y_pp_subset.iloc[train_index], y_pp_subset.iloc[test_index]
            
    #         if method_name == 'Polynomial Regression':
    #             X_train_poly = poly_features.fit_transform(X_train)
    #             X_test_poly = poly_features.transform(X_test)
    #             sys_model.fit(X_train_poly, y_sys_train)
    #             dia_model.fit(X_train_poly, y_dia_train)
    #             pp_model.fit(X_train_poly, y_pp_train)
    #             y_sys_pred.append(sys_model.predict(X_test_poly)[0])
    #             y_dia_pred.append(dia_model.predict(X_test_poly)[0])
    #             y_pp_pred.append(pp_model.predict(X_test_poly)[0])
    #         else:
    #             sys_model.fit(X_train, y_sys_train)
    #             dia_model.fit(X_train, y_dia_train)
    #             pp_model.fit(X_train, y_pp_train)
    #             y_sys_pred.append(sys_model.predict(X_test)[0])
    #             y_dia_pred.append(dia_model.predict(X_test)[0])
    #             y_pp_pred.append(pp_model.predict(X_test)[0])
        
    #     print("Systolic - MSE:", mean_squared_error(y_sys_subset, y_sys_pred))
    #     print("Systolic - R2 Score:", r2_score(y_sys_subset, y_sys_pred))
    #     print("Diastolic - MSE:", mean_squared_error(y_dia_subset, y_dia_pred))
    #     print("Diastolic - R2 Score:", r2_score(y_dia_subset, y_dia_pred))
    #     print("Pulse Pressure - MSE:", mean_squared_error(y_pp_subset, y_pp_pred))
    #     print("Pulse Pressure - R2 Score:", r2_score(y_pp_subset, y_pp_pred))




    # X = data_clean[features]
# y_sys = data_clean[target_sys]
# y_dia = data_clean[target_dia].astype(float)


# print(f'X: {X}')
# print(f'y_sys: {y_sys}')
# print(f'y_dia: {y_dia}')
# y_pp = y_sys - y_dia
# print(f'y_pp: {y_pp}')

# # 特徵縮放
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# # 留一交叉驗證
# loocv = LeaveOneOut()

# # 多項式回歸
# poly_features = PolynomialFeatures(degree=2)
# poly_reg_sys = LinearRegression()
# poly_reg_dia = LinearRegression()
# poly_reg_pp = LinearRegression()

# y_sys_pred_poly = []
# y_dia_pred_poly = []
# y_pp_pred_poly = []

# for train_index, test_index in loocv.split(X_scaled):
#     X_train, X_test = X_scaled[train_index], X_scaled[test_index]
#     y_sys_train, y_sys_test = y_sys.iloc[train_index], y_sys.iloc[test_index]
#     y_dia_train, y_dia_test = y_dia.iloc[train_index], y_dia.iloc[test_index]
#     y_pp_train, y_pp_test = y_pp.iloc[train_index], y_pp.iloc[test_index]
    
#     X_train_poly = poly_features.fit_transform(X_train)
#     X_test_poly = poly_features.transform(X_test)
    
#     poly_reg_sys.fit(X_train_poly, y_sys_train)
#     y_sys_pred_poly.append(poly_reg_sys.predict(X_test_poly)[0])
    
#     poly_reg_dia.fit(X_train_poly, y_dia_train)
#     y_dia_pred_poly.append(poly_reg_dia.predict(X_test_poly)[0])
    
#     poly_reg_pp.fit(X_train_poly, y_pp_train)
#     y_pp_pred_poly.append(poly_reg_pp.predict(X_test_poly)[0])

# # 決策樹回歸
# dt_reg_sys = DecisionTreeRegressor(random_state=42)
# dt_reg_dia = DecisionTreeRegressor(random_state=42)
# dt_reg_pp = DecisionTreeRegressor(random_state=42)

# y_sys_pred_dt = []
# y_dia_pred_dt = []
# y_pp_pred_dt = []

# for train_index, test_index in loocv.split(X_scaled):
#     X_train, X_test = X_scaled[train_index], X_scaled[test_index]
#     y_sys_train, y_sys_test = y_sys.iloc[train_index], y_sys.iloc[test_index]
#     y_dia_train, y_dia_test = y_dia.iloc[train_index], y_dia.iloc[test_index]
#     y_pp_train, y_pp_test = y_pp.iloc[train_index], y_pp.iloc[test_index]
    
#     dt_reg_sys.fit(X_train, y_sys_train)
#     y_sys_pred_dt.append(dt_reg_sys.predict(X_test)[0])
    
#     dt_reg_dia.fit(X_train, y_dia_train)
#     y_dia_pred_dt.append(dt_reg_dia.predict(X_test)[0])
    
#     dt_reg_pp.fit(X_train, y_pp_train)
#     y_pp_pred_dt.append(dt_reg_pp.predict(X_test)[0])

# # 評估模型性能
# print("Polynomial Regression:")
# print("Systolic - MSE:", mean_squared_error(y_sys, y_sys_pred_poly))
# print("Systolic - R2 Score:", r2_score(y_sys, y_sys_pred_poly))
# print("Diastolic - MSE:", mean_squared_error(y_dia, y_dia_pred_poly))
# print("Diastolic - R2 Score:", r2_score(y_dia, y_dia_pred_poly))
# print("Pulse Pressure - MSE:", mean_squared_error(y_pp, y_pp_pred_poly))
# print("Pulse Pressure - R2 Score:", r2_score(y_pp, y_pp_pred_poly))

# print("\nDecision Tree Regression:")
# print("Systolic - MSE:", mean_squared_error(y_sys, y_sys_pred_dt))
# print("Systolic - R2 Score:", r2_score(y_sys, y_sys_pred_dt))
# print("Diastolic - MSE:", mean_squared_error(y_dia, y_dia_pred_dt))
# print("Diastolic - R2 Score:", r2_score(y_dia, y_dia_pred_dt))
# print("Pulse Pressure - MSE:", mean_squared_error(y_pp, y_pp_pred_dt))
# print("Pulse Pressure - R2 Score:", r2_score(y_pp, y_pp_pred_dt))

# # 預測20筆資料的收縮壓、舒張壓和脈壓
# X_new = data_clean[features][:20]
# X_new_scaled = scaler.transform(X_new)
# X_new_poly = poly_features.transform(X_new_scaled)

# y_sys_pred_poly_new = poly_reg_sys.predict(X_new_poly)
# y_dia_pred_poly_new = poly_reg_dia.predict(X_new_poly)
# y_pp_pred_poly_new = poly_reg_pp.predict(X_new_poly)

# y_sys_pred_dt_new = dt_reg_sys.predict(X_new_scaled)
# y_dia_pred_dt_new = dt_reg_dia.predict(X_new_scaled)
# y_pp_pred_dt_new = dt_reg_pp.predict(X_new_scaled)

# print("\nPredictions for 20 samples:")
# print("Polynomial Regression:")
# print("Systolic:", y_sys_pred_poly_new)
# print("Diastolic:", y_dia_pred_poly_new)
# print("Pulse Pressure:", y_pp_pred_poly_new)

# print("\nDecision Tree Regression:")
# print("Systolic:", y_sys_pred_dt_new)
# print("Diastolic:", y_dia_pred_dt_new)
# print("Pulse Pressure:", y_pp_pred_dt_new)




# # SVR
# svr_sys = SVR(kernel='rbf', C=1.0, epsilon=0.1)
# svr_dia = SVR(kernel='rbf', C=1.0, epsilon=0.1)
# svr_pp = SVR(kernel='rbf', C=1.0, epsilon=0.1)

# y_sys_pred_svr = []
# y_dia_pred_svr = []
# y_pp_pred_svr = []

# for train_index, test_index in loocv.split(X_scaled):
#     X_train, X_test = X_scaled[train_index], X_scaled[test_index]
#     y_sys_train, y_sys_test = y_sys.iloc[train_index], y_sys.iloc[test_index]
#     y_dia_train, y_dia_test = y_dia.iloc[train_index], y_dia.iloc[test_index]
#     y_pp_train, y_pp_test = y_pp.iloc[train_index], y_pp.iloc[test_index]
    
#     svr_sys.fit(X_train, y_sys_train)
#     y_sys_pred_svr.append(svr_sys.predict(X_test)[0])
    
#     svr_dia.fit(X_train, y_dia_train)
#     y_dia_pred_svr.append(svr_dia.predict(X_test)[0])
    
#     svr_pp.fit(X_train, y_pp_train)
#     y_pp_pred_svr.append(svr_pp.predict(X_test)[0])

# # Random Forest Regression
# rf_reg_sys = RandomForestRegressor(n_estimators=100, random_state=42)
# rf_reg_dia = RandomForestRegressor(n_estimators=100, random_state=42)
# rf_reg_pp = RandomForestRegressor(n_estimators=100, random_state=42)

# y_sys_pred_rf = []
# y_dia_pred_rf = []
# y_pp_pred_rf = []

# for train_index, test_index in loocv.split(X_scaled):
#     X_train, X_test = X_scaled[train_index], X_scaled[test_index]
#     y_sys_train, y_sys_test = y_sys.iloc[train_index], y_sys.iloc[test_index]
#     y_dia_train, y_dia_test = y_dia.iloc[train_index], y_dia.iloc[test_index]
#     y_pp_train, y_pp_test = y_pp.iloc[train_index], y_pp.iloc[test_index]
    
#     rf_reg_sys.fit(X_train, y_sys_train)
#     y_sys_pred_rf.append(rf_reg_sys.predict(X_test)[0])
    
#     rf_reg_dia.fit(X_train, y_dia_train)
#     y_dia_pred_rf.append(rf_reg_dia.predict(X_test)[0])
    
#     rf_reg_pp.fit(X_train, y_pp_train)
#     y_pp_pred_rf.append(rf_reg_pp.predict(X_test)[0])

# # XGBoost Regression
# xgb_reg_sys = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
# xgb_reg_dia = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
# xgb_reg_pp = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)

# y_sys_pred_xgb = []
# y_dia_pred_xgb = []
# y_pp_pred_xgb = []

# for train_index, test_index in loocv.split(X_scaled):
#     X_train, X_test = X_scaled[train_index], X_scaled[test_index]
#     y_sys_train, y_sys_test = y_sys.iloc[train_index], y_sys.iloc[test_index]
#     y_dia_train, y_dia_test = y_dia.iloc[train_index], y_dia.iloc[test_index]
#     y_pp_train, y_pp_test = y_pp.iloc[train_index], y_pp.iloc[test_index]
    
#     xgb_reg_sys.fit(X_train, y_sys_train)
#     y_sys_pred_xgb.append(xgb_reg_sys.predict(X_test)[0])
    
#     xgb_reg_dia.fit(X_train, y_dia_train)
#     y_dia_pred_xgb.append(xgb_reg_dia.predict(X_test)[0])
    
#     xgb_reg_pp.fit(X_train, y_pp_train)
#     y_pp_pred_xgb.append(xgb_reg_pp.predict(X_test)[0])

# # 評估模型性能
# print("SVR:")
# print("Systolic - MSE:", mean_squared_error(y_sys, y_sys_pred_svr))
# print("Systolic - R2 Score:", r2_score(y_sys, y_sys_pred_svr))
# print("Diastolic - MSE:", mean_squared_error(y_dia, y_dia_pred_svr))
# print("Diastolic - R2 Score:", r2_score(y_dia, y_dia_pred_svr))
# print("Pulse Pressure - MSE:", mean_squared_error(y_pp, y_pp_pred_svr))
# print("Pulse Pressure - R2 Score:", r2_score(y_pp, y_pp_pred_svr))

# print("\nRandom Forest Regression:")
# print("Systolic - MSE:", mean_squared_error(y_sys, y_sys_pred_rf))
# print("Systolic - R2 Score:", r2_score(y_sys, y_sys_pred_rf))
# print("Diastolic - MSE:", mean_squared_error(y_dia, y_dia_pred_rf))
# print("Diastolic - R2 Score:", r2_score(y_dia, y_dia_pred_rf))
# print("Pulse Pressure - MSE:", mean_squared_error(y_pp, y_pp_pred_rf))
# print("Pulse Pressure - R2 Score:", r2_score(y_pp, y_pp_pred_rf))

# print("\nXGBoost Regression:")
# print("Systolic - MSE:", mean_squared_error(y_sys, y_sys_pred_xgb))
# print("Systolic - R2 Score:", r2_score(y_sys, y_sys_pred_xgb))
# print("Diastolic - MSE:", mean_squared_error(y_dia, y_dia_pred_xgb))
# print("Diastolic - R2 Score:", r2_score(y_dia, y_dia_pred_xgb))
# print("Pulse Pressure - MSE:", mean_squared_error(y_pp, y_pp_pred_xgb))
# print("Pulse Pressure - R2 Score:", r2_score(y_pp, y_pp_pred_xgb))


# 定義回歸方法列表
# regression_methods = [
#     ('Polynomial Regression', poly_reg_sys, poly_reg_dia, poly_reg_pp),
#     ('Decision Tree Regression', dt_reg_sys, dt_reg_dia, dt_reg_pp),
#     ('SVR', svr_sys, svr_dia, svr_pp),
#     ('Random Forest Regression', rf_reg_sys, rf_reg_dia, rf_reg_pp),
#     ('XGBoost Regression', xgb_reg_sys, xgb_reg_dia, xgb_reg_pp)
# ]