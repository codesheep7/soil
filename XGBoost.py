import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import numpy as np

# 1. 读取Excel数据
file_path = 'D:\\数据\\修正后\\倒数.xlsx' # 替换为你的Excel文件路径
data = pd.read_excel(file_path)

# 2. 数据预处理
# 假设最后一列为有机质含量，前面的列为光谱数据
X = data.iloc[:, :-1]  # 高光谱数据
y = data.iloc[:, -1]   # 有机质含量

# 3. 数据标准化
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 4. 拆分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=12)

# 5. 定义和训练XGBoost模型，添加正则化参数
model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=150,
    learning_rate=0.01,
    max_depth=3,
    reg_lambda=1.0,  # L2正则化项，适度增加此值来抑制过拟合
    reg_alpha=0.1   # L1正则化项，适度增加此值来增加稀疏性
)
model.fit(X_train, y_train)


# 6. 预测和评估
# 训练集评估
y_train_pred = model.predict(X_train)
train_mse = mean_squared_error(y_train, y_train_pred)
train_rmse = np.sqrt(train_mse)  # 计算RMSE
train_r2 = r2_score(y_train, y_train_pred)

# 测试集评估
y_test_pred = model.predict(X_test)
test_mse = mean_squared_error(y_test, y_test_pred)
test_rmse = np.sqrt(test_mse)  # 计算RMSE
test_r2 = r2_score(y_test, y_test_pred)

# 输出结果（保留三位小数）
print("训练集评价指标:")
print(f"Mean Squared Error (训练集): {train_mse:.3f}")
print(f"Root Mean Squared Error (训练集): {train_rmse:.3f}")
print(f"R-squared Score (训练集): {train_r2:.3f}")

print("\n测试集评价指标:")
print(f"Mean Squared Error (测试集): {test_mse:.3f}")
print(f"Root Mean Squared Error (测试集): {test_rmse:.3f}")
print(f"R-squared Score (测试集): {test_r2:.3f}")

# 7. 模型保存（可选）
model.save_model('xgboost_organic_model.json')
# # 训练集数据
# train_results = pd.DataFrame({
#     'Actual (Train)': y_train,
#     'Predicted (Train)': y_train_pred
# })
#
# # 测试集数据
# test_results = pd.DataFrame({
#     'Actual (Test)': y_test,
#     'Predicted (Test)': y_test_pred
# })
#
# # 将训练集和测试集结果写入Excel
# with pd.ExcelWriter('处理前尺度1.xlsx') as writer:
#     train_results.to_excel(writer, sheet_name='Train Results', index=False)
#     test_results.to_excel(writer, sheet_name='Test Results', index=False)
#
# print("\n预测结果已保存到 '处理前尺度1.xlsx'.")


