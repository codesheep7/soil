# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.cross_decomposition import PLSRegression
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
#
# for random_state in range(51):  # 从0到50遍历不同的random_state值
#     # 从Excel文件加载数据
#     file_path = 'D:\\数据\\修正后\\微分.xlsx'  # 替换为你的Excel文件路径
#     df = pd.read_excel(file_path)
#
#     # 假设特征列是所有列除了最后一列（目标变量）
#     X = df.iloc[:, :-1].values
#     y = df.iloc[:, -1].values
#
#     # 划分训练集和测试集
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_state)
#
#     # 建立偏最小二乘回归模型
#     n_components = 3  # 设置偏最小二乘回归中的主成分数量
#     pls = PLSRegression(n_components=n_components)
#     pls.fit(X_train, y_train)
#
#     # 在训练集上进行预测
#     y_train_pred = pls.predict(X_train)
#
#     # 评估模型性能在训练集上
#     mse_train = mean_squared_error(y_train, y_train_pred)
#     mae_train = mean_absolute_error(y_train, y_train_pred)
#     r2_train = r2_score(y_train, y_train_pred)
#     # 计算均方根误差
#     rmse_train = np.sqrt(mse_train)
#
#     # 输出均方根误差和对应的random_state
#     print(f"random_state: {random_state}")
#     print(f"训练集均方根误差 (RMSE)：{rmse_train}")
#     print(f"训练集决定系数 (R^2)：{r2_train}")
#     # 在测试集上进行预测
#     y_test_pred = pls.predict(X_test)
#
#     # 评估模型性能在测试集上
#     mse_test = mean_squared_error(y_test, y_test_pred)
#     mae_test = mean_absolute_error(y_test, y_test_pred)
#     r2_test = r2_score(y_test, y_test_pred)
#     # 计算均方根误差
#     rmse_test = np.sqrt(mse_test)
#
#     # 输出均方根误差和对应的random_state
#     print(f"测试集均方根误差 (RMSE)：{rmse_test}")
#     print(f"测试集决定系数 (R^2)：{r2_test}")
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 从Excel文件加载数据
file_path = 'D:\\数据\\S7HOU.xlsx'  # 替换为你的Excel文件路径
df = pd.read_excel(file_path)

# 假设特征列是所有列除了最后一列（目标变量）
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=12)

# 建立偏最小二乘回归模型
n_components = 3 # 设置偏最小二乘回归中的主成分数量
pls = PLSRegression(n_components=n_components)
pls.fit(X_train, y_train)

# 在训练集上进行预测
y_train_pred = pls.predict(X_train)

# 评估模型性能在训练集上
mse_train = round(mean_squared_error(y_train, y_train_pred), 3)
mae_train = round(mean_absolute_error(y_train, y_train_pred), 3)
r2_train = round(r2_score(y_train, y_train_pred), 3)

# 计算均方根误差
rmse_train = round(np.sqrt(mse_train),3)
rpd_train = round(np.std(y_train)/rmse_train,3)
#print(y_train)
#print(y_train_pred)
# 输出均方根误差
print(f"训练集均方根误差 (RMSE)：{rmse_train}")

# 输出评价指标
print(f"训练集均方误差 (MSE)：{mse_train}")
#print(f"训练集平均绝对误差 (MAE)：{mae_train}")
print(f"训练集决定系数 (R^2)：{r2_train}")
print("RPD：", rpd_train)
# 在测试集上进行预测
y_test_pred = pls.predict(X_test)

# 评估模型性能在测试集上
mse_test = round(mean_squared_error(y_test, y_test_pred),3)
mae_test = round(mean_absolute_error(y_test, y_test_pred),3)
r2_test = round(r2_score(y_test, y_test_pred),3)
# 计算均方根误差
rmse_test = round(np.sqrt(mse_test),3)
rpd_test = round(np.std(y_test)/rmse_test,3)
# 输出均方根误差
print(f"测试集均方根误差 (RMSE)：{rmse_test}")

# 输出评价指标
print(f"测试集均方误差 (MSE)：{mse_test}")
#print(f"测试集平均绝对误差 (MAE)：{mae_test}")
print(f"测试集决定系数 (R^2)：{r2_test}")
print("RPD：", rpd_test)

#
# y_test_pred_flat = y_test_pred.flatten()
# y_test_flat = y_test.flatten()
# data = {'测试集预测值':y_test_pred_flat, '测试集实测值': y_test_flat}
# df = pd.DataFrame(data)
#
# # 将 DataFrame 写入 Excel 文件
# df.to_excel('对数的微分测试集.xlsx', index=False)
#
# y_train_pred_flat = y_train_pred.flatten()
# y_train_flat = y_train.flatten()
# data = {'训练集预测值':y_train_pred_flat, '训练集实测值': y_train_flat}
# df = pd.DataFrame(data)
#
# # 将 DataFrame 写入 Excel 文件
# df.to_excel('对数的微分训练集.xlsx', index=False)
