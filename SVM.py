import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# 读取Excel文件
data = pd.read_excel('D:\\数据\\修正后\\倒对微.xlsx')


# 自变量（该数据集的前13项）
X = data.iloc[:, :-1].values

# 因变量（该数据集的最后1项，即第14项）
y = data.iloc[:, -1].values
# 提取特征和目标变量
#X = data.drop(columns=['SOM含量'])  # SOM含量为目标变量
#y = data['SOM含量']


# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=12)

# 建立支持向量机回归模型
svm_model = SVR(kernel='linear')  # 可选择不同的核函数，如径向基函数（RBF）,C=0.1,epsilon = 0.1,gamma='scale'
svm_model.fit(X_train, y_train)

# 预测
y_train_pred = svm_model.predict(X_train)
y_test_pred = svm_model.predict(X_test)

# 评估模型
#mse = mean_squared_error(y_test, y_pred)
#print("均方误差（MSE）：", mse)

train_pred = svm_model.predict(X_train)
#train_mae = mean_absolute_error(y_train, train_pred)
train_mse = round(mean_squared_error(y_train, train_pred),3)
train_rmse = round(np.sqrt(train_mse),3)
train_r2 = round(r2_score(y_train, train_pred),3)
train_rpd = round(np.std(y_train)/train_rmse,3)

print("训练集评价指标：")
#print("平均绝对误差（MAE）：", train_mae)
print("均方误差（MSE）：", train_mse)
print("均方根误差（RMSE）：", train_rmse)
print("R^2 分数（R-squared）：", train_r2)
print("RPD：", train_rpd)
# 计算测试集上的指标
#test_mae = mean_absolute_error(y_test, y_pred)
test_mse = round(mean_squared_error(y_test, y_test_pred),3)
test_rmse = round(np.sqrt(test_mse),3)
test_r2 = round(r2_score(y_test, y_test_pred),3)
test_rpd = round(np.std(y_test)/test_rmse,3)
print("\n测试集评价指标：")
#print("平均绝对误差（MAE）：", test_mae)
print("均方误差（MSE）：", test_mse)
print("均方根误差（RMSE）：", test_rmse)
print("R^2 分数（R-squared）：", test_r2)
print("RPD：", test_rpd)



# y_test_pred_flat = y_test_pred.flatten()
# y_test_flat = y_test.flatten()
# data = {'测试集预测值':y_test_pred_flat, '测试集实测值': y_test_flat}
# df = pd.DataFrame(data)
#
# # 将 DataFrame 写入 Excel 文件
# df.to_excel('倒对微测试集.xlsx', index=False)
#
# y_train_pred_flat = y_train_pred.flatten()
# y_train_flat = y_train.flatten()
# data = {'训练集预测值':y_train_pred_flat, '训练集实测值': y_train_flat}
# df = pd.DataFrame(data)
#
# # 将 DataFrame 写入 Excel 文件
# df.to_excel('倒对微训练集.xlsx', index=False)