# # import matplotlib.pyplot as plt  # 数据可视化库
# # import seaborn as sns  # 高级数据可视化工具
# import warnings
# import pandas as pd
# import random
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error, explained_variance_score, mean_absolute_error, r2_score
# from deepforest import CascadeForestRegressor
#
# # 固定随机种子
# random.seed(12)  # Python 自带随机数生成器
# np.random.seed(12)  # NumPy 随机数生成器
#
# warnings.filterwarnings(action='ignore')  # 忽略告警
#
# # 读取数据
# df = pd.read_excel('D:\\数据\\db5修正后\\尺度10.xlsx', engine='openpyxl')
#
# # 查看数据前5行
# print('*************查看数据前5行*****************')
# print(df.head())
#
# # 数据缺失值统计
# print('**************数据缺失值统计****************')
# print(df.info())
#
# # 描述性统计分析
# print(df.describe())
# print('******************************')
#
# # 提取特征变量和标签变量
# y = df['target'].values  # 使用 df['有机质'] 来获取目标变量
# X = df.drop('target', axis=1).values
#
# # 划分训练集和测试集
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=12)
#
# # 构建深度森林回归模型
# cfr_model = CascadeForestRegressor(n_estimators=2, min_samples_leaf=1, max_layers=3)  # 构建模型
# cfr_model.fit(X_train, y_train)  # 拟合模型
# y_pred = cfr_model.predict(X_test)  # 预测
#
# # 模型评估
# print('----------------模型评估-----------------')
# print('深度森林回归模型-R方值：{}'.format(round(r2_score(y_test, y_pred), 4)))
# print('深度森林回归模型-均方误差：{}'.format(round(mean_squared_error(y_test, y_pred), 4)))
# print('深度森林回归模型-可解释方差值：{}'.format(round(explained_variance_score(y_test, y_pred), 4)))
# print('深度森林回归模型-平均绝对误差：{}'.format(round(mean_absolute_error(y_test, y_pred), 4)))
#
#
# # # 真实值与预测值比对图
# # plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
# # plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
# # plt.plot(y_test, label="真实值", color="blue", linewidth=1.5, linestyle="-")  # 绘制真实值折线图
# # plt.plot(y_pred, label="预测值", color="red", linewidth=1.5, linestyle="-.")  # 绘制预测值折线图
# # plt.legend()  # 设置图例
# # plt.title("深度森林回归模型真实值与预测值比对图")  # 设置标题名称
# # plt.show()  # 显示图片

import warnings
import pandas as pd
import random
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, explained_variance_score, mean_absolute_error, r2_score
from deepforest import CascadeForestRegressor

# 固定随机种子
random.seed(12)  # Python 自带随机数生成器
np.random.seed(12)  # NumPy 随机数生成器

warnings.filterwarnings(action='ignore')  # 忽略告警

# 读取数据
df = pd.read_excel('D:\\数据\\db5修正前\\尺度1.xlsx', engine='openpyxl')

# 提取特征变量和标签变量
y = df['target'].values  # 使用 df['有机质'] 来获取目标变量
X = df.drop('target', axis=1).values

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=12)

# 构建深度森林回归模型
cfr_model = CascadeForestRegressor(n_estimators=2, min_samples_leaf=1, max_layers=3)
cfr_model.fit(X_train, y_train)

# 预测
y_train_pred = cfr_model.predict(X_train)  # 训练集预测
y_test_pred = cfr_model.predict(X_test)    # 测试集预测

# 定义评估函数
def evaluate_model(y_true, y_pred, dataset_name="Dataset"):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)  # 计算RMSE
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    evs = explained_variance_score(y_true, y_pred)

    print(f'--------- {dataset_name} 评估结果 ---------')
    print('R²:', round(r2, 4))
    print('均方误差（MSE）:', round(mse, 4))
    print('均方根误差（RMSE）:', round(rmse, 4))
    print('平均绝对误差（MAE）:', round(mae, 4))
    print('可解释方差（EVS）:', round(evs, 4))
    print('------------------------------------------')

# 输出训练集和测试集的评价指标
evaluate_model(y_train, y_train_pred, dataset_name="训练集")
evaluate_model(y_test, y_test_pred, dataset_name="测试集")

# # 将训练集和测试集的预测值与实际值保存到Excel文件
# train_results = pd.DataFrame({
#     '实际值': y_train.flatten(),  # 确保y_train是1D数组
#     '预测值': y_train_pred.flatten()  # 确保y_train_pred是1D数组
# })
#
# test_results = pd.DataFrame({
#     '实际值': y_test.flatten(),  # 确保y_test是1D数组
#     '预测值': y_test_pred.flatten()  # 确保y_test_pred是1D数组
# })
#
# # 保存到 Excel 文件
# with pd.ExcelWriter('处理前尺度1.xlsx') as writer:
#     train_results.to_excel(writer, sheet_name='训练集预测', index=False)
#     test_results.to_excel(writer, sheet_name='测试集预测', index=False)
#
# print("训练集和测试集的预测结果已保存到 '处理前尺度1.xlsx'.")