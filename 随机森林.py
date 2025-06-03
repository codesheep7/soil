# # 数据的相关性分析
# import seaborn as sns  # 导入数据集分布可视化库
# from matplotlib import pyplot as plt
# import pandas as pd
# from pylab import mpl
# data = pd.read_excel("E:\高淳水质\光谱信息提取\副本6月基地数据汇总(1).xlsx", header=0,sheet_name="有机质相关性")
# mpl.rcParams['font.sans-serif']=['SimHei']
# mpl.rcParams['axes.unicode_minus']=False
# sns.heatmap(data.corr(), cmap="YlGnBu", annot=True)  # 绘制热力图
# plt.title('相关性分析热力图')  # 设置标题
# plt.show()
# # 提取特征变量和标签变量
# X = data.drop(columns=['有机质'])
# y = data['有机质']
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor
# 全局随机状态
RANDOM_STATE =12


# 导入数据集
#datasets=pd.read_excel("D:\桌面\test.xlsx", header=0,sheet_name="Sheet2")
excel_path = 'D:\\数据\\微分后.xlsx'  # 替换为你的 Excel 文件路径
datasets = pd.read_excel(excel_path)
# 输出数据预览
print(datasets.head())

# 自变量（该数据集的前13项）
X = datasets.iloc[:, :-1].values

# 因变量（该数据集的最后1项，即第14项）
y = datasets.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=RANDOM_STATE)

# 评估回归性能
# criterion ：
# 回归树衡量分枝质量的指标，支持的标准有三种：
# 1）输入"mse"使用均方误差mean squared error(MSE)，父节点和叶子节点之间的均方误差的差额将被用来作为特征选择的标准，
# 这种方法通过使用叶子节点的均值来最小化L2损失
# 2）输入“friedman_mse”使用费尔德曼均方误差，这种指标使用弗里德曼针对潜在分枝中的问题改进后的均方误差
# 3）输入"mae"使用绝对平均误差MAE（mean absolute error），这种指标使用叶节点的中值来最小化L1损失

# 此处使用mse
forest = RandomForestRegressor(n_estimators=500,
                                criterion='friedman_mse',
                                random_state=RANDOM_STATE,
                               max_depth=3,
                                n_jobs=-1)
# forest = RandomForestRegressor(n_estimators=400,
#                                criterion='friedman_mse',
#                                random_state=0,
#                                max_depth=5,
#                                min_samples_split=6,
#                                min_samples_leaf=4,
#                                max_features='sqrt',
#                                n_jobs=-1)
forest.fit(X_train, y_train)
y_train_pred = forest.predict(X_train)
y_test_pred = forest.predict(X_test)

#rf.fit(X_train,y_train)
#pred=rf.predict(X_test)
#print("真实值：",y_test)
#print("预测值：",pred)
#直接使用决策树模型
#tree_clf = DecisionTreeRegressor(random_state=4)
#tree_clf.fit(X_train,y_train)
#y_train_pred = tree_clf.predict(X_train)
#y_test_pred = tree_clf.predict(X_test)
#使用bagging模型
# bag_clf = BaggingRegressor(
#     DecisionTreeRegressor(), #基本模型
#     n_estimators=1000,
#     bootstrap=True,
#     n_jobs=-1,#并行计算
#     random_state = 0
# )
# bag_clf.fit(X_train,y_train)
# y_train_pred = bag_clf.predict(X_train)
# y_test_pred = bag_clf.predict(X_test)


print('MSE train: %.3f, test: %.3f' % (
     mean_squared_error(y_train, y_train_pred),
     mean_squared_error(y_test, y_test_pred)))

print('RMSE train: %.3f, test: %.3f' % (
    np.sqrt(mean_squared_error(y_train, y_train_pred)),
    np.sqrt( mean_squared_error(y_test, y_test_pred))))
print('R^2 train: %.3f, test: %.3f' % (
  r2_score(y_train, y_train_pred),
   r2_score(y_test, y_test_pred)))
print('RPD train: %.3f, test: %.3f' % (
  np.std(y_train)/ np.sqrt(mean_squared_error(y_train, y_train_pred)),
   np.std(y_test)/np.sqrt( mean_squared_error(y_test, y_test_pred))))
data = {'测试集预测值':y_test_pred, '测试集实测值': y_test}
df = pd.DataFrame(data)

# 将 DataFrame 写入 Excel 文件
df.to_excel('微分后测试集.xlsx', index=False)
data = {'训练集预测值':y_train_pred, '训练集实测值': y_train}
df = pd.DataFrame(data)

# 将 DataFrame 写入 Excel 文件
df.to_excel('微分后训练集.xlsx', index=False)
# print("测试集预测值为：\n",y_test_pred)
# print("测试集实测值为：\n",y_test)
# print("训练集预测值为：\n",y_train_pred)
# print("训练集实测值为：\n",y_train)

