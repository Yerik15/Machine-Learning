import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('../datasets/ex1data1.txt', names=['Population', 'profit'])
data.head()
data.insert(0, 'ones', 1)
data.head()
data.plot.scatter('Population', 'profit')
plt.show()

X = data.iloc[:,0:-1]
X.head()
X = X.values
X.shape
y = data.iloc[:,-1]
y.head()
y = y.values
y.shape
y = y.reshape(97,1)
y.shape

#正规方程
def normalEquation(X,y):
    theta = np.linalg.inv(X.T@X)@X.T@y
    return theta
theta = normalEquation(X,y)
print(theta)
theta.shape

#代价函数
def cost_func(X,y,theta):
    inner = np.power(X@theta-y,2)
    return np.sum(inner)/(2*len(X))#m = len(X)

theta = np.zeros((2,1))
theta.shape
cost1 = cost_func(X,y,theta)
print(cost1)

#梯度下降
def gradient_Abscent(X, y, theta, alpha, count):
    costs = []
    for i in range(count):
        theta = theta - (X.T @ (X @ theta - y)) * alpha / len(X)
        cost = cost_func(X, y, theta)
        costs.append(cost)
        if i % 100 == 0:
            print(cost)
    return theta, costs


alpha = 0.02
count = 2000
theta1, costs = gradient_Abscent(X, y, theta, alpha, count)
#代价函数可视化
fig,ax = plt.subplots()
ax.plot(np.arange(count),costs)
ax.set(xlabel = 'count',ylabel = 'cost')
plt.show()

# 拟合函数可视化
x = np.linspace(y.min(), y.max(), 100)  # 网格数据
y_ = theta1[0, 0] + theta1[1, 0] * x  # 取theta第一行第一个和第二行第一个

fig, ax = plt.subplots()
ax.scatter(X[:, 1], y, label='training')  # 绘制数据集散点图取x所有行，第2列population
ax.plot(x, y_, 'r', label='predict')  # 绘制预测后的直线
ax.legend()
ax.set(xlabel='population', ylabel='profit')
plt.show()

#人口预测功能哈哈哈
x_predict = float(input('输入预测人口：'))
predict1 = np.array([1,x_predict])@theta1
print(predict1)