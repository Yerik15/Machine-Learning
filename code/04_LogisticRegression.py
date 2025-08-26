import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt

# 1.读取数据
path = '../datasets/ex2data1.txt'
data = pd.read_csv(path, names=['Exam1', 'Exam2', 'Accepted'])  # 读取数据
data.head()  # 显示数据前五行

# 2.可视化数据集
fig, ax = plt.subplots()  # 此句显示图像
ax.scatter(data[data['Accepted'] == 0]['Exam1'], data[data['Accepted'] == 0]['Exam2'], c='r', marker='x', label='y=0')
ax.scatter(data[data['Accepted'] == 1]['Exam1'], data[data['Accepted'] == 1]['Exam2'], c='b', marker='o', label='y=1')
ax.legend()  # 显示标签
ax.set_xlabel('exam1')  # 设置坐标轴标签
ax.set_ylabel('exam2')

plt.show()


def get_Xy(data):
    # 在第一列插入1
    data.insert(0, 'ones', 1)
    # 取除最后一列以外的列
    X_ = data.iloc[:, 0:-1]
    # 取特征值
    X = X_.values
    # 取最后一列
    y_ = data.iloc[:, -1]
    y = y_.values.reshape(len(y_), 1)
    return X, y


X, y = get_Xy(data)
print(X)
print(y)
# (100,3)
print(X.shape)
# （100，1）
print(y.shape)


# 3.损失函数
# sigmoid函数
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def Cost_Function(X, y, theta):
    A = sigmoid(X @ theta)
    first = y * np.log(A)
    second = (1 - y) * np.log(1 - A)
    return -np.sum(first + second) / len(X)


theta = np.zeros((3, 1))  # 初始化theta
print(theta.shape)
cost_init = Cost_Function(X, y, theta)
print(cost_init)


# 4.梯度下降
def gradientDescent(X, y, theta, alpha, iters):
    m = len(X)
    costs = []
    for i in range(iters):
        A = sigmoid(X @ theta)
        # X.T:X的转置
        theta = theta - (alpha / m) * X.T @ (A - y)
        cost = Cost_Function(X, y, theta)  # 计算每次迭代损失
        costs.append(cost)  # 总损失
        # if i % 1000 == 0:
        #     print(cost)
    return costs, theta


# 初始化参数
alpha = 0.004
iters = 200000
costs, theta_final = gradientDescent(X, y, theta, alpha, iters)  # 梯度下降完得到的参数theta和损失
print(theta_final)


# 5.预测
def predict(X, theta):
    prob = sigmoid(X @ theta)  # 逻辑回归的假设函数
    return [1 if x >= 0.5 else 0 for x in prob]


print(predict(X, theta_final))

y_ = np.array(predict(X, theta_final))  # 将预测结果转换为数组
print(y_)  # 打印预测结果
y_pre = y_.reshape(len(y_), 1)  # 将预测结果转换为一列

# 预测准确率
acc = np.mean(y_pre == y)
print(acc)  # 0.86


#6.决策边界
# 决策边界就是Xθ=0的时候
coef1 = - theta_final[0, 0] / theta_final[2, 0]
coef2 = - theta_final[1, 0] / theta_final[2, 0]
x = np.linspace(20, 100, 100)
f = coef1 + coef2 * x
fig, ax = plt.subplots()
ax.scatter(data[data['Accepted'] == 0]['Exam1'], data[data['Accepted'] == 0]['Exam2'], c='r', marker='x', label='y=0')
ax.scatter(data[data['Accepted'] == 1]['Exam1'], data[data['Accepted'] == 1]['Exam2'], c='b', marker='o', label='y=1')
ax.legend()
ax.set_xlabel('exam1')
ax.set_ylabel('exam2')
ax.plot(x, f, c='g')
plt.show()