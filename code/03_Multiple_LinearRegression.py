import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#读取数据
data =  pd.read_csv('../datasets/ex1data2.txt',names = ['size','bedrooms','price'])#文件路径,数据集在datasets文件夹
data.head()

#均值归一化
def normalize_feature(data):#定义均值归一化函数
    return (data - data.mean())/data.std()#（x-x的均值）/x的方差
data = normalize_feature(data)#调用均值归一化函数
data.head()

# 数据集可视化
data.plot.scatter('size', 'price', label='size')  # 画出房间大小与价格数据集散点图
plt.show()
data.plot.scatter('bedrooms', 'price', label='size')  # 画出卧室数量大小与价格数据集散点图
plt.show()

data.insert(0, 'ones', 1)  # 在数据集中插入第一列，列名为ones,数值为1
data.head()

# 数据切片
x = data.iloc[:, 0:-1]  # 取x的所有行，取x第一列之后的所有列
x.head()
x = x.values  # 将x由dataframe（数据框）格式转化为ndarray(多维数组)格式
x.shape  # 查看x的形状  (47, 3)

y = data.iloc[:, -1]
y.head()
y = y.values
y.shape  # (47,)

y = y.reshape(47, 1)  # 对y的格式进行转化
y.shape  # (47,1)

# 损失函数
def cost_func(x, y, theta):
    inner = np.power(x @ theta - y, 2)
    return np.sum(inner) / (2 * len(x))  # 调用np.power,幂数为2

# 初始化参数theta
theta = np.zeros((3, 1))  # 将theta初始化为一个（3，1）的数组
# yinwei
cost1 = cost_func(x, y, theta)  # 初始化theta得到的代价函数值

#梯度下降
def gradientDescent(x,y,theta,counts):
    costs = []#创建存放总损失值的空列表
    for i in range(counts):#遍历迭代次数
        theta = theta - x.T@(x@theta-y)*alpha/len(x)
        cost = cost_func(x,y,theta)#调用损失函数得到迭代一次的cost
        costs.append(cost)#将cost传入costs列表
        if i%100 == 0:  #迭代100次，打印cost值
            print(cost)
    return theta,costs

alpha_iters = [0.003, 0.03, 0.0001, 0.001, 0.01]#设置alpha
counts = 200#循环次数

fig, ax = plt.subplots()
for alpha in alpha_iters:  # 迭代不同学习率alpha
    _, costs = gradientDescent(x, y, theta, counts)  # 得到损失值
    ax.plot(np.arange(counts), costs, label=alpha)  # 设置x轴参数为迭代次数，y轴参数为cost
    ax.legend()  # 加上这句  显示label

ax.set(xlabel='counts',  # 图的坐标轴设置
       ylabel='cost',
       title='cost vs counts')  # 标题
plt.show()  # 显示图像