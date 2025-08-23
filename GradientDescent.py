from numpy import *

#数据集大小
m = 20
#x坐标
x_0 = ones((m,1))
x_1 = arange(1,m+1).reshape(m,1)
x = hstack((x_0, x_1))
#y坐标
y = array([
    3, 4, 5, 5, 2, 4, 7, 8, 11, 8, 12,
    11, 13, 13, 16, 17, 18, 17, 19, 21
]).reshape(m, 1)
#学习率
alpha = 0.01

#代价函数定义
def cost(theta, x, y):
    diff = dot(x, theta) - y
    return (1/(2*m))*dot(diff.transpose(), diff)

#定义对应的梯度函数
def gradient_function(theta, x, y):
    diff = dot(x, theta) - y
    return (1/m)*dot(x.transpose(), diff)

#梯度下降迭代
def gradient_descent(x,y,alpha):
    theta = array([1,1]).reshape(2,1)
    gradient = gradient_function(theta, x, y)
    while not all(abs(gradient) <= 1e-5):
        theta = theta - alpha*gradient
        gradient = gradient_function(theta, x, y)
    return theta

optimal = gradient_descent(x,y,alpha)
print("optimal = ", optimal)
print("cost function = ", cost(optimal,x,y)[0][0])


# 画图
def plot(X, Y, theta):
    import matplotlib.pyplot as plt
    ax = plt.subplot(111)
    ax.scatter(X, Y, s=30, c="red", marker="s")
    plt.xlabel("X")
    plt.ylabel("Y")
    x = arange(0, 21, 0.2)  # x的范围
    y = theta[0] + theta[1] * x
    ax.plot(x, y)
    plt.show()


plot(x_1,y,optimal)