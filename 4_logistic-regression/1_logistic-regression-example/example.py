import numpy as np
import matplotlib.pyplot as plt


def get_data(filename):
    """获取数据
    Args:
        filename (string): 数据所在文件的文件名
    Returns:
        datas (np.array): 数据列表, 每一项为 [1.0, x1, x2]
        labels (np.array): 标签列表
    """
    datas = []
    labels = []
    with open(filename, 'r', encoding='utf-8') as fp:
        for line in fp.readlines():
            line = line.strip().split('\t')
            datas.append([1.0, float(line[0]), float(line[1])])
            labels.append(int(line[2]))
    return np.array(datas), np.array(labels)


def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


def train(X, Y, mode='bgd', epochs=500):
    """梯度下降法训练logistic regression的参数
    Args:
        X (np.array): 数据列表, 每一项为 [1.0, x1, x2], 在本例中 X.shape = (100, 3)
        Y (np.array): 标签列表, 在本例中 Y.shape = (100,)
        mode (string, default): bgd -batch gradient descent, sgd -stochastic gradient descent
        epochs (int): max epochs
    """
    m, n = np.shape(X)

    if mode == 'bgd':
        Y = Y.T.reshape(-1, 1)  # Y.shape = (100,) -> Y.shape = (100, 1)
        weights = np.random.randn(n, 1)  # 随机初始化参数, 在本例中 weights.shape = (3, 1), 隐式 bias
        lr = 0.001  # learning rate
        loss_list = []
        for epoch in range(epochs):
            A = sigmoid(np.matmul(X, weights))  # A.shape = (100, 1)
            loss = -np.sum(np.multiply(Y, np.log(A)) + np.multiply(1 - Y, np.log(1 - A))) / len(X)  # 计算 loss
            loss_list.append(loss)
            dw = np.matmul(X.T, A - Y)  # 计算梯度
            weights -= lr * dw  # 梯度下降更新参数
        # 画出loss走势图
        plt.figure()
        plt.plot(range(0, epochs), loss_list)
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.show()

    elif mode == 'sgd':
        weights = np.random.randn(n)
        base_lr = 0.01  # 基准 learning rate
        loss_list = []
        for epoch in range(epochs):
            loss = 0.0
            data_index = list(range(m))
            for i in range(m):
                rand_index = int(np.random.uniform(0, len(data_index)))  # 随机选择一条数据
                lr = 4 / (1.0 + i + epoch) + base_lr  # 动态 learning rate, 随着训练次数的增加而减小
                a = sigmoid(np.dot(X[data_index[rand_index]], weights))
                loss -= Y[data_index[rand_index]] * np.log(a) + (1 - Y[data_index[rand_index]]) * np.log(1 - a)
                dw = X[data_index[rand_index]] * (a - Y[data_index[rand_index]])  # 计算梯度
                weights -= lr * dw  # 梯度下降
                del(data_index[rand_index])
            loss_list.append(loss / m)
        # 画出loss走势图
        plt.figure()
        plt.plot(range(0, epochs), loss_list)
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.show()


if __name__ == '__main__':
    datas, labels = get_data('./data/test-set.txt')
    train(datas, labels, mode='sgd', epochs=300)
