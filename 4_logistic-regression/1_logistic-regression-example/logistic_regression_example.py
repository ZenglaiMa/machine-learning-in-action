import numpy as np
import matplotlib.pyplot as plt


def get_data(filename):
    """获取数据
    Args:
        filename (string): 数据所在文件的文件名
    Returns:
        feature_mat (np.array): 特征矩阵, shape = (sample_num, feature_num)
        labels (np.array): 标签, shape = (sample_num, 1)
    """
    with open(filename, mode='r', encoding='utf-8') as fp:
        lines = fp.readlines()
        sample_num = len(lines)
        feature_num = len(lines[0].strip().split('\t'))  # 多一个特征, 第一个特征值设为 1.0, 隐式 bias
        feature_mat = np.zeros((sample_num, feature_num))
        labels = np.zeros((sample_num, 1))
        for idx, line in enumerate(lines):
            example = line.strip().split('\t')
            feature_mat[idx] = [1.0, example[0], example[1]]  # 第一个特征值设为 1.0, 隐式 bias
            labels[idx] = [example[2]]
    return feature_mat, labels


def sigmoid(x):
    """sigmoid function
    """
    return 1.0 / (1 + np.exp(-x))


def train(X, Y, mode='BGD', epochs=500):
    """梯度下降法训练logistic regression的参数
    Args:
        X (np.array): 特征矩阵, shape = (sample_num, feature_num)
        Y (np.array): 标签, shape = (sample_num, 1)
        mode (string, default): BGD -batch gradient descent, SGD -stochastic gradient descent
        epochs (int, default): max epochs
    """
    m, n = np.shape(X)  # m is sample_num, n is feature_num

    if mode == "BGD":
        weights = np.random.randn(n, 1)  # 随机初始化参数
        lr = 0.001  # learning rate
        loss_list = []
        for epoch in range(epochs):
            A = sigmoid(np.matmul(X, weights))
            loss = -np.sum(np.multiply(Y, np.log(A)) + np.multiply(1 - Y, np.log(1 - A))) / len(X)  # 计算 loss
            loss_list.append(loss)

            dw = np.matmul(X.T, A - Y)  # 计算梯度
            weights -= lr * dw  # 梯度下降更新参数

        plt.figure()
        plt.plot(range(0, epochs), loss_list)
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.show()

    elif mode == "SGD":
        Y = Y.reshape(m)  # 将二维array转为一维array, 即 shape(sample_num, 1) -> shape(sample_num,)
        weights = np.random.randn(n)
        base_lr = 0.01  # 基准 learning rate
        loss_list = []
        for epoch in range(epochs):
            data_index = list(range(m))
            loss = 0.0
            for i in range(m):
                rand_index = int(np.random.uniform(0, len(data_index)))  # 随机选择一条样本
                lr = 4 / (1.0 + i + epoch) + base_lr  # 动态 learning rate, 随着训练的进行而减小

                a = sigmoid(np.dot(X[data_index[rand_index]], weights))
                loss -= Y[data_index[rand_index]] * np.log(a) + (1 - Y[data_index[rand_index]]) * np.log(1 - a)  # 计算 loss
                dw = X[data_index[rand_index]] * (a - Y[data_index[rand_index]])  # 计算梯度
                weights -= lr * dw  # 梯度下降更新参数

                del(data_index[rand_index])
            loss_list.append(loss / m)

        plt.figure()
        plt.plot(range(0, epochs), loss_list)
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.show()


if __name__ == '__main__':
    X, Y = get_data('./data/test-set.txt')
    train(X, Y, mode='SGD')
