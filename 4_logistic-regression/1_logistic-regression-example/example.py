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


def train(X, Y):
    """梯度下降法训练logistic regression的参数
    Args:
        X (np.array): 数据列表, 每一项为 [1.0, x1, x2], 在本例中 X.shape = (100, 3)
        Y (np.array): 标签列表, 在本例中 Y.shape = (100,)
    """
    Y = Y.T.reshape(-1, 1)  # Y.shape = (100, 1)
    _, n = np.shape(X)
    weights = np.random.randn(n, 1)  # 随机初始化参数, 在本例中 weights.shape = (3, 1), 隐式 bias
    lr = 0.001  # learning rate

    max_epochs = 500
    loss_list = []
    for epoch in range(max_epochs):
        A = sigmoid(np.matmul(X, weights))  # A.shape = (100, 1)
        loss = -np.sum((np.multiply(Y, np.log(A)) + np.multiply(1 - Y, np.log(1 - A)))) / len(X)  # 计算 loss
        loss_list.append(loss)
        gradient = np.matmul(X.T, A - Y)  # 计算梯度
        weights -= lr * gradient  # 梯度下降更新参数
    # 画出loss走势图
    plt.figure()
    plt.plot(range(0, max_epochs), loss_list)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()


if __name__ == '__main__':
    datas, labels = get_data('./data/test-set.txt')
    train(datas, labels)
