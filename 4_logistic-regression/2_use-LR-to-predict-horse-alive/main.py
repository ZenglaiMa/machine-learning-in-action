import numpy as np


def get_data(filename):
    """获取数据
    Args:
        filename (string): 文件名
    Returns:
        feature_mat (np.array): 特征矩阵, shape = (sample_num, feature_num)
        labels (np.array): 标签, shape = (sample_num, 1)
    """
    with open(filename, mode='r', encoding='utf-8') as fp:
        lines = fp.readlines()
        sample_num = len(lines)
        feature_num = len(lines[0].strip().split('\t')) - 1
        feature_mat = np.zeros((sample_num, feature_num))
        labels = np.zeros((sample_num, 1))
        for idx, line in enumerate(lines):
            example = line.strip().split('\t')
            feature_mat[idx] = example[:-1]
            labels[idx] = [example[-1]]
    return feature_mat, labels


def sigmoid(x):
    """ sigmoid function
    """
    return 1.0 / (1 + np.exp(-x))


def train_lr(X, Y, epochs=150):
    """使用随机梯度下降法训练logistic regression的参数
    Args:
        X (np.array): 样本, shape = (sample_num, feature_num)
        Y (np.array): 标签, shape = (sample_num, 1)
        epochs (int, default): 最大训练次数
    Returns:
        weights (np.array): 训练好的权重
        bias (float): 训练好的偏置值
    """
    sample_num, feature_num = np.shape(X)
    Y = Y.reshape(sample_num)  # 将Y转成一维array

    weights = np.ones(feature_num)  # 初始化 weights
    bias = 1.0  # 初始化 bias

    base_lr = 0.01  # 基准 learning rate
    for epoch in range(epochs):
        data_index = list(range(sample_num))
        for i in range(sample_num):  # 随机梯度下降, 每随机经过一个样本都执行一次梯度下降
            rand_index = int(np.random.uniform(0, len(data_index)))  # 随机选择一条样本
            del(data_index[rand_index])
            lr = 4 / (1.0 + epoch + i) + base_lr  # 动态 learning rate, 随训练进行越来越小

            a = sigmoid(np.dot(X[rand_index], weights) + bias)
            dw = X[rand_index] * (a - Y[rand_index])  # 计算梯度
            db = a - Y[rand_index]
            weights -= lr * dw  # 梯度下降
            bias -= lr * db

    return weights, bias


def classifier(input_x, weights, bias):
    return 1.0 if np.dot(weights, input_x) + bias > 0.5 else 0.0


if __name__ == '__main__':
    train_X, train_Y = get_data('./data/horseColicTraining.txt')
    weights, bias = train_lr(train_X, train_Y, epochs=500)

    test_X, test_Y = get_data('./data/horseColicTest.txt')
    test_Y = test_Y.reshape(len(test_Y))  # 二维array转为一维array

    error_count = 0
    for i in range(len(test_X)):
        result = classifier(test_X[i], weights, bias)
        if result != test_Y[i]:
            error_count += 1
    print('error rate: %.3f%%' % (error_count * 1.0 / len(test_X) * 100))
