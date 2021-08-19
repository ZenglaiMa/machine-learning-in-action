import numpy as np
import matplotlib.pyplot as plt


def load_data(filename):
    """准备数据
    Args:
        filename (string): 文件名
    Returns:
        feature_mat (np.array): 特征矩阵, shape = (sample_num, feature_num)
        label_mat (np.array): 标签矩阵, shape = (sample_num, 1)
    """
    with open(filename, mode='r', encoding='utf-8') as fp:
        lines = fp.readlines()
        sample_num, feature_num = len(lines), len(lines[0].strip().split('\t')) - 1
        feature_mat = np.zeros((sample_num, feature_num))
        label_mat = np.zeros((sample_num, 1))
        for idx, line in enumerate(lines):
            example = line.strip().split('\t')
            feature_mat[idx] = example[:-1]
            label_mat[idx] = [example[-1]]
    return feature_mat, label_mat


def lowess(test_point, X, Y, k=1.0):
    """局部加权的回归: 对于有周期性、波动性等的数据, 如果简单地以线性方式拟合会造成偏差较大(欠拟合),
    而局部加权回归(lowess)能够较好地处理这种问题, 可以拟合出一条符合整体趋势的线.
    算法思想详见 <machine learning in action>
    Args:
        test_point (np.array): 单条样本, 一个样本点
        X (np.array): feature matrix, shape = (sample_num, feature_num)
        Y (np.array): label matrix, shape = (sample_num, 1)
        k (float, optional): use to calculate local weight. Defaults to 1.0.
    Returns:
        y_hat (float): test_point 点处的预测值
    """
    sample_num = np.shape(X)[0]
    weights = np.eye(sample_num)

    for i in range(sample_num):
        diff_vec = test_point - X[i]
        weights[i][i] = np.exp(np.dot(diff_vec, diff_vec) / (-2.0 * (k ** 2)))

    x_T_x = np.matmul(X.T, np.matmul(weights, X))
    if np.linalg.det(x_T_x) == 0:
        print('this matrix is singular, cannot do inverse')
        return

    theta_hat = np.matmul(np.matmul(np.linalg.inv(x_T_x), X.T), np.matmul(weights, Y))
    test_point = test_point.reshape(1, -1)
    y_hat = np.matmul(test_point, theta_hat)

    return y_hat.item()


if __name__ == '__main__':
    X, Y = load_data('./data/ex.txt')
    sample_num = np.shape(X)[0]
    y_hat = np.zeros(sample_num)

    for i in range(sample_num):
        y_hat[i] = lowess(X[i], X, Y, k=0.01)  # attempt to set k = 1.0 / 0.01 / 0.003

    # 画图
    sotred_index = X[:, 1].argsort()
    plt.figure()
    plt.scatter(X[:, 1], Y[:, 0])
    plt.plot(X[sotred_index][:, 1], y_hat[sotred_index], c='red')
    plt.show()
