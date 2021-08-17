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


def regression(X, Y):
    """使用最小二乘法求解回归系数, 最小二乘法通过最小化平方误差来寻找数据的最佳函数匹配
    具体操作过程: 平方误差为 sum((y_i - x_i * w) ** 2), i from 1 to n
    写成矩阵形式为: (Y - Xw).T * (Y - Xw), 对 w 求导得 dw=X.T*(Y-Xw), 令 dw=0
    得 w_hat = (X.T * X).inv * X.T * Y
    Args:
        X (np.array): faeture matrix, shape = (sample_num, feature_num)
        Y (np.array): label matrix, shape = (sample_num, 1)
    """
    x_T_x = np.matmul(X.T, X)
    if np.linalg.det(x_T_x) == 0:  # 行列式为 0, 矩阵不可逆
        print('this matrix is singular, cannot do inverse')
        return
    w_hat = np.matmul(np.matmul(np.linalg.inv(x_T_x), X.T), Y)

    return w_hat


if __name__ == '__main__':
    X, Y = load_data('./data/ex.txt')
    w_hat = regression(X, Y)  # 使用最小二乘法获得回归系数
    y_hat = np.matmul(X, w_hat)  # 得到预测值

    # 画图
    plt.figure()
    plt.scatter(X[:, 1], Y[:, 0])
    plt.plot(X[:, 1], y_hat[:, 0])
    plt.show()
