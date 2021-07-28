import numpy as np


def get_data(filename):
    with open(filename) as fp:
        lines = fp.readlines()
        feature_mat = np.zeros((len(lines), 3))  # 特征矩阵
        labels = []  # 类别标签
        index = 0
        for line in lines:
            line = line.strip().split('\t')
            feature_mat[index, :] = line[:3]
            if line[-1] == 'didntLike':
                labels.append(1)
            elif line[-1] == 'smallDoses':
                labels.append(2)
            elif line[-1] == 'largeDoses':
                labels.append(3)
            index += 1
    return feature_mat, labels


# 归一化 (X-X_min)/(X_max-X_min)
def normalization(feature_mat):
    x_min = np.min(feature_mat, axis=0)
    x_max = np.max(feature_mat, axis=0)
    return (feature_mat - x_min) / (x_max - x_min), x_min, (x_max - x_min)
