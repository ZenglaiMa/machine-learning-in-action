import numpy as np
import os


def img2vec(filename):
    """将图片(文本形式)转成向量, 每张图片的像素为32*32, 对应文本文件里形式为 32行*每行32个字符
    """
    img_vec = np.zeros((1, 32 * 32))
    with open(filename) as fp:
        for i in range(32):
            line = fp.readline()
            for j in range(32):
                img_vec[0, 32 * i + j] = line[j]
    return img_vec


def get_data(data_path):
    """获取结构化数据
    """
    filenames = os.listdir(data_path)
    files_num = len(filenames)

    feature_mat = np.zeros((files_num, 1024))  # 特征矩阵
    labels = []  # 标签

    for i in range(files_num):
        filename = filenames[i]
        feature_mat[i] = img2vec(os.path.join(data_path, filename))
        labels.append(int(filename.split('_')[0]))

    return feature_mat, labels
