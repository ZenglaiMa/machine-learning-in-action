import os

from kNN import kNN
from data_process import get_data


def test():
    feature_mat, labels = get_data(os.path.join('./data', 'datingTestSet.txt'))

    test_rate = 0.1  # 用10%的数据来测试分类器的效果
    test_data_size = int(len(feature_mat) * test_rate)  # 10%的测试数据的个数
    error_count = 0  # 统计分类错误的数量
    for i in range(test_data_size):  # 用前10%的数据来测试
        result = kNN(feature_mat[i, :], feature_mat[test_data_size:], labels[test_data_size:], 4)
        print('分类结果: %d, 真实类别: %d' % (result, labels[i]))
        if result != labels[i]:
            error_count += 1
    print('错误率: %.2f%%' % (error_count * 1.0 / test_data_size * 100))


if __name__ == '__main__':
    test()
