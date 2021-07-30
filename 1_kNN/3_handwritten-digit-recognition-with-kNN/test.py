from kNN import kNN
from data_process import get_data


def test():
    # 获取训练数据
    train_data, train_labels = get_data('./data/trainingDigits')
    # 获取测试数据
    test_data, test_labels = get_data('./data/testDigits')

    test_data_size = len(test_data)

    error_count = 0
    for i in range(test_data_size):
        result = kNN(test_data[i], train_data, train_labels, 3)
        print('预测: %d, 真实: %d' % (result, test_labels[i]))
        if result != test_labels[i]:
            error_count += 1

    print('预测错误总数: %d' % (error_count))
    print('预测错误率为: %.2f%%' % (error_count * 1.0 / test_data_size * 100))


if __name__ == '__main__':
    test()
