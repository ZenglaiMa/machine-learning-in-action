from data_process import get_data
from bayes import train_bayes_classifier, bayes_classifier

import random


def test():
    train_set, train_labels = get_data(ham_path='./email/ham', spam_path='./email/spam')
    train_set = list(train_set)  # train_set是 np.array类型, 将其转为 python list, np.array是无法进行del()操作的

    # 从50封邮件中, 随机抽取10封作为测试集, 剩下的40封作为训练集
    test_set = []
    test_labels = []
    for i in range(10):
        rand_index = int(random.uniform(0, len(train_set)))
        test_set.append(train_set[rand_index])
        test_labels.append(train_labels[rand_index])
        del(train_set[rand_index])
        del(train_labels[rand_index])

    # 训练朴素贝叶斯分类器
    p_w_given_c0, p_w_given_c1, p_c0, p_c1 = train_bayes_classifier(train_set, train_labels)

    # 测试
    error_count = 0  # 分类错误的数量
    for i in range(len(test_set)):
        result = bayes_classifier(test_set[i], p_w_given_c0, p_w_given_c1, p_c0, p_c1)
        print('预测: %d, 实际: %d' % (result, test_labels[i]))
        if result != test_labels[i]:
            error_count += 1
    print('错误率: %.2f%%' % (error_count * 1.0 / len(test_set) * 100))


if __name__ == '__main__':
    test()
