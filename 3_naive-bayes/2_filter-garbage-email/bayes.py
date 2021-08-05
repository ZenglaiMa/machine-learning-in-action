import numpy as np


def train_bayes_classifier(documents, labels):
    """训练朴素贝叶斯 0-1分类器 p(c_i|w) = [p(w|c_i) * p(c_i)] / p(w)
    其中, p(c_i) 表示在所有类别中 c_i 类出现的概率;
    又, 根据条件独立性假设, 有: p(w|c_i) = p(w_0|c_i)p(w_1|c_i)p(w_2|c_i)...p(w_n|c_i);
    p(w)一直不变, 故无需考虑.
    Args:
        documents - 训练文档数据
        labels - 训练类别标签(只有 0,1 两种标签)
    Returns:
        p_w_given_c0 - p(w|c_0)
        p_w_given_c1 - p(w|c_1)
        p_c0 - p(c_0)
        p_c1 - p(c_1)
    """
    p_c1 = sum(labels) * 1.0 / len(labels)  # 1-类的概率, 即 p(c_1)
    p_c0 = 1 - p_c1  # 0-类的概率, 即 P(c_0)

    # wi_num_given_c0 = np.zeros(len(documents[0]))  # 已知 0-类的情况下词 w_i 出现的数量
    # wi_num_given_c1 = np.zeros(len(documents[0]))  # 已知 1-类的情况下词 w_i 出现的数量
    # words_appear_num_given_c0 = 0  # 已知 0-类的情况下所有词出现的总数量
    # words_appear_num_given_c1 = 0  # 已知 1-类的情况下所有词出现的总数量

    # 计算 p(w|c_i) = p(w_0|c_i)p(w_1|c_i)p(w_2|c_i)...p(w_n|c_i) 时,
    # 如果其中一个概率值为0, 那么最后的乘积也为0. 为降低这种影响, 可以将词的出现数初始化为1,
    # 并将分母初始化为2, 即拉普拉斯平滑.
    wi_num_given_c0 = np.ones(len(documents[0]))
    wi_num_given_c1 = np.ones(len(documents[0]))
    words_appear_num_given_c0 = 2
    words_appear_num_given_c1 = 2

    for i in range(len(documents)):  # 遍历所有文档
        if labels[i] == 1:  # 1-类
            wi_num_given_c1 += documents[i]
            words_appear_num_given_c1 += sum(documents[i])
        else:  # 0-类
            wi_num_given_c0 += documents[i]
            words_appear_num_given_c0 += sum(documents[i])

    # p_w_given_c0 = wi_num_given_c0 * 1.0 / words_appear_num_given_c0  # p(w|c_0)
    # p_w_given_c1 = wi_num_given_c1 * 1.0 / words_appear_num_given_c1  # p(w|c_1)

    # 取对数(ln)概率, 防止很小的数联乘可能造成的下溢或错误结果
    p_w_given_c0 = np.log(wi_num_given_c0 * 1.0 / words_appear_num_given_c0)
    p_w_given_c1 = np.log(wi_num_given_c1 * 1.0 / words_appear_num_given_c1)

    return p_w_given_c0, p_w_given_c1, p_c0, p_c1


def bayes_classifier(to_classify_vec, p_w_given_c0, p_w_given_c1, p_c0, p_c1):
    """朴素贝叶斯 0-1分类器
    Args:
        to_classify_vec - 待分类向量
        p_w_given_c0 - # p(w|c_0)
        p_w_given_c1 - # p(w|c_1)
        p_c0 - P(c_0)
        p_c1 - P(c_1)
    Returns:
        1 - 1-类
        0 - 0-类
    """
    p0 = np.sum(p_w_given_c0 * to_classify_vec) + np.log(p_c0)  # 待分类向量是0类的概率, 因为用对数概率表示, 所以联乘变联加
    p1 = np.sum(p_w_given_c1 * to_classify_vec) + np.log(p_c1)  # 待分类向量是1类的概率
    return 0 if p0 > p1 else 1
