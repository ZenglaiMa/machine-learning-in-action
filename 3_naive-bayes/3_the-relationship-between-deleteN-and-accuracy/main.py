import os
import jieba
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB  # 多项式朴素贝叶斯分类器


random.seed(0)  # 种下随机数种子, 便于测试
np.random.seed(0)


def process_raw_text(raw_text_folder_root_path, test_size=0.2):
    """处理原始文本数据
    Args:
        raw_text_folder_root_path (string): 原始文本所在文件夹根路径, 本例中为 ./data/samples
        test_size (float, optional): 测试集所占比例
    Returns:
        train_data_list (list): 训练集文档列表, 列表的每一个元素都是一个文本中的所有单词组成的列表
        train_label_list (list): 训练集标签列表
        test_data_list (list): 测试集文档列表, 列表的每一个元素都是一个文本中的所有单词组成的列表
        test_label_list (list): 测试集标签列表
    """
    data_list = []  # 初始化数据列表, 其中每一项都是一个文件中文本内容组成的列表
    labels = []  # 初始化标签列表, 保存对应的标签

    folder_path_list = os.listdir(raw_text_folder_root_path)
    for folder_path in folder_path_list:
        file_root_path = os.path.join(raw_text_folder_root_path, folder_path)
        filenames = os.listdir(file_root_path)
        for filename in filenames:
            with open(os.path.join(file_root_path, filename), encoding='utf-8') as fp:
                raw_text = fp.read()
            data_list.append(list(jieba.cut(raw_text)))  # list(jieba.cut(text)) 结巴分词并将结果转为list
            labels.append(folder_path)

    data_label_list = list(zip(data_list, labels))  # 将data和对应的label压缩在一起, 注意zip()的用法, 其返回一个对象, 用list()将其转为list
    random.shuffle(data_label_list)  # 将数据随机打散
    index = int(len(data_label_list) * test_size)  # 从该位置划分
    train_set = data_label_list[index:]  # 划分出训练集
    test_set = data_label_list[:index]  # 划分出测试集
    train_data_list, train_label_list = zip(*train_set)  # 解压缩, type(train_data_list) is a tuple
    test_data_list, test_label_list = zip(*test_set)

    return list(train_data_list), list(train_label_list), list(test_data_list), list(test_label_list)


def sort_by_appear_frequency(data_list):
    """按照单词的出现频率对单词进行排序
    Args:
        data_list (list): 数据列表
    Returns:
        sorted_word_list (list): 排序好的单词列表
    """
    word_appear_dict = {}
    for word_list in data_list:
        for word in word_list:
            if word in word_appear_dict:
                word_appear_dict[word] += 1
            else:
                word_appear_dict[word] = 1
    sorted_word_appear_dict = sorted(word_appear_dict.items(), key=lambda f: f[1], reverse=True)  # 按单词出现次数从大到小排序
    sorted_word_list, _ = zip(*sorted_word_appear_dict)

    return list(sorted_word_list)


def create_stopwords_set(stopwords_filename):
    """创建停用词集合
    Args:
        stopwords_filename (string): 停用词文件文件名
    Returns:
        stopwords_set (set): 停用词构成的集合
    """
    stopwords_set = set()
    with open(stopwords_filename, mode='r', encoding='utf-8') as fp:
        for line in fp.readlines():
            word = line.strip()
            if len(word) > 0:
                stopwords_set.add(word)
    return stopwords_set


def extract_feature_words(all_words_list, delete_n, stopwords_set):
    """抽取特征词, 即从出现的所有词中抽取我们认为有用的词, 又即构建有用的词的词表
    Args:
        all_words_list (list): 所有出现的单词组成的列表, 已按词频从高到低排序好
        delete_n (int): 删除词频高的前 delete_n个词
        stopwords_set (set): 停用词集合
    Returns:
        feature_words_dict (dict): 特征词字典, 键为特征词, 值为其索引
    """
    feature_words_dict = {}
    n = 1
    for i in range(delete_n, len(all_words_list)):
        if n > 2000:  # 抽取2000个特征词, 即我们将每个文档的维度设置为 2000
            break
        word = all_words_list[i]
        if (not word.isdigit()) and (word not in stopwords_set) and (1 < len(word) < 5):  # 不是数字、不是停用词、长度大于1小于5, 才可被当做特征词
            if word not in feature_words_dict:
                feature_words_dict[word] = len(feature_words_dict)
                n += 1
    return feature_words_dict


def get_text_feature(train_data_list, test_data_list, vocab_dict):
    """获取特征矩阵
    Args:
        train_data_list (list): 训练集数据
        test_data_list (list): 测试集数据
        vocab_dict (dict): 词表
    Returns:
        train_feature_mat (list): 训练集特征矩阵
        test_feature_mat(list): 测试集特征矩阵
    """
    train_feature_mat = np.zeros((len(train_data_list), len(vocab_dict)))
    test_feature_mat = np.zeros((len(test_data_list), len(vocab_dict)))

    for idx, train_data in enumerate(train_data_list):
        for word in train_data:
            if word in vocab_dict:
                train_feature_mat[idx][vocab_dict[word]] += 1

    for idx, test_data in enumerate(test_data_list):
        for word in test_data:
            if word in vocab_dict:
                test_feature_mat[idx][vocab_dict[word]] += 1

    return train_feature_mat, test_feature_mat


def text_classifier(train_features, test_features, train_labels, test_labels):
    """使用sklearn.MultinomialNB构建朴素贝叶斯文本分类器
    Args:
        train_features (list): 训练集特征
        test_features (list): 测试集特征
        train_labels (list): 训练集标签
        test_labels (list): 测试集特征
    Returns:
        test_accuracy (float): 测试精度
    """
    classifier = MultinomialNB().fit(train_features, train_labels)  # fit(X,y) Fit Naive Bayes classifier according to X, y
    test_accuracy = classifier.score(test_features, test_labels)  # score(X,y) Returns the mean accuracy on the given test data and labels
    return test_accuracy


def test():
    data_path = './data/samples'
    stopwords_path = './stopwords_cn.txt'

    train_data_list, train_label_list, test_data_list, test_label_list = process_raw_text(data_path)
    all_words_list = sort_by_appear_frequency(train_data_list)  # 获取训练集看过的所有单词, 已经按词频从大到小排序好
    stopwords_set = create_stopwords_set(stopwords_path)

    # 以下代码探究删除 delete_n个词频高的词对最终分类精度的影响
    test_accuracies = []
    delete_ns = range(0, 1000, 20)  # 0, 20, 40, ..., 980
    for delete_n in delete_ns:
        vocab = extract_feature_words(all_words_list, delete_n, stopwords_set)
        train_feature_mat, test_feature_mat = get_text_feature(train_data_list, test_data_list, vocab)
        test_accuracy = text_classifier(train_feature_mat, test_feature_mat, train_label_list, test_label_list)
        test_accuracies.append(test_accuracy)
    # 画图
    plt.figure()
    plt.plot(delete_ns, test_accuracies)
    plt.title('relationship between delete_n and test_accuracy')
    plt.xlabel('delete_n')
    plt.ylabel('test_accurecy')
    plt.show()


if __name__ == '__main__':
    test()
