import re
import os
import numpy as np


def tokenizer(text):
    """分词器, 将输入的文本进行分词
    Args:
        text (string): 原始文本
    Returns:
        分词后得到的 token list
    """
    tokens = re.split(r'\W+', text)  # 正则表达式 \W+ 表示匹配数字、字母、下划线
    return [token.lower() for token in tokens if len(token) > 2]  # 每个token变小写, 并去掉少于2个字符的token


def create_vocabulary(email_list):
    """创建词表
    Args:
        email_list (list): 分好词的邮件列表
    Returns:
        vocab (dict): 词表, {key=word, value=position}
    """
    vocab = {}
    for email in email_list:
        for word in email:
            if word not in vocab:
                vocab[word] = len(vocab)
    return vocab


def get_data(ham_path, spam_path):
    """获取数据, 垃圾邮件和非垃圾邮件都是25封
    Args:
        ham_path (string): 非垃圾邮件路径
        spam_path (string): 垃圾邮件路径
    Returns:
        feature_mat (np.array): 特征矩阵
        labels (list): 标签
    """
    email_list = []  # 保存每一封分好词的邮件
    labels = []  # 保存邮件的标签
    for i in range(1, 26):  # 遍历25封邮件
        with open(os.path.join(spam_path, str(i) + '.txt')) as fp:  # 读取垃圾邮件内容
            email_list.append(tokenizer(fp.read()))  # 将分词后的内容保存到 email_list
            labels.append(1)  # 1 表示是垃圾邮件
        with open(os.path.join(ham_path, str(i) + '.txt')) as fp:  # 读取非垃圾邮件
            email_list.append(tokenizer(fp.read()))  # 将分词后的内容保存到 email_list
            labels.append(0)  # 0 表示是非垃圾邮件

    vocab = create_vocabulary(email_list)  # 创建词表

    feature_mat = np.zeros((len(email_list), len(vocab)))  # 构建特征矩阵
    for idx, email in enumerate(email_list):
        for word in email:
            if word in vocab:
                feature_mat[idx][vocab[word]] += 1

    return feature_mat, labels
