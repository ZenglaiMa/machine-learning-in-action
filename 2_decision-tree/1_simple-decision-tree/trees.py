import math
import operator


def create_dataset():
    """构建数据集
    Returns:
        data_set - 数据集
        feature_name - 分类属性名
    """
    data_set = [
        [0, 0, 0, 0, 'no'],
        [0, 0, 0, 1, 'no'],
        [0, 1, 0, 1, 'yes'],
        [0, 1, 1, 0, 'yes'],
        [0, 0, 0, 0, 'no'],
        [1, 0, 0, 0, 'no'],
        [1, 0, 0, 1, 'no'],
        [1, 1, 1, 1, 'yes'],
        [1, 0, 1, 2, 'yes'],
        [1, 0, 1, 2, 'yes'],
        [2, 0, 1, 2, 'yes'],
        [2, 0, 1, 1, 'yes'],
        [2, 1, 0, 1, 'yes'],
        [2, 1, 0, 2, 'yes'],
        [2, 0, 0, 0, 'no']
    ]
    # 特征(属性)
    feature_list = ['年龄', '有工作', '有自己的房子', '信贷情况']

    return data_set, feature_list


def calc_shannon_entropy(dataset):
    """计算香农熵(信息熵): E = -SUM(p_i * log2(p_i)) (i from 1 to n, n is the number of classes)
    Args:
        dataset - 数据集
    Returns:
        shannon entropy - 香农熵(信息熵)
    """
    dataset_size = len(dataset)

    # 构建一个字典保存每个类别的样本的数量: {key: label, value: count(label)}
    label_count = {}
    for data in dataset:
        label = data[-1]
        label_count[label] = label_count.get(label, 0) + 1

    # 计算香农熵
    shannon_entropy = 0.0
    for key in label_count.keys():
        prob_key = label_count[key] * 1.0 / dataset_size
        shannon_entropy -= prob_key * math.log2(prob_key)

    return shannon_entropy


def split_dataset(dataset, feature, feature_value):
    """根据特征划分数据集
    Args:
        dataset - 数据集
        feature - 根据这个特征进行划分
        feature_value - feature的取值, 每个子节点对应着该特征的一个取值
    Returns:
        splited_dataset - 划分后的数据集
    """
    splited_dataset = []  # 保存划分后的数据集
    for data in dataset:
        if data[feature] == feature_value:
            # 按feature划分数据集(去掉feature这个特征)并将其保存到划分后的数据集中, 注意 append 和 extend 的用法区别
            reduced_feature_vec = data[:feature]
            reduced_feature_vec.extend(data[feature + 1:])
            splited_dataset.append(reduced_feature_vec)
    return splited_dataset


def choose_best_feature(dataset):
    """选择用来划分数据集的最优(信息增益最大)的特征
    信息增益 g(D, A) = H(D) - H(D|A) = H(D) - SUM(|D_v| / |D| * H(D_v))
    Args:
        dataset - 数据集
    Returns:
        best_feature_index - 最优特征的索引
    """
    feature_num = len(dataset[0]) - 1  # 特征数量
    base_entropy = calc_shannon_entropy(dataset)  # 计算数据集的经验熵, 即 H(D)
    max_info_gain = 0.0  # 最大信息增益
    best_feature_index = -1  # 最优特征的索引

    for i in range(feature_num):  # 遍历所有特征
        feature_values = set([example[i] for example in dataset])  # 获取数据集该特征的所有特征值, set()用来去重

        conditional_entropy = 0.0  # 条件熵, 即 H(D|A)
        for feature_value in feature_values:
            splited_data_set = split_dataset(dataset, i, feature_value)
            prob = len(splited_data_set) * 1.0 / len(dataset)  # 即 |D_v| / |D|
            conditional_entropy += prob * calc_shannon_entropy(splited_data_set)  # 计算条件熵

        info_gain = base_entropy - conditional_entropy  # 计算该特征的信息增益
        if info_gain > max_info_gain:
            max_info_gain = info_gain
            best_feature_index = i

    return best_feature_index


def count_label(label_list):
    """统计每个label出现的次数并返回出现次数最多的label
    Args:
        label_list - label列表
    Returns:
        出现次数最多的label
    """
    label_count = {}
    for label in label_list:
        label_count[label] = label_count.get(label, 0) + 1
    sorted_label_count = sorted(label_count.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_label_count[0][0]


def create_decision_tree(dataset, feature_list):
    """ID3算法创建决策树
    Args:
        dataset - 数据集
        feature_list - 特征名
    Returns:
        decision_tree - 生成好的决策树
    """
    label_list = [example[-1] for example in dataset]  # 获取类别标签列表
    if label_list.count(label_list[0]) == len(label_list):  # 每条数据都同属一个类别
        return label_list[0]
    if len(dataset[0]) == 1:  # 已无特征可分
        return count_label(label_list)

    best_feature_index = choose_best_feature(dataset)  # 选择最优特征, 返回其索引
    best_feature_name = feature_list[best_feature_index]  # 得到最优特征的特征名

    decision_tree = {best_feature_name: {}}

    del(feature_list[best_feature_index])  # 本轮使用该特征进行划分后, 该特征就要删除了
    feature_values = set([example[best_feature_index] for example in dataset])  # 获取该特征的所有取值
    for feature_value in feature_values:  # 每个取值都是一个分支
        splited_dataset = split_dataset(dataset, best_feature_index, feature_value)  # 划分数据集
        decision_tree[best_feature_name][feature_value] = create_decision_tree(splited_dataset, feature_list)

    return decision_tree


if __name__ == '__main__':
    dataset, feature_list = create_dataset()
    decision_tree = create_decision_tree(dataset, feature_list)
    print(decision_tree)
