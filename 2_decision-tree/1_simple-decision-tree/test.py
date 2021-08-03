from tree import create_decision_tree


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


def classify(tree, feature_list, test_vec):
    """将决策树用于分类
    Args:
        tree - 决策树, 字典类型, 如 {'有自己的房子': {0: {'有工作': {0: 'no', 1: 'yes'}}, 1: 'yes'}}
        feature_list - 特征名列表
        test_vec - 待分类向量
    Returns:
        label - 预测的标签
    """
    root_key = list(tree.keys())[0]  # 注意 dict.keys() 返回值类型为 dict_keys, 不能直接索引, 可以使用 list() 将其转成 list 后再索引
    root_value = tree[root_key]
    root_key_index = feature_list.index(root_key)
    for key in root_value.keys():
        if test_vec[root_key_index] == key:
            if type(root_value[key]).__name__ == 'dict':
                label = classify(root_value[key], feature_list, test_vec)
            else:
                label = root_value[key]
    return label


if __name__ == '__main__':
    dataset, feature_list = create_dataset()
    decision_tree = create_decision_tree(dataset, feature_list[:])

    result = classify(decision_tree, feature_list, [0, 0, 1, 1])

    if result == 'yes':
        print('放贷')
    else:
        print('不放贷')
