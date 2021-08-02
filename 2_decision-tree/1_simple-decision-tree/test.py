from data_process import create_dataset

import pickle  # 用于序列化对象, 将其保存到磁盘上


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
    _, feature_list = create_dataset()
    with open('./model/decision_tree', mode='rb') as fp:
        tree = pickle.load(fp)  # 反序列化为对象

    result = classify(tree, feature_list, [0, 0, 1, 1])

    if result == 'yes':
        print('放贷')
    else:
        print('不放贷')
