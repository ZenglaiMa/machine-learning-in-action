import numpy as np
import operator  # 运算符模块, kNN算法执行排序操作时用到这个模块提供的函数


def create_dataset():
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0.0, 0.0], [0.0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify(input_x, dataset, labels, k):
    dataset_size = len(dataset)
    # np.tile(a, (y, x)): 将a在Y轴方向复制y次, 在X轴方向复制x次, 返回维度不变.
    # 这样将input_x变成一个矩阵, 便于计算其和每一个样本的距离.
    input_mat = np.tile(input_x, (dataset_size, 1))
    # 计算欧氏距离
    distances = (np.sum((input_mat - dataset) ** 2, axis=1)) ** 0.5
    # argsort()返回从小到大排序的索引值
    sorted_distance_ids = distances.argsort()
    # {key=class: value=count}
    class_count = {}
    for i in range(k):
        key = labels[sorted_distance_ids[i]]
        # {}.get(key, default): 返回指定key的value, 若不存在返回default
        class_count[key] = class_count.get(key, 0) + 1
    # key = operator.itemgetter(1)根据字典的值进行排序
    # key = operator.itemgetter(0)根据字典的键进行排序
    # reverse=True 降序排序
    sorted_class_count = sorted(class_count.items(), reverse=True, key=operator.itemgetter(1))
    return sorted_class_count[0][0]


if __name__ == '__main__':
    group, labels = create_dataset()
    result = classify([0, 0.5], group, labels, 3)
    print(result)
