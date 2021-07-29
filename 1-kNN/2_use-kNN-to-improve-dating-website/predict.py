import numpy as np
import os

from kNN import kNN
from data_process import get_data, normalization


def predict():
    result_list = ['讨厌', '有点喜欢', '非常喜欢']

    percent_video_games = float(input("玩视频游戏所消耗时间百分比: "))
    fly_miles = float(input("每年获得的飞行常客里程数: "))
    ice_cream = float(input("每周消费的冰淇淋公升数: "))

    input_x = np.array([fly_miles, percent_video_games, ice_cream])

    feature_mat, labels = get_data(os.path.join('./data', 'datingTestSet.txt'))
    feature_mat, x_min, x_range = normalization(feature_mat)  # 归一化

    result = kNN((input_x - x_min) / x_range, feature_mat, labels, 4)  # 测试数据也要归一化
    print('你可能%s这个人!' % (result_list[result - 1]))


if __name__ == '__main__':
    predict()
