# -*- coding:utf-8 -*-
#
# 机器学习一个简单的归一化的类
# 当机器学习的基础是数据，而数据大都有着不止一个维度（x1,x2,x3, ... ,xn）
# 当不同维度之间的数值相差太大，以至于一个维度数值的改变量级在另一个维度面前不值一提的时候，就需要将数据归一化 或者 叫做标准化
# 将该变量转化为百分比的形式，使各个维度都能为数据的训练尽一份力。
# 同时还应该拥有反归一化的功能
# by 自由爸爸

import numpy as np
import pandas as pd


class Normalization_zybb:
    # 因为对数据归一化的方法各式各样，所以尽可能地用多个模块分别写
    def __init__(self, input_array):
        # input_array ：需要进行归一化处理的输入序列，类型：numpy array 对象。
        self.input_array = input_array
        self.array_normalized = input_array.astype(float)  # 用于存储归一化之后的序列（浮点型）
        self.ndim = input_array.ndim  # 维度 （仅仅限于处理1 维 或者二维的列表，再多就不行了）
        self.normalized_statu = False  # 数据是否归一化的判别

        self.linear_norm_parameter = None  # 离差标准化的参数（最大值 与 最小值 序列）
        self.z_score_norm_parameter = None  # z_score 标准化的参数（均值和标准差序列）

    # 2 -1 归一化的第二种方法
    def z_score_normalize(self):
        # Z-score标准化
        # 结果成正态分布
        # 公式： after_num = (input_num - 平均值mean ) / 标准差std
        # 首先验证数据
        if self.normalized_statu != False:
            print("-*- Z-score标准化：貌似已经进行了归一化处理，信息：{0}".format(self.normalized_statu))
            print("将不会进行再次归一化处理")
            return
        # 首先计算样本的均值和标准差
        if self.ndim == 2:
            self.z_score_norm_parameter = np.vstack((self.input_array.mean(0), self.input_array.std(0))).T  # 均值 与 标准差
        elif self.ndim == 1:
            self.z_score_norm_parameter = np.array([self.input_array.mean(), self.input_array.std()])  # 均值与标准差

        # 下面是归一化算法的核心
        if self.ndim == 2:
            # 2 维数据
            for i in range(len(self.z_score_norm_parameter)):
                # 前提得是 最大最小值不是相同的，否则分母为0，没有意义了
                std_num = self.z_score_norm_parameter[i][1]  # 方差
                if std_num != 0:
                    self.array_normalized[:, i] = (self.input_array[:, i] - self.z_score_norm_parameter[i][0]) / std_num
                else:
                    self.array_normalized[:, i] = self.input_array[:, i] - self.z_score_norm_parameter[i][0]
        elif self.ndim == 1:
            # 1 维数据
            std_num = self.z_score_norm_parameter[1]  # 方差
            if std_num != 0:
                self.array_normalized = (self.input_array - self.z_score_norm_parameter[0]) / std_num
            else:
                self.array_normalized = self.input_array - self.z_score_norm_parameter[0]

    # z_score 反归一化的函数
    def rev_z_score_normalize(self, predict_array):
        # 输入待反归一化的数据 ， 使用之前线性归一使用的参数，进行反归一化
        rev_array_normalized = predict_array.copy()
        if self.ndim == 1:  # 一维数组
            # 1 维数据
            max_del_min = self.z_score_norm_parameter[1]
            rev_array_normalized = predict_array * max_del_min + self.z_score_norm_parameter[0]

        elif self.ndim == 2:
            # 2 维数据
            for i in range(len(self.z_score_norm_parameter)):
                max_del_min = self.z_score_norm_parameter[i][1]
                rev_array_normalized[:, i] = predict_array[:, i] * max_del_min + self.z_score_norm_parameter[i][0]
        return rev_array_normalized


def test_z_score():
    # z_score 标准化的测试
    # d1 = np.array([[1, 2, 3], [0, 2, 1], [5, 2, 2], [10, 2, 0]])
    data_set = pd.read_csv("newcow.csv")
    input_features = np.array([data_set['activity'], data_set['position'], data_set['temperature'], data_set['walking']]).T
    # d1 = np.array([[1, 2, 3], [0, 2, 1], [5, 2, 2]])
    # d1 = np.array([1, 2, 3, 0, 2, 1, 5, 2, 2])
    # d1 = np.array([2, 2, 2])
    t1 = Normalization_zybb(input_features)
    t1.z_score_normalize()  #
    print(t1.array_normalized)
    # ok
    # 反归一化
    #print(t1.rev_z_score_normalize(t1.array_normalized))
    # 正常


if __name__ == '__main__':
    test_z_score()
