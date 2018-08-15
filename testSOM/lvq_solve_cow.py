# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import random
from norm import Normalization_zybb

test_feature = False

# 输入：[ 活动因子，卧姿状态，体表温度，高步行次数 ]
# 输出：五种发情情况的代表节点数据

col1 = 'position'
col2 = 'temperature'
col3 = 'walking'


def load_data():
    # data_set的格式是pandas集合
    data_set = pd.read_csv("newcow.csv")  # 载入原始数据
    '''更改特征个数'''
    if test_feature:
        input_features = np.array([data_set[col1], data_set[col2], data_set[col3]]).T
    else:
        input_features = np.array(
            [data_set['activity'], data_set['position'], data_set['temperature'], data_set['walking']]).T

    output_categorys = np.array([data_set['test']]).T  # 对应的类别（0,1,2,3,4）来表示(自然状态,活动量1级,活动量2级,活动量3级,发情)

    return input_features, output_categorys


# 为了验证算法分类的准确性，从72个样本中取出一部分(两种数据样本分别取25)作为训练数据，留下的样本(22条)验证算法准确度
# 对数据进行归一化
def norm_input_data(input_features):
    normer = Normalization_zybb(input_features)
    normer.z_score_normalize()  # 调用z-score方法对输入特征进行归一化处理
    return normer.array_normalized  # 将归一化后的数据作为新的特征数据


features_raw, output_categorys = load_data()  # 加载数据
normed_features = norm_input_data(features_raw)  # 归一化数据

# print(normed_features, '\n', output_categorys)  # 特征和所属类别一一对应

# 抽取一部分数据用于训练(36=25+11)
train_features = np.concatenate((normed_features[:25], normed_features[36:42], normed_features[43:46],
                                 normed_features[49:56], normed_features[61:70]))
# 剩余的用作验证
confirm_features = np.concatenate((normed_features[25:36], normed_features[42:43], normed_features[46:49],
                                   normed_features[56:61], normed_features[70:72]))
# 抽取一部分数据用于训练
train_output_categorys = np.concatenate((output_categorys[:25], output_categorys[36:42], output_categorys[43:46],
                                         output_categorys[49:56], output_categorys[61:70]))
# 剩余的用作验证
confirm_output_categorys = np.concatenate((output_categorys[25:36], output_categorys[42:43], output_categorys[46:49],
                                           output_categorys[56:61], output_categorys[70:72]))


class Lvq_Solve_Cow():
    def __init__(self, input_features, output_categorys):
        self.input_features = input_features  # 特征值序列
        self.output_categorys = output_categorys  # 对应的种类

        '''更改特征个数'''
        if test_feature:
            self.feature_count = 3  # 特征数目为4
        else:
            self.feature_count = 4  # 特征数目为4
        self.category_count = 5  # 种类数目为5
        self.category_feature_list = []  # 存储着输出序列的特征
        self.step_size = 0.5  # 步长
        self.decrease_rate = 0.99  # 衰减率 用于衰减步长
        self.min_step_size = 0.10  # 最小步长

    # 随机生成输出特征（前4位）与所属种类（最后的i）
    def initial_categorys(self):
        for i in range(self.category_count):
            '''更改特征个数'''
            if test_feature:
                self.category_feature_list.append([random.random(), random.random(), random.random(), i])
            else:
                self.category_feature_list.append(
                    [random.random(), random.random(), random.random(), random.random(), i])

    # 分别计算一头奶牛的输入特征与五个输出特征之间的距离, 将距离存储于列表之中，并作为返回数据
    def calc_distance_between(self, input_feature):
        distance_list = []
        for a_category in self.category_feature_list:
            the_distance = 0
            for i in range(self.feature_count):
                the_distance += (a_category[i] - input_feature[i]) ** 2
            distance_list.append(the_distance)  # 暂时不开根号
        return distance_list

    # 为了找到最近的输出节点，就要根据calc_distance_between计算得到的距离，找到最小值的索引。
    # 获取一个列表中最小值的索引
    def get_min_index_in_list(self, a_list):
        return a_list.index(min(a_list))

    # 获取了离当前输入特征值最近的那个输出节点之后，判断输出节点的种类 ，如果二者种类相同，将输出节点向当前的输入节点靠拢，否则，将输出节点向远方调节
    # 第一个参数为最近的那个输出节点的索引，第二个参数为输入节点中的某一个输入特征的索引
    def move_output(self, min_distance_index, input_feature_index):
        the_closest_output = self.category_feature_list[min_distance_index]  # 最近的输出节点

        a_input_feature = self.input_features[input_feature_index]  # 当前奶牛的输入特征
        a_input_category = self.output_categorys[input_feature_index]  # 奶牛的类别

        if a_input_category == the_closest_output[-1]:  # 代表的是相同的种类 ，拉近输出节点
            for i in range(self.feature_count):
                the_closest_output[i] += self.step_size * (a_input_feature[i] - the_closest_output[i])

        else:  # 否则，拉远输出节点
            for i in range(self.feature_count):
                the_closest_output[i] -= self.step_size * (a_input_feature[i] - the_closest_output[i])

        self.category_feature_list[min_distance_index] = the_closest_output

    # 遍历所有的已知的数据 分别找到离之最近的输出节点并调节其位置
    # 遍历一遍所有的输入样本算是一次lvq算法
    def a_lvq_loop(self):
        for a_input_feature_index in range(len(self.input_features)):
            a_input_feature = self.input_features[a_input_feature_index]
            # 计算离五个输出节点的距离
            the_distance_list = self.calc_distance_between(a_input_feature)
            # 找到离之最近的输出节点的索引
            min_distance_index = self.get_min_index_in_list(the_distance_list)
            # 移动输出节点
            self.move_output(min_distance_index, a_input_feature_index)

    # 这里执行1500次lvq算法循环好了
    def main_looper(self):
        # print("训练中……")
        for _ in range(1500):
            self.a_lvq_loop()
            # 衰减步长 且步长不应小于最小步长
            self.step_size *= self.decrease_rate
            if self.step_size < self.min_step_size:
                self.step_size = self.min_step_size


# 使用22个验证样本，看一下准确率
def confirm_precision():
    lvq_category_list = []
    for a_confirm_feature in confirm_features:
        # 计算得到最近那个输出节点的距离列表
        the_distance_list = LVQ.calc_distance_between(a_confirm_feature)
        min_distance_index = LVQ.get_min_index_in_list(the_distance_list)
        # 当前输入样本的节点应该与最近的输出节点的种类一样
        lvq_category_list.append(LVQ.category_feature_list[min_distance_index][-1])

    # lvq_category_list与准确的confirm_output_categorys比对，看一下错误率
    # print(lvq_category_list)
    # print(list(confirm_output_categorys))
    all_counts = len(lvq_category_list)
    right_counts = 0
    for i in range(all_counts):
        if lvq_category_list[i] == confirm_output_categorys[i]:
            right_counts += 1
    precision = right_counts / all_counts if all_counts != 0 else 0
    print("总数：{0}，正确数目{1}，准确率：{2}%".format(all_counts, right_counts, int(precision * 100)))
    return float(precision)


sum = 0
for _ in range(100):
    LVQ = Lvq_Solve_Cow(train_features, train_output_categorys)
    LVQ.initial_categorys()  # 随机初始化输出节点
    LVQ.main_looper()
    sum += confirm_precision()
print("总准确率：{0}%".format(int(sum / 100 * 100)))
# print(LVQ.category_feature_list)


# if __name__ == '__main__':
#     confirm_precision()
