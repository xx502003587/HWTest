# -*- coding:utf-8 -*-
"""
# 机器学习系统设计 第二章 为Iris数据集分类
# Iris 数据集也叫作鸢尾花卉数据集，是最早应用于统计分类的数据集，书上这么说。数据集中共150个，并给出每一朵花的4个特征：花萼长度，花萼宽度，花瓣长度，花瓣宽度，并且给出该花所属的种类（山鸢尾花、变色鸢尾花、还是维吉尼亚鸢尾花）
# 目的是，当给出一朵新的花的上述4个特征（没见过的），可以通过上面的数据来判断该花是这三种花中的哪一种！
# 书中使用了统计学和类似于近邻算法的东西来解决问题，作为叩开机器学习大门的第一响。但我们应该明白，解决这中分类问题，我们总有更好的办法，且不止一种。
# 这是一种典型的有监督式（给出了特征值和所属类别）的分类问题。所以这里尝试使用lvq算法（学习向量量化神经网络）来解决。如果还不懂
# -*- by 自由爸爸 -*-
"""
# 1 数据准备
# <0> 首先一切的前提是准备数据集
import numpy as np
from sklearn.datasets import load_iris


def load_data():
    # data_set的格式是pandas集合
    data_set = load_iris()  # 载入原始数据

    input_features = data_set['data']  # 共计150个样本，每个样本4个特征值（依次分别代表花萼长度，花萼宽度，花瓣长度，花瓣宽度）
    print(type(input_features))
    output_categorys = data_set['target']  # 对应的类别（0，1,2）来表示
    print(type(output_categorys))
    return input_features, output_categorys
    # 我们只需要特征和所属类别就好，根据四个特征值（数字），来判断属于 0/1/2 中的哪一种


# <1> 准备思路 为了验证算法分类的准确性，我会从150个样本中取出一部分作为训练数据，留下的样本验证算法准确度
# 为了保证算法的准确，首先需要对数据进行归一化。如果你还不了解数据归一化，可以参考我的这篇文章，可以直接将最后归一化模块的源代码复制下来，粘贴到同目录下一个叫做norm.py（名字可变，只要能作为模块导入并调用就好）的文件里。
from norm import Normalization_zybb  # 导入归一化那篇文章的模块


def norm_input_data(input_features):
    Normer = Normalization_zybb(input_features)
    Normer.z_score_normalize()  # 调用z-score方法对输入特征进行归一化处理
    #Normer.linear_normalize()  # 调用线性归一化的方法对输入特征进行归一化处理
    return Normer.array_normalized  # 将归一化后的数据作为新的特征数据


features_raw, output_categorys = load_data()
normed_features = norm_input_data(features_raw)
print(normed_features, '\n', output_categorys)  # 特征和所属类别一一对应

train_features = np.concatenate((normed_features[:35], normed_features[50:85], normed_features[100:135]))  # 抽取一部分数据用于训练
confirm_features = np.concatenate((normed_features[35:50], normed_features[85:100], normed_features[135:]))  # 剩余的用作验证

train_output_categorys = np.concatenate(
    (output_categorys[:35], output_categorys[50:85], output_categorys[100:135]))  # 抽取一部分数据用于训练

confirm_output_categorys = np.concatenate(
    (output_categorys[35:50], output_categorys[85:100], output_categorys[135:]))  # 剩余的用作验证

# 至此 数据准备阶段的工作算是完成了,下面是本文的核心：lvq
# 2
# <0> 首先整理一下大致的思路：
# 首先明确地知道整个数据集可以分为3个种类：0/1/2，所以只需要定义3个输出节点，并且每一个节点特征数为4，且分别指定一个对应类别，这个没什么说的
# 然后就是调用lvq算法来训练数据了，我会将lvq那一套（所有需要的模块）从头至尾写出来，所以这篇文章会有点长

# <1> 开始吧
import random


class Lvq_solve_iris():
    # @in Lvq_solve_iris
    # 因为是模块化地编写，为了方便阅读和理解，我用这条注释“# @in Lvq_solve_iris”来表示当前函数是对象Lvq_solve_iris的内置函数
    def __init__(self, input_features, output_categorys):
        # 初始化Lvq_solve_iris对象
        self.input_features = input_features  # 特征值序列
        self.output_categorys = output_categorys  # 对应的种类
        self.feature_count = 4  # 特征数目为4
        self.category_count = 3  # 种类数目为3
        self.category_feature_list = []  # 存储着输出序列的特征
        self.step_size = 0.5  # 步长
        self.decrease_rate = 0.99  # 衰减率 用于衰减步长
        self.min_step_size = 0.10  # 最小步长

    # 随机生成三条输出端数据 分别代表三个种类
    # @in Lvq_solve_iris
    def initial_categorys(self):
        for i in range(self.category_count):
            self.category_feature_list.append(
                [random.random(), random.random(), random.random(), random.random(), i])  # 随机生成输出特征（前4位）与所属种类（最后的i）

    # 计算一朵花的输入特征，分别与三个输出特征之间的距离, 将距离存储于列表之中，并作为返回数据
    # @in Lvq_solve_iris
    def calc_distance_between(self, input_feature):
        distance_list = []
        for a_category in self.category_feature_list:
            the_distance = 0
            for i in range(self.feature_count):
                the_distance += (a_category[i] - input_feature[i]) ** 2
            distance_list.append(the_distance)  # 就不开根号2了
        return distance_list

    # 为了找到最近的输出节点，就要根据calc_distance_between计算得到的距离，找到最小值的索引。
    # 获取一个列表中最小值的索引
    # @in Lvq_solve_iris
    def get_min_index_in_list(self, a_list):
        return a_list.index(min(a_list))

    # 获取了离当前输入特征值最近的那个输出节点之后，判断输出节点的种类 ，如果二者种类相同，将输出节点向当前的输入节点靠拢，否则，将输出节点向远方调节
    # @in Lvq_solve_iris
    def move_output(self, min_distance_index, input_feature_index):  # 第一个参数为最近的那个输出节点的索引，第二个参数为100个输入节点中的某一个输入特征的索引
        the_closest_output = self.category_feature_list[min_distance_index]  # 最近的输出节点

        a_input_feature = self.input_features[input_feature_index]  # 当前那朵花的输入特征

        a_input_category = self.output_categorys[input_feature_index]  # 花的类别

        if (a_input_category == the_closest_output[-1]):  # 代表的是相同的种类 ，拉近输出节点
            for i in range(self.feature_count):
                the_closest_output[i] += self.step_size * (a_input_feature[i] - the_closest_output[i])
        else:  # 否则，拉远输出节点
            for i in range(self.feature_count):
                the_closest_output[i] -= self.step_size * (a_input_feature[i] - the_closest_output[i])
        self.category_feature_list[min_distance_index] = the_closest_output

    # 这个时候需要遍历所有的已知的数据 分别找到离之最近的输出节点并调节其位置
    # @in Lvq_solve_iris
    def a_lvq_loop(self):  # 遍历尽一遍所有的输入样本算是一次lvq算法
        for a_input_feature_index in range(len(self.input_features)):
            a_input_feature = self.input_features[a_input_feature_index]
            # 计算离三个输出节点的距离
            the_distance_list = self.calc_distance_between(a_input_feature)
            # 找到离之最近的输出节点的索引
            min_distance_index = self.get_min_index_in_list(the_distance_list)
            # 移动输出节点
            self.move_output(min_distance_index, a_input_feature_index)

    # 这里执行1500次lvq算法循环好了
    # @in Lvq_solve_iris
    def main_looper(self):
        print("训练之中……")
        for _ in range(1500):
            self.a_lvq_loop()
            # 衰减步长 且步长不应小于最小步长
            self.step_size *= self.decrease_rate
            if self.step_size < self.min_step_size:
                self.step_size = self.min_step_size


# if __name__ == '__main__':
LVQ = Lvq_solve_iris(train_features, train_output_categorys)
LVQ.initial_categorys()  # 随机初始化输出节点
LVQ.main_looper()
print(LVQ.category_feature_list)


# ok ! lvq主要算法逻辑已经编写完毕 ，那么我要考虑这么一个问题，执行lvq训练数据集的目的是什么？或者这么问，这里lvq算法得到的结论是什么，有什么用？
# 答 ： 我想要根据训练数据集，得出某种结论（这里是3个输出节点数据），然后当我输入新的样本的时候，可以根据训练所得到的结论（这里是3个输出节点数据），用公式计算得到新样本所属的种类。在这里，lvq得到的输出层3个特征数据可以作为三种花的平均特征，所以当新数据
# 进入，我只需要比较新数据与这3个节点之间的距离，比较得到离之最近的那个节点，然后我可以说：新数据的种类与这个最近节点的种类是相同的。
# 这也是我接下来要使用45个验证样本，要做的事情。看一下lvq算法的准确率！
def confirm_precision():
    """
       # 3 验证数据准确率
       还记得文章开始未加入训练的的特征样本集合吗：
       confirm_features 与 confirm_output_categorys（所对应类别），
       我要拿他们来验证数据
   """
    # output_feature_list = LVQ.category_feature_list     # 三个输出节点的特征与种类
    # print("三个特征的代表：{0}".format(output_feature_list))
    lvq_category_list = []
    for a_confirm_feature in confirm_features:
        # 计算得到最近那个输出节点的距离列表
        the_distance_list = LVQ.calc_distance_between(a_confirm_feature)
        min_distance_index = LVQ.get_min_index_in_list(the_distance_list)
        # 那么当前输入样本的节点应该与最近的输出节点的种类一样
        lvq_category_list.append(LVQ.category_feature_list[min_distance_index][-1])

    # lvq_category_list与准确的confirm_output_categorys比对，看一下错误率
    print(lvq_category_list)
    print(list(confirm_output_categorys))
    all_counts = len(lvq_category_list)
    right_counts = 0
    for i in range(all_counts):
        if lvq_category_list[i] == confirm_output_categorys[i]:
            right_counts += 1
    precision = right_counts / all_counts if all_counts != 0 else 0
    print("总数：{0}，正确数目{1}，准确率：{2}%".format(all_counts, right_counts, int(precision * 100)))


if __name__ == '__main__':
    confirm_precision()
