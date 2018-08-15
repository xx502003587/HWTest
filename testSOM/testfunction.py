# -*- coding:utf-8 -*-

import pandas as pd
import numpy as np

# state_list = ["N", "I", "II", "III", "O"]
# state_number = [0, 1, 2, 3, 4]
#
# data = pd.read_csv("cow.csv")
# print(data.columns)
#
# data['test'] = data.apply(lambda x: state_number[state_list.index(x.state)], axis=1)
#
# data.drop(['state'], axis=1, inplace=True)
# data.drop(['id'], axis=1, inplace=True)
#
# # print(type(data['id']))
# # fo = np.array([data['id'],data['test']])
# # print(fo)
# data.to_csv("newcow.csv", header=1, index=False)
#
# data = pd.read_csv("newcow.csv")
# a2 = np.array([data['activity'],data['position'],data['temperature'],data['walking']])
# print(a2.T)
data_set = pd.read_csv("newcow.csv")  # 载入原始数据
input_features = np.array(
    [data_set['activity'], data_set['position'], data_set['temperature'], data_set['walking']]).T
output_categorys = np.array([data_set['test']]).T  # 对应的类别（0,1,2,3,4）来表示(自然状态,活动量1级,活动量2级,活动量3级,发情)

print(input_features[61:70])