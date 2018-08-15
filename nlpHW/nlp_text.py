#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
from operator import *
import os
import nltk
import re
import math
import json

filename = []
path = "data/train/"


def read_file():
    """读取所有文件内容生成2元模型写入json文件"""
    sen_cnt = 0
    gram2 = dict()
    # word_set = set()
    # 读文件
    with open("data/pdf.txt", 'r') as fp1:
        text = fp1.read()
        # 分句
        sens = nltk.sent_tokenize(text)
        sen_cnt += len(sens)
        # 过滤无关标点符号
        strinfo = re.compile('[^\'a-zA-Z,]')
        for i in range(len(sens)):
            sens[i] = strinfo.sub(" ", sens[i])

        # 分词，并增加句子开始结束标志
        words = []
        for sen in sens:
            tmp = nltk.word_tokenize(sen)
            words.append(tmp)

        # 计算二元组出现频数
        for item in words:
            # print item
            for i in range(len(item) - 1):
                if not gram2.has_key(item[i]):
                    tmp = dict()
                    tmp[item[i + 1]] = 1
                    gram2[item[i]] = tmp
                else:
                    if not gram2[item[i]].has_key(item[i + 1]):
                        gram2[item[i]][item[i + 1]] = 1
                    else:
                        gram2[item[i]][item[i + 1]] += 1

    #累加
    for out_key in gram2:
        cnt = 0
        for in_key in gram2[out_key]:
            cnt += gram2[out_key][in_key]
        gram2[out_key]['totalcnt'] = cnt


    return gram2


if __name__ == "__main__":
    gram2 = read_file()
    word_set = 11
    smooth = 0.5

    '''对测试语句进行分词处理'''
    test_sen = "John _____ a book"
    options = ["read", "eat", "hit"]
    opt_letter = ["A", "B", "C"]
    opt_prob = [1] * len(options)

    # test_sens = nltk.sent_tokenize(test_sen)

    # 过滤无关标点符号
    # strinfo = re.compile('[^\'a-zA-Z]')
    # test_sen = strinfo.sub(" ", test_sen)

    for k in range(len(options)):
        option = test_sen.replace('_____', options[k])
        # 分词
        token = nltk.word_tokenize(option)

        '''计算测试语句概率'''
        # 令p[i]表示条件概率 p( test[i+1] | test[i] ) = c(test [i & i+1]) / c( test[i] )
        prob = [1] * (len(token) - 1)

        for i in range(len(token) - 1):
            # 如果外部字典含有当前词
            if gram2.has_key(token[i]):
                # 并且内部字典含有下一个词，则直接进行计算
                if gram2[token[i]].has_key(token[i + 1]):
                    cnt_up = gram2[token[i]][token[i + 1]] + smooth
                    cnt_down = word_set * smooth

                    # for j in range(len(gram2[token[i]])):
                    #     cnt_down += gram2[token[i]][gram2[token[i]].keys()[j]]
                    cnt_down += gram2[token[i]]["totalcnt"]
                    prob[i] = round(truediv(cnt_up, cnt_down), 6)
                # 如果内部字典不含下一个词，则表示二元组没出现过，则进行平滑
                else:
                    cnt_up = smooth
                    cnt_down = word_set * smooth
                    # for j in range(len(gram2[token[i]])):
                    #     cnt_down += gram2[token[i]][gram2[token[i]].keys()[j]]
                    cnt_down += gram2[token[i]]["totalcnt"]
                    prob[i] = round(truediv(cnt_up, cnt_down), 6)

            # 如果外部字典不含当前词，则表示该词没有出现过
            else:
                cnt_up = smooth
                cnt_down = word_set * smooth
                prob[i] = round(truediv(cnt_up, cnt_down), 6)

        # result = round(math.log(reduce(mul, prob)), 4)
        if len(prob) > 0:
            result = round(reduce(mul, prob), 6)
        else:
            result = 0

        opt_prob[k] = result

    print opt_prob
    print "{} {}".format(opt_letter[opt_prob.index(max(opt_prob))], options[opt_prob.index(max(opt_prob))])
