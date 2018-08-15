#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
from operator import *
import os
import nltk
import re
import math
import json

import solve_answer as solve

filename = []
path = "data/train/"


def scan_files():
    global filename
    filename = os.listdir(os.getcwd() + "/data/train/")


def read_file():
    """读取所有文件内容生成2元模型写入json文件"""
    sen_cnt = 0
    gram2 = dict()
    # 读文件
    ix = 1
    for file in filename:
        with open(path + file, 'r') as fp1:
            print "NO.{} {}".format(ix, file)
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
                # tmp.insert(0, "<bos>")
                # tmp.append("<eos>")
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
        ix += 1

    # 累加
    for out_key in gram2:
        cnt = 0
        for in_key in gram2[out_key]:
            cnt += gram2[out_key][in_key]
        gram2[out_key]['totalcnt'] = cnt

    js_obj = json.dumps(gram2)
    file_object = open('data/json_file_sec.json', 'w')
    file_object.write(js_obj)
    file_object.close()


def read_json():
    """读取json文件转化为dict作为2元模型"""
    file_object = open('data/json_file_sec.json', 'r')
    text = file_object.read()
    decode_json = json.loads(text)
    return decode_json


def read_title(path,num):
    """读取试题"""
    titles = []
    with open(path, "r") as f:
        for i in range(num):
            lst = []
            line = f.readline().replace("\n", "").split(" ")
            lst.extend(line)
            line = f.readline().replace("\n", "")
            lst.append(line)
            line = f.readline().replace("\n", "")
            lst.append(line)
            titles.append(lst)
    return titles


if __name__ == "__main__":
    """扫描文件名列表"""
    scan_files()

    """生成二元模型写入文件"""
    #read_file()

    """从文件读取到2元模型"""
    gram2 = read_json()

    # 语料库token数
    token_set = len(gram2.keys())
    # 平滑指数
    smooth = 0.5
    # 选项
    opt_letter = ["a", "b", "c", "d", "e"]

    """读取验证集试题和答案"""
    deve_path = "data/deve_set_sample.txt"
    test_path = "data/test_set_sample.txt"
    test_num = 800

    titles = read_title(test_path, test_num)
    answ_writer = open("data/test_answers.txt","w")

    """对每个题进行计算"""
    for lst in titles:
        title_no = lst[0]  # 题号
        #title_answer = lst[1]  # 答案
        title = lst[1]  # 题目
        title_options = lst[2].split(" ")  # 选项
        #print "{}".format(title_no)

        opt_prob = [1] * len(title_options)

        for k in range(len(title_options)):
            option = title.replace('_____', title_options[k])
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
                        cnt_down = token_set * smooth

                        # for j in range(len(gram2[token[i]])):
                        #     cnt_down += gram2[token[i]][gram2[token[i]].keys()[j]]
                        cnt_down += gram2[token[i]]["totalcnt"]
                        prob[i] = round(truediv(cnt_up, cnt_down), 6)
                    # 如果内部字典不含下一个词，则表示二元组没出现过，则进行平滑
                    else:
                        cnt_up = smooth
                        cnt_down = token_set * smooth
                        # for j in range(len(gram2[token[i]])):
                        #     cnt_down += gram2[token[i]][gram2[token[i]].keys()[j]]
                        cnt_down += gram2[token[i]]["totalcnt"]
                        prob[i] = round(truediv(cnt_up, cnt_down), 6)

                # 如果外部字典不含当前词，则表示该词没有出现过
                else:
                    cnt_up = smooth
                    cnt_down = token_set * smooth
                    prob[i] = round(truediv(cnt_up, cnt_down), 6)

            # result = round(math.log(reduce(mul, prob)), 4)
            # result = math.log(reduce(mul, prob))
            if len(prob) > 0:
                result = reduce(mul, prob)
            else:
                result = 0

            opt_prob[k] = result

        #print opt_prob
        string = "{} {} {}\n".format(title_no, opt_letter[opt_prob.index(max(opt_prob))], title_options[opt_prob.index(max(opt_prob))])
        answ_writer.write(string)
        #print string
    answ_writer.close()
    #solve.cal_accuracy()