#!/usr/bin/env python
# -*- coding: utf-8 -*-

import gensim
from collections import defaultdict
import pandas as pd
import datetime
starttime = datetime.datetime.now()

contents = defaultdict(list)  # 存放文章内容 key为文章日期号码 value为内容list
date_index_list = list()  # 存放所有文章的日期号码索引
page_index = ""  # 记录每一篇文章的日期号码

with open("199801_clear.txt", "r") as file:
    for line in file.readlines():
        if line.rstrip():  # 忽略空行
            line = line.decode("gbk")  # 将文件解码为gbk格式即可显示中文
            all_words = line.split()
            date_index = all_words[0].rsplit("-", 1)[0]  # 获取到文章的日期作为id索引
            contents[date_index] += all_words[1:]  # 累加获取整片文章的内容 此处若使用defaultdict则可以直接进行累加  否则需要使用循环累加
            if page_index != date_index:
                page_index = date_index
                date_index_list.append(page_index)

page_content_list = list()  # 将内容字典中的value提取出来作为一个list
for item in contents:
    page_content_list.append(contents[item])

# 对一些无关的词进行过滤
# 根据对语料库的分析  /w是标点符号  /y为"了""之"之类的无意词  /u为助词，如"的""着"  /c为连词，如"和"  /k多为 "们"
page_text = [
    [word for word in document if
     '/w' not in word and '/y' not in word and '/u' not in word and '/c' not in word and '/k' not in word]
    for document in page_content_list]

# 对只出现过一次的词进行过滤
times = defaultdict(int)
for page in page_text:
    for word in page:
        times[word] += 1
page_text = [[word for word in text if times[word] > 1] for text in page_text]

dictionary = gensim.corpora.Dictionary(page_text)  # 给所有的词例生成一个唯一的id标识
corpus = [dictionary.doc2bow(text) for text in page_text]  # 利用上一步的id标识给每片文章中的每个词进行计数

# 计算TFidf值，该值从普遍性和代表性方面进行了量化
tfidf = gensim.models.TfidfModel(corpus)
corpus_tfidf = tfidf[corpus]
result = gensim.similarities.MatrixSimilarity(corpus_tfidf)

data = pd.DataFrame(result[corpus_tfidf], index=date_index_list, columns=date_index_list)
data.to_csv("homework.csv")

output = open("result100.csv", "w")
# 前100个文档中最相似的30个文档
for i in range(0, 100):
    tmp = sorted(enumerate(result[corpus_tfidf[i]]), key=lambda x: x[1], reverse=True)
    result100 = []
    for j, m in tmp:
        result100.append([date_index_list[j], m])

    string = date_index_list[i] + ","
    for k in range(0, 30):
        string += str(result100[k][0]) + "->" + str(result100[k][1]) + ","
    string += "\r\n"
    output.write(string)

print "finish"
endtime = datetime.datetime.now()

print (endtime - starttime).seconds