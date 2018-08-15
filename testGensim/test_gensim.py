#!/usr/bin/env python
# -*- coding: utf-8 -*-

from gensim import corpora, models, similarities
from collections import defaultdict
import pandas as pd
import datetime
starttime = datetime.datetime.now()


with open("199801_clear.txt", "r") as file:
    doc_cont = defaultdict(list)
    date_indexs = list()
    index = ""
    for line in file.readlines():
        if line.rstrip():
            line = line.decode("gbk")
            all_words = line.split()
            date_index = all_words[0].rsplit("-", 1)[0]
            doc_cont[date_index] += all_words[1:]
            if index != date_index:
                index = date_index
                date_indexs.append(index)

documents = list()
for item in doc_cont:
    documents.append(doc_cont[item])

ptexts = [
    [word for word in document if
     '/w' not in word and '/y' not in word and '/u' not in word and '/c' not in word and '/k' not in word]
    for document in documents]

times = defaultdict(int)
for page in ptexts:
    for word in page:
        times[word] += 1
ptexts = [[word for word in text if times[word] > 1] for text in ptexts]

dictionary = corpora.Dictionary(ptexts)
corpus = [dictionary.doc2bow(text) for text in ptexts]

tfidf = models.TfidfModel(corpus)
corpus_tfidf = tfidf[corpus]
result = similarities.MatrixSimilarity(corpus_tfidf)

data = pd.DataFrame(result[corpus_tfidf], index=date_indexs, columns=date_indexs)
data.to_csv("text_result.csv")

output = open("text_result_100.csv", "w")

for i in range(0, 100):
    tmp = sorted(enumerate(result[corpus_tfidf[i]]), key=lambda x: x[1], reverse=True)
    result100 = []
    for j, m in tmp:
        result100.append([date_indexs[j], m])

    string = date_indexs[i] + ","
    for k in range(0, 30):
        string += str(result100[k][0]) + "#" + str(result100[k][1]) + ","
    string += "\r\n"
    output.write(string)

print "finish"
endtime = datetime.datetime.now()
print (endtime - starttime).seconds
