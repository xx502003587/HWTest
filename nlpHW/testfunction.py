#!/usr/bin/env python
# -*- coding: utf-8 -*-
import nltk
import re

all = []

with open("data/deve_set_sample.txt","r") as f:
    for i in range(240):
        lst = []
        line = f.readline().replace("\n","").split(" ")
        lst.extend(line)
        line = f.readline().replace("\n","")
        lst.append(line)
        line = f.readline().replace("\n","")
        lst.append(line)
        all.append(lst)
        print lst
print "s"
