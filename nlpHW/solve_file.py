#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import shutil


path = "data/Training_Data/"
write_path = "data/train/"
shutil.rmtree(write_path)
os.mkdir(write_path)

filename = os.listdir(os.getcwd() + "/data/Training_Data/")

i = 1
for file in filename:
    with open(path + file, 'r') as fp1:
        print "{} {}".format(i, file)
        text = fp1.read()
        text = text.split("END")[3][1:]
        fp2 = open(write_path + file, "w")
        fp2.write(text)
        fp2.close()
        i += 1
