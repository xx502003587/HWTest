#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
import random
import numpy as np


def solve_develop_format():
    """
    处理验证集格式
    输出格式为：
    题号  答案
    题目
    选项
    """

    title_path = "data/development_set.txt"
    answer_path = "data/development_set_answers.txt"
    # path = "data/test_set.txt"

    deve_set = open("data/deve_set_sample.txt", "w")

    with open(title_path, "r") as f1:
        with open(answer_path, "r") as f2:
            for i in range(240):
                # 分解题目，拿出空白以及前后的词
                subject = f1.readline().replace("\n", "").split(" ")
                blank_ix = subject.index("_____")
                blank = subject[blank_ix - 1:blank_ix + 2]

                # 分解选项
                options = [0] * 5
                for i in range(5):
                    options[i] = f1.readline().replace("\n", "").split(" ")[-1]

                # 分解答案
                answer = f2.readline().replace("\n", "").split(" ")[1][1:2]

                string = "{} {}\n{}\n{}\n".format(subject[0][:-1], answer, " ".join(blank), " ".join(options))

                f1.readline()
                f1.readline()
                deve_set.write(string)
    deve_set.close()


def solve_test_format():
    """
    处理测试集格式
    输出格式为：
    题号
    题目
    选项
    """
    title_path = "data/test_set.txt"

    deve_set = open("data/test_set_sample.txt", "w")

    with open(title_path, "r") as f1:
        for i in range(800):
            # 分解题目，拿出空白以及前后的词
            subject = f1.readline().replace("\n", "").split(" ")
            blank_ix = subject.index("_____")
            blank = subject[blank_ix - 1:blank_ix + 2]

            # 分解选项
            options = [0] * 5
            for i in range(5):
                options[i] = f1.readline().replace("\n", "").split(" ")[-1]

            string = "{}\n{}\n{}\n".format(subject[0][:-1], " ".join(blank), " ".join(options))
            f1.readline()
            f1.readline()
            deve_set.write(string)
    deve_set.close()


def cal_accuracy():
    cnt = 0
    with open("data/deve_answers.txt", "r") as f:
        for i in range(240):
            answer = f.readline().replace("\n", "").split(" ")
            if answer[0] == answer[2]:
                cnt += 1
        print "{}%".format(round(cnt / 240 * 100, 2))


def general_random():
    letter = ["a", "b", "c", "d", "e"]
    return letter[random.randint(0, 3)]


def random_accuracy(rd):
    cnt = 0

    with open("data/deve_answers.txt", "r") as f:
        for i in range(240):
            answer = f.readline().replace("\n", "").split(" ")[0]
            if answer == rd[i]:
                cnt += 1
        return (cnt / 240) * 100


if __name__ == "__main__":
    # solve_develop_format()
    solve_test_format()
