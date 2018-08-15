# coding=utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

name = ['201601-citibike-tripdata.csv', '201602-citibike-tripdata.csv', '201603-citibike-tripdata.csv',
        '201604-citibike-tripdata.csv', '201605-citibike-tripdata.csv', '201606-citibike-tripdata.csv',
        '201607-citibike-tripdata.csv', '201608-citibike-tripdata.csv', '201609-citibike-tripdata.csv',
        '201610-citibike-tripdata.csv', '201611-citibike-tripdata.csv', '201612-citibike-tripdata.csv']

readPath = 'K:/workspace/PycharmProjects/test/data/'

'''4  [5]start纬度 [6]start经度 8 [9]end纬度 [10]end经度'''


def cal():
    dt = dict()
    for i in range(len(name)):
        data = pd.read_csv(readPath + name[i])
        for row in data.values:
            if not dt.has_key(row[4]):
                lst = list()
                lst.append(row[5])
                lst.append(row[6])
                dt[row[4]] = lst

            if not dt.has_key(row[8]):
                lst = list()
                lst.append(row[9])
                lst.append(row[10])
                dt[row[8]] = lst

    return dt


def write(dt):
    # lst = list()
    file = open("long.csv", "wb")

    for i in range(len(dt.keys())):
        tmp = dt.keys()[i] + "," + str(dt[dt.keys()[i]][0]) + "," + str(dt[dt.keys()[i]][1]) + "\r\n"
        file.write(tmp)

    file.close()


def test():
    dt = dict()
    dt["0"] = [0, 1]
    dt["1"] = [1, 2]
    dt["2"] = [3, 2]
    dt["3"] = [3, 4]

    for i in range(len(dt.keys())):
        tmp = dt.keys()[i] + "," + str(dt[dt.keys()[i]][0]) + "," + str(dt[dt.keys()[i]][1])
        print tmp


def read():
    file = pd.read_csv("long4.csv")
    # for row in file.values:
    print file.columns
    print type(file)

    plt.rc('font', family='STXihei', size=10)
    plt.scatter(file['longitude'], file['latitude'], 50, color='blue', marker='+', linewidth=2, alpha=0.8)
    plt.xlabel(u'经度')
    plt.ylabel(u'纬度')
    plt.axis([-74.09, -73.92, 40.64, 40.81])

    plt.grid(color='#95a5a6', linestyle='--', linewidth=1, axis='both', alpha=0.4)
    plt.show()
    # plt.savefig("map1.png")
    # plt.draw();
    # plt.close()


def write():
    reader = pd.read_csv("long.csv")
    writer = open("long4.csv", 'wb')
    lst = list()
    for row in reader.values:
        row[1] = round(row[1], 4)
        row[2] = round(row[2], 4)
        tmp = row[0] + "," + str(row[1]) + "," + str(row[2]) + "\r\n"
        writer.write(tmp)
    writer.close()


def cluster():
    data = pd.DataFrame(pd.read_csv('loc_info.csv'))

    dataSet = []
    staName = []
    for row in data.values:
        staName.append(row[0])
        dataSet.append([row[1], row[2]])

    k=60
    clf = KMeans(n_clusters=k)
    s = clf.fit(dataSet)
    numSamples = len(dataSet)
    centroids = clf.labels_

    mark = ['+r', '+g', '+b', '+c', '+m', '+y', '+k',
            'xr', 'xg', 'xb', 'xc', 'xm', 'xy', 'xk',
            '*r', '*g', '*b', '*c', '*m', '*y', '*k',
            'pr', 'pg', 'pb', 'pc', 'pm', 'py', 'pk',
            '^r', '^g', '^b', '^c', '^m', '^y', '^k']
    # 画出所有样例点 属于同一分类的绘制同样的颜色
    for i in xrange(numSamples):
        plt.plot(dataSet[i][1], dataSet[i][0], mark[clf.labels_[i]])  # mark[markIndex])

    plt.rcParams['font.sans-serif'] = ['consolas']
    plt.title({'fontsize': 15})
    plt.xlabel({'fontsize': 15})
    plt.ylabel({'fontsize': 15})
    plt.title(str(k) + " clusters")
    plt.xlabel('longitude')
    plt.ylabel('latitude')
    path = "img/cluster" + str(k) + '.png'

    plt.draw()
    plt.savefig(path)
    plt.close()

        #write_cluster_station(staName, dataSet, clf, s, numSamples, centroids)


def write_cluster_station(staName, dataSet, clf, s, numSamples, centroids):
    main_lst = []
    for i in range(30):
        tmp = []
        main_lst.append(tmp)

    for i in range(numSamples):
        main_lst[clf.labels_[i]].append(staName[i])

    tt = dict()
    for i in range(30):
        tt[str(i)] = 0
    for i in range(numSamples):
        tt[str(clf.labels_[i])] += 1
    print tt.values()

    max = find_max(tt)

    for i in range(len(main_lst)):

        rest = max - len(main_lst[i])
        for k in range(rest):
            main_lst[i].append("null")
        print main_lst[i]
        print

    value = dict()
    for i in range(30):
        value["cluster" + str(i)] = main_lst[i]

    dataframe = pd.DataFrame(value)
    dataframe.to_csv("test.csv", index=False)


def find_max(tt):
    max = 0
    for i in range(len(tt)):
        if tt[str(i)] > max:
            max = tt[str(i)]

    return max


if __name__ == '__main__':
    # dt = cal()
    # write(dt)
    # read()
    # write()
    cluster()

    # write_cluster_station()
