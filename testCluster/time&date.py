# coding=utf-8


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model

# 热力图
import matplotlib.cm as cm
from matplotlib.colors import LogNorm


# Function to get data
def get_data(file_name):
    data = pd.read_csv(file_name)
    X_parameter = []
    Y_parameter = []
    Z_parameter = []
    for x, y, z in zip(data['x'], data['y'], data['z']):
        X_parameter.append(int(x))
        Y_parameter.append(int(y))
        Z_parameter.append(float(z))
    return X_parameter, Y_parameter, Z_parameter


# Function to show Thermodynamic diagram
def draw_thermodynamic_diagram(fileName,date):
    print(fileName)

    x, y, z = get_data(fileName)
    x_min = np.min(x)
    x_max = np.max(x)
    y_min = np.min(y)
    y_max = np.max(y)

    height = y_max - y_min + 1
    width = x_max - x_min + 1
    arr = np.zeros((height, width))  # arr 热力图中的值阵

    for i in range(len(x)):
        arr[y[i] - y_min, x[i] - x_min] = z[i]

    # 热力图默认左上为0,0
    # 所以热力图的显示和arr是一致的
    # 未解决以左下为0,0,
    plt.imshow(arr, extent=(np.amin(x), np.amax(x), np.amax(y), np.amin(y)),
               cmap=cm.hot, norm=LogNorm(), interpolation="nearest")
    plt.colorbar()
    plt.savefig('month'+str(date+1) +'.png')  # 先存，再show
    plt.draw()
    plt.close()

    return


for date in range(0, 12):
    fileName = 'data/month'+str(date+1)+'.csv'
    # for hour in range(9, 21):
    draw_thermodynamic_diagram(fileName, date)

