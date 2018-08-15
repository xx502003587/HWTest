#coding=utf-8

from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

lst1 = [1,2,3,4,5,6]
lst2 = [7,8,9]

np1 = pd.DataFrame(lst1)
np2 = pd.DataFrame(lst2)


n = np1.append(np2,ignore_index=True)
print n