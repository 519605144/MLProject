#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   data_proccess.py    
@Contact :   519605144@qq.com
@License :   (C)Copyright 2017-2018, Liugroup-NLPR-CASIA

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/6/16 16:47   huanghao      1.0         None
'''
import numpy as np
import pandas as pd
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing

class fillna_method():
       def fillna_mean(self, data):
              return self.fillna(data, inplace=True)

       # 填补缺失值随机森林
       def fillna_forest(self):
              pass

class continuous_data_process(fillna_method):
       #连续数据处理
       def __init__(self):
              data = datasets.load_boston()

       # 数据标准化
       def standardard_data(self, *args, **kwargs):

              result = preprocessing.scale(self)
              return result

       # 数据归一化
       def normalize_data(self, *args, **kwargs):
              result = preprocessing.normalize(self)
              return result

       # 数据异常点分位数
       def outlier_box(self, *args, **kwargs):
              Q1 = self.quantile(q=0.25)
              Q3 = self.quantile(q=0.75)
              low_whisker = Q1 - 1.5 * (Q3 - Q1)
              up_whisker = Q3 + 1.5 * (Q3 - Q1)
              all_data = self[((self > up_whisker) | (self < low_whisker))]
              all_data_index = all_data.index
              return all_data, all_data_index

       # 数据异常点3sigam
       def outlier_sigma(self, *args, **kwargs):
              mean_value = self.mean()
              std_value = self.std()
              all_data = self[(mean_value - 3 * std_value > self) | (mean_value + 3 * std_value < self)]
              all_data_index = all_data.index
              return all_data, all_data_index

       # 填补缺失值平均值


       #  数据分箱
       def opt_binning(self):
              pass

class discrete_data_proccess(fillna_method):
       def dummy_decode(self):
              result = pd.get_dummies(self)
              return result

class time_data_proccess():
       def time_info_extract(self):
              pass

boston_price = datasets.load_boston()
clip_board = pd.read_clipboard()
x_data = boston_price.data
y_data = boston_price.target
tss1 = '2013-10-10 23:40:00'



