import pandas as pd
import os
import numpy as np
# %%
path = os.path.abspath('')
data = pd.read_csv('train.csv')

# %%
data_variables = data.columns
data_shape = data.shape
data_describe = data.describe()
x_data = data.drop(['Survived', 'PassengerId'], axis=1)
y_data = data['Survived']

data_null_rate = data.isnull().sum() / data.shape[0]

data_null_rate.apply(lambda x: True if x > 0.5 else False)


# %%
# 如果丢失率大于80%，则删除该变量
# 其他的填充方式，默认knn，提供众数，中位数等填充

def data_null(data, type):
       data_null_rate = data.isnull().sum() / data.shape[0]
       for index, value in enumerate(data_null_rate):
              if value > 0.6:
                     data_index = data_null_rate.index[index]
                     data.drop(data_index, axis=1, inplace=True)
              else:
                     pass
       return data

from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=5)
x_s_data = x_data.select_dtypes(include=['object'])
x_d_data = x_data.select_dtypes(exclude=['object'])
for x in x_s_data.columns:
       classs_mapping = {label:idx for idx, label in enumerate(set(x_s_data[x]))}
       x_s_data[x] = x_s_data[x].map(classs_mapping)
print(x_s_data)
# print(imputer.fit_transform(x_d_data))



