import pandas as pd
import os
import numpy as np

#%%
path = os.path.abspath('')
data = pd.read_csv('train.csv')


#%%
variables = data.columns
