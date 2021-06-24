#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   models.py    
@Contact :   519605144@qq.com
@License :   (C)Copyright 2017-2018, Liugroup-NLPR-CASIA

@Modify Time     ( @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/6/23 16:45   huanghao      1.0         None
'''
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
#公共算法，包括knn，randomforest， rbm，xgboost

class regress_models():
       def svm_reg_model(self):
              model = SVR(self)

       def knn__reg_model(self):
              model = KNeighborsRegressor(self)

       def rf_reg_model(self):
              model = RandomForestRegressor(self)

       def xgb_reg_model(self):
              model = XGBRegressor(self)

       def lgbm_reg_model(self):
              model = LogisticRegression(self)


class classify_models():
       def svm_clf_model(self):
              model = SVC(self)

       def knn_clf_model(self):
              model = KNeighborsClassifier(self)

       def log_clf_model(self):
              model = LogisticRegression(self)

       def rf_clf_model(self):
              model = RandomForestClassifier(self)

       def xgb_clf_model(self):
              model = XGBClassifier(self)

       def lgbm_clf_modek(self):
              model = LGBMClassifier(self)

