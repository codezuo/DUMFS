#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：项目 
@File    ：LightGradientBoostingMachine.py
@Description：
@Author  ：heisenberg
@Date    ：2024/2/21 17:09 
'''

import joblib
from lightgbm import LGBMClassifier
from sklearn.utils import column_or_1d
from sklearn.model_selection import train_test_split
import pandas as pd
from final.MutiScore import muti_score
import final.ServiceDiscoveryDataPreprocessing as sddp

def LGBMClassifier(X_new,y):
    y = column_or_1d(y, warn=True)
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.25, random_state=2)
    # 模型训练
    gbm = LGBMClassifier(num_leaves=31,learning_rate=0.05,n_estimators=20)
    gbm.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=5)

    # 模型存储
    joblib.dump(gbm,'lightgbm_mode.pkl')
    # 模型加载
    gbm = joblib.load('lightgbm_mode.pkl')
    return muti_score(gbm,X_new,y)

# if __name__ == '__main__':
#     path = '../data/service_discovery_dataset_generation_0501_0704.csv'
#     X_minMax, y = sddp.data_procession(path)
#     X_minMax = pd.DataFrame(X_minMax)
#     filter_feature = {22, 23, 51, 37, 9, 4, 50, 47, 3, 49, 8, 48, 24, 33, 46, 38, 27, 45}
#     X_new = sddp.feature_selected_data(filter_feature, X_minMax)
#     LGBMClassifier(X_new, y)