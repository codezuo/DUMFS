#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：项目 
@File    ：LogisticRegression.py
@Description：
@Author  ：heisenberg
@Date    ：2024/2/21 16:04 
'''
from sklearn.linear_model import LogisticRegression
import pandas as pd
from final.MutiScore import muti_score
import final.ServiceDiscoveryDataPreprocessing as sddp


def logistic_regression(X_new,y):
    lr = LogisticRegression()
    return muti_score(lr,X_new,y)
# if __name__ == '__main__':
#     path = '../data/service_discovery_dataset_generation_0501_0704.csv'
#     X_minMax, y = sddp.data_procession(path)
#     X_minMax = pd.DataFrame(X_minMax)
#     filter_feature = {22, 23, 51, 37, 9, 4, 50, 47, 3, 49, 8, 48, 24, 33, 46, 38, 27, 45}
#     X_new = sddp.feature_selected_data(filter_feature, X_minMax)
#     logistic_regression(X_new, y)
