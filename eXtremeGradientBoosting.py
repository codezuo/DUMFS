#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：项目 
@File    ：eXtremeGradientBoosting.py
@Description：
@Author  ：heisenberg
@Date    ：2024/2/21 17:11 
'''
from xgboost import XGBClassifier
import pandas as pd
from final.MutiScore import muti_score
import final.ServiceDiscoveryDataPreprocessing as sddp
def XgBoostClassifier(X_new,y):
    # xgboost实现二分类 对比的是0和1 而数据集中是-1和1因此需要将-1变为0
    y[y==-1]= 0
    model = XGBClassifier(random_state=30)
    return muti_score(model, X_new, y)

# if __name__ == '__main__':
#     path = '../data/service_discovery_dataset_generation_0501_0704.csv'
#     X_minMax, y = sddp.data_procession(path)
#     X_minMax = pd.DataFrame(X_minMax)
#     filter_feature = {22, 23, 51, 37, 9, 4, 50, 47, 3, 49, 8, 48, 24, 33, 46, 38, 27, 45}
#     X_new = sddp.feature_selected_data(filter_feature, X_minMax)
#     XgBoostClassifier(X_new, y)