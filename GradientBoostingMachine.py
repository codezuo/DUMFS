#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：项目 
@File    ：GradientBoostingMachine.py
@Description：
@Author  ：heisenberg
@Date    ：2024/2/21 17:03 
'''
from sklearn.ensemble import GradientBoostingClassifier

import pandas as pd
from final.MutiScore import muti_score
import final.ServiceDiscoveryDataPreprocessing as sddp
def GBMClassifier(X_new,y):
    model = GradientBoostingClassifier(random_state=30)
    return muti_score(model, X_new, y)
# if __name__ == '__main__':
#     path = '../data/service_discovery_dataset_generation_0501_0704.csv'
#     X_minMax, y = sddp.data_procession(path)
#     X_minMax = pd.DataFrame(X_minMax)
#     filter_feature = {22, 23, 51, 37, 9, 4, 50, 47, 3, 49, 8, 48, 24, 33, 46, 38, 27, 45}
#     X_new = sddp.feature_selected_data(filter_feature, X_minMax)
#     GBMClassifier(X_new, y)