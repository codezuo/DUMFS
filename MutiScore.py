#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：项目 
@File    ：MutiScore.py
@Description：
@Author  ：heisenberg
@Date    ：2024/2/21 16:05 
'''

import warnings
from sklearn.model_selection import cross_val_score
from time import time

def muti_score(model,X_new,y):
    time_start = time()
    warnings.filterwarnings('ignore')
    accuracy = cross_val_score(model, X_new, y, scoring='accuracy', cv=5)
    precision = cross_val_score(model, X_new, y, scoring='precision', cv=5)
    recall = cross_val_score(model, X_new, y, scoring='recall', cv=5)
    f1_score = cross_val_score(model, X_new, y, scoring='f1', cv=5)
    auc = cross_val_score(model, X_new, y, scoring='roc_auc', cv=5)
    print("准确率:",accuracy.mean())
    print("精确率:",precision.mean())
    print("召回率:",recall.mean())
    print("F1_score:",f1_score.mean())
    print("AUC:",auc.mean())
    time_end = time()
    consumeTime = time_end - time_start
    print('模型测试二分类消耗时间：', consumeTime)
    result = {}
    result[0] = accuracy.mean()
    result[1] = precision.mean()
    result[2] = recall.mean()
    result[3] = f1_score.mean()
    result[4] = auc.mean()
    result[5] = consumeTime
    result[6] = model
    return result