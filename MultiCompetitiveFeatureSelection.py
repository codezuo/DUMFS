#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：项目 
@File    ：MultiCompetitiveFeatureSelection.py
@Description：
@Author  ：heisenberg
@Date    ：2024/2/21 17:29 
'''

import math
from sklearn.feature_selection import RFECV
from sklearn.model_selection import train_test_split
from sklearn.utils import column_or_1d
from final.MutiScore import muti_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
import final.ServiceDiscoveryDataPreprocessing as sddp
import copy
import seaborn as sns #图表模块


# 皮尔森系数相关性画图
def pearson_drawing(X_minMax):
   colormap = plt.cm.RdBu  # 绘图库中的颜色查找表。比如A1是红色,A2是浅蓝色。 这样一种映射关系
   plt.figure(figsize=(60, 60))  # 创建一个新的图表，参数是尺寸，单位为英寸。
   plt.title('Pearson Correlation of Features', y=1.05, size=15)  # 给图表一个标题~~
   sns.heatmap(X_minMax.astype(float).corr(), linewidths=0.1, vmax=1.0, square=True, cmap=colormap, linecolor='white',
               annot=True)  # 将皮尔森系数值画成图表形式。
   # 保存图片到本地
   plt.savefig('Pearson_Correlation_of_Features_0501_0704.jpg')
   plt.show()
   result = X_minMax.astype(float).corr(name='pearson')
   print(result)

#计算每个特征的spearman系数，返回数组
def calcAttribute(data):
    prr = []
    n = data.shape[0]
    m = data.shape[1]
    x = [0] * n             #初始化特征x和类别y向量
    y = [0] * n
    for i in range(n):      #得到类向量
        y[i] = data[i][m-1]
    for j in range(m-1):    #获取每个特征的向量，并计算Pearson系数，存入到列表中
        for k in range(n):
            x[k] = data[k][j]
        prr.append(calcPearson(x,y))
    return prr
# Pearson feature selection
def calcPearson(x,y):
    x_mean,y_mean = calcMean(x,y)	#计算x,y向量平均值
    n = len(x)
    sumTop = 0.0
    sumBottom = 0.0
    x_pow = 0.0
    y_pow = 0.0
    for i in range(n):
        sumTop += (x[i]-x_mean)*(y[i]-y_mean)
    for i in range(n):
        x_pow += math.pow(x[i]-x_mean,2)
    for i in range(n):
        y_pow += math.pow(y[i]-y_mean,2)
    sumBottom = math.sqrt(x_pow*y_pow)
    p = sumTop/sumBottom
    return p
	#计算特征和类的平均值
def calcMean(x,y):
    sum_x = sum(x)
    sum_y = sum(y)
    n = len(x)
    x_mean = float(sum_x+0.0)/n
    y_mean = float(sum_y+0.0)/n
    return x_mean,y_mean
# 获取排序后的pearson系数的特征下标
def calcAttribute(data):
    pearsonSelectedFeatures = []
    n = data.shape[0]    #获取数据集行数和列数
    m = data.shape[1]
    x = [0] * n             #初始化特征x和类别y向量
    y = [0] * n
    for i in range(n):      #得到类向量
        y[i] = data[i][m-1]
    for j in range(m-1):    #获取每个特征的向量，并计算Pearson系数，存入到列表中
        for k in range(n):
            x[k] = data[k][j]
        pearsonSelectedFeatures.append(calcPearson(x,y))
    # 求m个最大的数值及其索引
    t = copy.deepcopy(pearsonSelectedFeatures)
    max_number = []
    # selected max 18 features
    max_index = []
    for _ in range(18):
        number = max(t)
        index = t.index(number)
        t[index] = 0
        max_number.append(number)
        max_index.append(index)
    # print(max_number)
    # print(max_index)
    return max_index
# TMFS feature selection
def tree_based_feature_selection(X,y):
    forest = ExtraTreesClassifier(n_estimators=250,
                                  random_state=0)
    y = column_or_1d(y, warn=True)
    forest.fit(X, y)
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]
    # Print the feature ranking
    print("Feature ranking:")
    ranks = []
    for f in range(X.shape[1]):
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
        # 向字典中添加数据 即特征和特征对应的重要性程度
        ranks.append(indices[f])
        #ranks[indices[f]] = importances[indices[f]]
    # Plot the feature importances of the forest
    plt.figure(figsize=(30, 30))
    plt.title("Feature importances")
    plt.bar(range(X.shape[1]), importances[indices],
            color="r", yerr=std[indices], align="center")
    plt.xticks(range(X.shape[1]), indices)
    plt.xlim([-1, X.shape[1]])
    plt.show()
    TMFSSelectedFeatures = ranks[:18]
    return TMFSSelectedFeatures
# RFE Feature Selection
def RFESelection(X,y):
    estimator = DecisionTreeClassifier(random_state=111)
    # 5折交叉
    selector = RFECV(estimator, step=1, cv=5)
    selector = selector.fit(X, y)
    ranking = selector.ranking_
    # RFE selected features
    selectedFeatures = []
    for index in range(0,len(ranking)):
        if ranking[index] == 1:
            selectedFeatures.append(index)
    return selectedFeatures;
def decisionTreeClassifier(X_new,y):
    X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.2)
    # 实例化决策树分类
    clf = DecisionTreeClassifier(max_depth=18,criterion="entropy",random_state=30,splitter='random')
    # 训练模型
    clf = clf.fit(X_train,y_train)
    muti_score(clf,X_new,y)

# get pearson,RFE,TMFS selected features and use decision tree to get OptimalFeatureSubset
def getOptimalFeatureSubset(X_minMax,y):
    PearsonSelectedFeatures = calcAttribute(X_minMax, y)
    RFESelectedFeatures = RFESelection(X_minMax, y)
    TMFSSelectedFeatures = tree_based_feature_selection(X_minMax, y)

    X_Pearson = sddp.feature_selected_data(PearsonSelectedFeatures, X_minMax)

    pearsonAccuracy = decisionTreeClassifier(X_Pearson,y)

    X_RFE = sddp.feature_selected_data(RFESelectedFeatures, X_minMax)
    RFEAccuracy = decisionTreeClassifier(X_RFE,y)

    X_TMFS = sddp.feature_selected_data(TMFSSelectedFeatures, X_minMax)
    TMFSAccuracy = decisionTreeClassifier(X_TMFS,y)
    # get OptimalFeatureSubset
    if pearsonAccuracy > RFEAccuracy and pearsonAccuracy > TMFSAccuracy:
        return PearsonSelectedFeatures
    elif RFEAccuracy > pearsonAccuracy and RFEAccuracy > TMFSAccuracy:
        return RFESelectedFeatures
    else:
        return TMFSSelectedFeatures
# if __name__ == '__main__':
#     path = '../data/service_discovery_dataset_generation_0501_0704.csv'
#     X_minMax,y = sddp.data_procession(path)
#     X_minMax = pd.DataFrame(X_minMax)
#     RFESelection(X_minMax,y)
