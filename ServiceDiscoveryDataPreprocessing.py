#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# coding: utf-8
'''
@Project ：dianli
@File    ：service_discovery_data_preprocessing.py
@Description：服务发现算法，数据预处理
输入：原始数据即科来回溯系统中获取的数据，
1、输出：新增一列，1：需要备案，0：不知是否需要备案，-1：不需要备案
2、并将tcp/udp字段更改为数字,tcp：700 udp:600
@Author  ：heisenberg
@Date    ：2022/8/19 18:30
   字段含义
    端点
    地理位置
    TCP / UDP
    总字节数
    发送字节数
    接收字节数
    进字节数
    出字节数
    总数据包
    发送数据包数
    接收数据包数
    进数据包数
    出数据包数
    每秒数据包数
    每秒字节数
    每秒发送字节数
    每秒接收字节数
    比特率
    发送比特率
    接收比特率
    比特率（有效载荷）
    发送比特率（有效载荷）
    接收比特率（有效载荷）
    字节发收比
    数据包发收比
    平均包长
    发送平均包长
    接收平均包长
    数据包峰值
    发送数据包峰值
    接收数据包峰值
    流量峰值
    发送流量峰值
    接收流量峰值
    比特率峰值
    发送比特率峰值
    接收比特率峰值
    广播数据包数
    组播数据包数
    发送TCP同步包
    接收TCP同步包
    发送TCP同步确认包
    接收TCP同步确认包
    发送TCP重置包
    接收TCP重置包
    创建会话数
    关闭会话数
    活动会话数
    连接建立重置次数
    连接建立无响应次数
    ICMP数据包数
    发送ICMP数据包数
    接收ICMP数据包数
    ICMP丢包数
    ICMP丢包率
    ICMP最小回应时间
    ICMP最大回应时间
    ICMP平均回应时间
其中作为y的字段为：端点、地理位置、TCP / UDP 剩余字段作为x的字段
'''
import seaborn as sns #图表模块
import matplotlib.pyplot as plt #绘图模块库
import pandas as pd
import numpy as np
import codecs

from boruta import BorutaPy
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn import preprocessing
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

# 数据预处理
# 1.对离散特征的取值之间没有大小的意义，进行one-hot编码(主要对地点、TCP/UDP两个特征进行one-hot编码)
# 2.对离散特征的取值之间有大小的意义，进行归一化（对除地点、TCP/UDP、ip地址三个特征之外的进行归一化
from sklearn.utils import column_or_1d
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics # 分类结果评价
from sklearn.model_selection import cross_val_score
import warnings


def  data_procession(path):
   data = pd.read_csv(path,encoding='gbk',header=None)
   min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
   X = data.loc[:,3:57]
   y = data.loc[:,58:]
   # 对第3列-第57列特征进行归一化操作
   X_minMax = min_max_scaler.fit_transform(X)
   return X_minMax,y
# 皮尔森系数相关性画图
def pearson_drawing(X_minMax):
   result = X_minMax.astype(float).corr(name='pearson')
   print(result)
# 树模型特征选择：
def boruta_feature_selection(X,y):
    X = X.values
    y = y.values.ravel()
    y = column_or_1d(y, warn=True)
    # 按y标签比例取样
    rf = RandomForestClassifier(n_jobs=-1,class_weight='balanced',max_depth=5)
    # 定义boruta特征选择方式
    feat_selector = BorutaPy(rf,n_estimators='auto',verbose=2,random_state=1)
    # 找到所有相关特征
    feat_selector.fit(X,y)
    print(feat_selector.support_)
    print(feat_selector.ranking_)
    # 在X上调用transform（）以将其筛选为选定的功能
    X_filtered = feat_selector.transform(X)
    print(X_filtered)
    return X_filtered
# 基于树的特征选择
def tree_based_feature_selection(X,y):
    # forest = ExtraTreesClassifier(n_estimators=250,
    #                               random_state=0)

    # forest = DecisionTreeClassifier(random_state=0)
    # y = column_or_1d(y, warn=True)
    # forest.fit(X, y)
    # importances = forest.feature_importances_
    # std = np.std([tree.feature_importances_ for tree in forest.estimators_],
    #              axis=0)
    # indices = np.argsort(importances)[::-1]
    #
    # # Print the feature ranking
    # print("Feature ranking:")
    # ranks = []
    # for f in range(X.shape[1]):
    #     print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
    #     # 向字典中添加数据 即特征和特征对应的重要性程度
    #     ranks.append(indices[f])
    #     #ranks[indices[f]] = importances[indices[f]]
    # # Plot the feature importances of the forest
    # plt.figure(figsize=(30, 30))
    # plt.title("Feature importances")
    # plt.bar(range(X.shape[1]), importances[indices],
    #         color="r", yerr=std[indices], align="center")
    # plt.xticks(range(X.shape[1]), indices)
    # plt.xlim([-1, X.shape[1]])
    # plt.show()
    # 计算后，特征重要性大于0.3的有13个特征，将这13个特征提取出来生成新的数据集：
    # 1. feature 51 (0.081612)
    # 2. feature 24 (0.074381)
    # 3. feature 22 (0.071327)
    # 4. feature 48 (0.058764)
    # 5. feature 33 (0.052001)
    # 6. feature 30 (0.049039)
    # 7. feature 28 (0.046813)
    # 8. feature 50 (0.044490)
    # 9. feature 31 (0.042266)
    # 10. feature 25 (0.042040)
    # 11. feature 27 (0.038713)
    # 12. feature 49 (0.038075)
    # 13. feature 23 (0.034335)
    # 已选择的特征
    # feature_selected = ranks[:18]
    # print("feature_selected:\n", feature_selected)
    y[y == -1] = 0
    model = DecisionTreeClassifier()
    model.fit(X, y)
    importances = pd.DataFrame(data={
        'Attribute': X.columns,
        'Importance': model.feature_importances_
    })
    importances = importances.sort_values(by='Importance', ascending=False)
    # 可视化
    plt.bar(x=importances['Attribute'], height=importances['Importance'], color='#087E8B')
    plt.title('Feature importances obtained from coefficients', size=20)
    plt.xticks(rotation='vertical')
    plt.show()
    # feature_selected = ranks[:18]
    # print("feature_selected:\n", feature_selected)
    # return feature_selected
# 根据已选择的特征列名合并出新的数据集
def feature_selected_data(feature_selected,X_minMax):
    # 根据指定特征列名 生成新的数据集
    X_new = X_minMax.loc[:,feature_selected]
    return X_new

def random_forest_classification(X_new,y):
    # 先将数据集的80%作为训练集，20%作为测试集
    X_train,X_test,y_train,y_test = train_test_split(X_new,y,random_state=0,train_size=0.8)
    print('训练集和测试集 shape',X_train.shape,X_test.shape,y_train.shape,y_test.shape)
    model = RandomForestClassifier()  # 实例化模型RandomForestClassifier
    model.fit(X_train, y_train)  # 在训练集上训练模型
    print(model)  # 输出模型RandomForestClassifier

    # 在测试集上测试模型
    expected = y_test  # 测试样本的期望输出
    predicted = model.predict(X_test)  # 测试样本预测
    print(type(list(expected)),type(list(predicted)))
    # 输出结果
    print(metrics.classification_report(expected, predicted))  # 输出结果，精确度、召回率、f-1分数
    print(metrics.confusion_matrix(expected, predicted))  # 混淆矩阵

    auc = metrics.roc_auc_score(y_test, predicted)
    accuracy = metrics.accuracy_score(y_test, predicted)  # 求精度
    #-----结果-----#
    # 设置average='micro'时 所有指标的值都一样
    # Accuracy:  0.9973110247983269
    # Precision:  0.9973110247983269
    # --recall: 0.9973110247983269
    # --f1: 0.9973110247983269
    # 不设置average时，指标不一样
    # Accuracy:  0.9979085748431431
    # Precision:  0.9988998899889989
    # --recall: 0.9972542559033498
    # --f1: 0.998076394613905

    # 可以直接用来衡量模型的好坏,其结果指的是对整体 样本的预测准确度,accuracy 的值越大,说明模型越好．P=(TP+TN)/(TP+TN+FP+FN)
    print("Accuracy(值越大，模型越好): " , (accuracy))
    precision = metrics.precision_score(y_test,predicted)
    # 精确率是表示预测为正样本中，被实际为正样本的比例 P = TP / (TP + FP)
    print("Precision: " , (precision))
    recall = metrics.recall_score(expected, predicted)
    # 召回率表示实际为正样本中，预测为正样本的比例 P = TP / (TP + FN)
    print('Recall:', recall)
    # F1分数 2*precision*recall/(precision+recall)
    f1 = metrics.f1_score(np.array(expected), np.array(predicted))
    print('F1:', f1)
from sklearn.feature_selection import RFECV
# 递归特征消除法进行特征选择
# 筛选后的特征{3,4,5,18,20,21,22,23,24,25,26,29,31,37,38,39,40,41,43,45,46,47,48,49,52}共25个
def RFESelection(X,y):
    estimator = DecisionTreeClassifier(random_state=111)
    print("start RFE...")
    # 5折交叉
    selector = RFECV(estimator, step=1, cv=5)
    selector = selector.fit(X, y)
    print("end RFE...")
    # 哪些特征入选最后特征，true表示入选
    print(selector.support_)
    # 每个特征的得分排名，特征得分越低（1最好），表示特征越好
    print(selector.ranking_)
    #  挑选了几个特征
    print(selector.n_features_)
    # 每次交叉迭代各个特征得分
    print(selector.grid_scores_)

from feature_selection_ga import FeatureSelectionGA, FitnessFunction
# 使用遗传算法特征选择
def featureSelectionGA(X_new,y):
    df = pd.DataFrame(y)
    y = df.values
    #y = column_or_1d(y,warn=True)
    """1.1. 第一种模型验证方法"""
    # 切分数据集
    X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.2)
    model = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5)
    fsga = FeatureSelectionGA(model, X_train, y_train, ff_obj=FitnessFunction())
    print("fsga",fsga)
    pop = fsga.generate(10)
    print("pop:",pop)
# 使用SVM进行分类 
def SVM_classification(X_new,y):
    y = column_or_1d(y,warn=True)
    """1.1. 第一种模型验证方法"""
    # 切分数据集
    X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.2)
    print('使用SVM进行分类 训练集和测试集 shape', X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    """2. 通过网格搜索寻找最优参数"""
    """
    未通过网格搜索寻找最优参数
    SVM 准确率: 0.893286219081272
    SVM Precision: 0.9174311926605505
    SVM Recall: 0.8883248730964467
    SVM F1: 0.9026434558349451
    """
    parameters = {
        'gamma': np.linspace(0.0001, 0.1),
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    }
    model = svm.SVC()
    grid_model = GridSearchCV(model, parameters, cv=10, return_train_score=True)
    grid_model.fit(X_train, y_train)
    # 用测试集做预测
    predicted = grid_model.predict(X_test)
    accuracy = metrics.accuracy_score(y_test,predicted)
    # 输出模型的最优参数
    print(grid_model.best_params_)
    print('SVM Accuracy: ', accuracy)
    precision = metrics.precision_score(y_test,predicted)
    # 精确率是表示预测为正样本中，被实际为正样本的比例 P = TP / (TP + FP)
    print("SVM Precision: " , precision)
    recall = metrics.recall_score(y_test, predicted)
    # 召回率表示实际为正样本中，预测为正样本的比例 P = TP / (TP + FN)
    print('SVM Recall:', recall)
    # F1分数 2*precision*recall/(precision+recall)
    f1 = metrics.f1_score(np.array(y_test), np.array(predicted))
    print('SVM F1:', f1)

# 使用决策树做二分类
def decisionTreeClassifier(X_new,y):
    X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.2)
    # 实例化决策树分类
    clf = DecisionTreeClassifier(max_depth=18,criterion="entropy",random_state=30,splitter='random')
    # 训练模型
    clf = clf.fit(X_train,y_train)
    predicted = clf.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, predicted)
    print('DecisionTree Accuracy: ', accuracy)
    muti_score(clf,X_new,y)


# 构建评分函数，并采取５折交叉验证的方式评分 https://blog.csdn.net/weixin_41710583/article/details/85016622
# 交叉验证是一个将整体数据集平均划分为k份，先取第一份子集数据作为测试集，剩下的k-1份子集数据作为训练集进行一次实验；
# 然后再取第二份子集数据，剩下的k-1份子集数据进行一次实验，不断往复，最后重复k次的过程 我们称之为k折交叉检验，交叉检验是我们进行参数调整过程中非常重要的一个方法。
# 所以cross_val_score中的X,y为整体数据集的X和y
def muti_score(model,X_new,y):
    warnings.filterwarnings('ignore')
    accuracy = cross_val_score(model, X_new,y, scoring='accuracy', cv=5)
    precision = cross_val_score(model, X_new,y, scoring='precision', cv=5)
    recall = cross_val_score(model, X_new,y, scoring='recall', cv=5)
    f1_score = cross_val_score(model, X_new,y, scoring='f1', cv=5)
    auc = cross_val_score(model, X_new,y, scoring='roc_auc', cv=5)
    print("准确率:",accuracy.mean())
    print("精确率:",precision.mean())
    print("召回率:",recall.mean())
    print("F1_score:",f1_score.mean())
    print("AUC:",auc.mean())




if __name__ == '__main__':
    path = '../data/service_discovery_dataset_generation_0501_0704.csv'
    X_minMax,y = data_procession(path)
    X_minMax = pd.DataFrame(X_minMax)
    # print("X_minMax",X_minMax)
    # print("y",y)

    #boruta_feature_selection(X_minMax,y)

    #featureSelectionGA(X_minMax,y)
    # print(X_minMax.shape)
    # 画对应的皮尔森相关系数
    pearson_drawing(X_minMax)

