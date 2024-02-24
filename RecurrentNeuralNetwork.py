#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：项目 
@File    ：RecurrentNeuralNetwork.py
@Description：
@Author  ：heisenberg
@Date    ：2024/2/23 21:19 
'''
from sklearn.metrics import accuracy_score,precision_score,roc_auc_score, f1_score, recall_score
import numpy as np         #大量数据函数
import pandas as pd        #读取csv，iloc
import final.ServiceDiscoveryDataPreprocessing as sddp
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.models import Sequential
from tensorflow.python.keras.layers import Dense,Dropout, Activation,SimpleRNN,Embedding
from tensorflow.keras.optimizers import Adam
from time import time
def createMode(X_new):
    model = Sequential()
    model.add(SimpleRNN(units=24, return_sequences=True, input_shape=(input, 1)))
    model.add(Dropout(0.25))
    model.add(SimpleRNN(24, return_sequences=True))
    model.add(Dropout(0.25))
    model.add(SimpleRNN(16))
    # 输出层
    model.add(Dense(units=1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])
    return model

def RNNClassifier(X_new,y):
    # 定义五折交叉验证
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    # 执行交叉验证
    cv_scores = {}
    sum_accuracy = 0
    sum_precision = 0
    sum_recall = 0
    sum_auc = 0
    sum_f1score = 0
    startTime = time()
    model = createMode(X_new, y)
    for train_index, test_index in kfold.split(X_new,np.argmax(y, axis=1)):
        X_train, X_test = X_new[train_index], X_new[test_index]
        y_train, y_test = y[train_index], y[test_index]
        y[y == -1] = 0
        y = y.values.ravel()
        model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
        # 在测试集上评估模型
        y_pred = model.predict(X_test)
        y_pred = np.argmax(y_pred, axis=1)
        y_true = np.argmax(y_test, axis=1)
        accuracy = accuracy_score(y_true, y_pred)
        sum_accuracy += accuracy
        precision = precision_score(y_true, y_pred)
        sum_precision += precision
        # 计算AUC
        auc = roc_auc_score(y_true, y_pred)
        sum_auc += auc
        # 计算F1分数
        f1 = f1_score(y_true, y_pred)
        sum_f1score += f1
        # 计算召回率
        recall = recall_score(y_true, y_pred)
        sum_recall += recall
    endTime = time()
    consumeTime = endTime-startTime
    #  5次结果取平均值
    cv_scores[0] = sum(sum_accuracy) / 5
    cv_scores[1] = sum(sum_precision) / 5
    cv_scores[2] = sum(sum_recall) / 5
    cv_scores[3] = sum(sum_f1score) / 5
    cv_scores[4] = sum(sum_auc) / 5
    cv_scores[5] = consumeTime
    cv_scores[6] = model
    return cv_scores

# if __name__ == '__main__':
#     path = '../data/service_discovery_dataset_generation_0501_0704.csv'
#     X_minMax, y = sddp.data_procession(path)
#     filter_feature = {22, 23, 51, 37, 9, 4, 50, 47, 3, 49, 8, 48, 24, 33, 46, 38, 27, 45}
#     #X_minMax = pd.DataFrame(X_minMax)
#     X_new = sddp.feature_selected_data(filter_feature, X_minMax)
#     #filter_feature = [22, 23, 51, 37, 9, 4, 50, 47, 3, 49, 8, 48, 24, 33, 46, 38, 27, 45]
#     X_minMax = X_minMax.reshape(len(X_minMax), X_minMax.shape[1], 1, 1)
#     #X_minMax = pd.DataFrame(X_minMax)
#     print(X_minMax)
#     #X_new = sddp.feature_selected_data(filter_feature, X_minMax)
#     #CNNClassifier(filter_feature,y)