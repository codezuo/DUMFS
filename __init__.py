#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：项目 
@File    ：__init__.py.py
@Description：
@Author  ：heisenberg
@Date    ：2024/2/21 16:02 
'''
import final.MultiCompetitiveFeatureSelection as mcfs
import final.DecisionTree
import final.eXtremeGradientBoosting
import final.GradientBoostingMachine
import final.LightGradientBoostingMachine
import final.LogisticRegression
import final.RandomForest
import final.SupportVectorMachine
import final.ServiceDiscoveryDataPreprocessing as sddp
import final.ConvolutionalNeuralNetwork
import final.RecurrentNeuralNetwork
import final.DeepNeuralNetworks
import final.MultilayerPerceptron
import pandas as pd
def getBestMode(X_new, y):
    decisionTreeResult = DecisionTree.decisionTreeClassifier(X_new,y)
    eXtremeGradientBoostingResult = eXtremeGradientBoosting.XgBoostClassifier(X_new,y)
    gradientBoostingMachineResult = GradientBoostingMachine.GBMClassifier(X_new,y)
    lightGradientBoostingMachineResult = LightGradientBoostingMachine.LGBMClassifier(X_new,y)
    logisticRegressionResult = LogisticRegression.logistic_regression(X_new,y)
    randomForestResult = RandomForest.random_forest_classification(X_new,y)
    supportVectorMachineResult = SupportVectorMachine.SVMClassification(X_new,y)
    CNNResult = ConvolutionalNeuralNetwork.CNNClassifier(X_new,y)
    RNNResult = RecurrentNeuralNetwork.RNNClassifier(X_new,y)
    DNNResult = DeepNeuralNetworks.DNNClassifier(X_new,y)
    MLPResult = MultilayerPerceptron.mlp_cls(X_new,y)

    DTAns = metricProcess(decisionTreeResult)
    XGBoostAns = metricProcess(eXtremeGradientBoostingResult)
    GBMAns = metricProcess(gradientBoostingMachineResult)
    LGBMAns = metricProcess(lightGradientBoostingMachineResult)
    LRAns = metricProcess(logisticRegressionResult)
    RFAns = metricProcess(randomForestResult)
    SVMAns = metricProcess(supportVectorMachineResult)
    CNNAns = metricProcess(CNNResult)
    RNNAns = metricProcess(RNNResult)
    DNNAns = metricProcess(DNNResult)
    MLPLAns = metricProcess(MLPResult)
    # 假设您有7个结果存储在列表results中
    ans = [DTAns, XGBoostAns, GBMAns, LGBMAns, LRAns, RFAns, SVMAns,CNNAns,RNNAns,DNNAns,MLPLAns]
    models = [decisionTreeResult[6], eXtremeGradientBoostingResult[6],gradientBoostingMachineResult[6],
              lightGradientBoostingMachineResult[6],logisticRegressionResult[6],randomForestResult[6],
              supportVectorMachineResult[6],CNNResult[6],RNNResult[6],DNNResult[6],
              MLPResult[6]]
    # 使用zip()函数将值和模型一一对应起来
    value_model_pairs = zip(ans, models)
    # best model and result
    max_value, max_model = max(value_model_pairs)
    return max_model

# 指标处理
def metricProcess(result):
    accuracy = result[0]
    precision = result[1]
    recall = result[2]
    f1_score = result[3]
    auc = result[4]
    consumeTime = result[5]
    # 权重
    parameter0 = 0.3
    parameter1 = 0.125
    parameter2 = 0.125
    parameter3 = 0.125
    parameter4 = 0.125
    parameter5 = 0.2
    ans = accuracy * parameter0 + precision * parameter1 + recall * parameter2 + f1_score * parameter3 + auc * parameter4 + parameter5 / consumeTime
    return ans

if __name__ == '__main__':
    path = '../data/service_discovery_dataset_generation_0501_0704.csv'
    X_minMax, y = sddp.data_procession(path)
    X_minMax = pd.DataFrame(X_minMax)
    # use multicompetitivefeatureselection to getOptimalFeatureSubset
    optimalFeatureSubset = mcfs.getOptimalFeatureSubset(X_minMax,y)
    X_new = sddp.feature_selected_data(optimalFeatureSubset, X_minMax)
    getBestMode(X_new, y)