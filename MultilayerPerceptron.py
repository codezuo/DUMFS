import csv
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay, confusion_matrix, roc_curve
import numpy as np
import pandas as pd
from sklearn import metrics # 分类结果评价
import matplotlib.pyplot as plt
from time import time


def load_dataset(path):
    dataset_file = csv.reader(open(path))
    vector_x = []  # 样本
    y = []  # 样本所对应的标签
    # 从文件读取训练数据集
    for content in dataset_file:
        # 如果读取的不是表头
        if dataset_file.line_num != 1:
            # 读取一行并转化为列表
            content = list(map(float, content))
            if len(content) != 0:
                vector_x.append(content[1:12])  # 第1-12列是样本的特征
                y.append(content[-1])  # 最后一列是样本的标签
    return vector_x, y  # 返回训练数据集


# 训练模型
def mlp_cls(X_new, y):
    # 输入层->第一层->第二层->输出层
    #    12      30     20      1  # 节点个数
    # MLPClassifier参数说明详情见https://www.cnblogs.com/-X-peng/p/14225973.html
    mlp = MLPClassifier(solver='adam', alpha=0, hidden_layer_sizes=(30, 20), random_state=1,max_iter=1000)
    mlp.fit(X_new, y)  # 训练
    return mlp_cls_predict(mlp, X_new, y)


from sklearn.model_selection import cross_val_score, KFold
import final.ServiceDiscoveryDataPreprocessing as sddp
# 模型预测
def mlp_cls_predict(model, X_new, y):
    mlp_start_time = time()
    y[y == -1] = 0
    y = y.values.ravel()
    # 定义深度神经网络模型
    # 定义五折交叉验证
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    # 执行交叉验证
    accuracy = cross_val_score(model, X_new, y, scoring='accuracy', cv=kfold)
    precision = cross_val_score(model, X_new, y, scoring='precision', cv=kfold)
    recall = cross_val_score(model, X_new, y, scoring='recall', cv=kfold)
    f1_score = cross_val_score(model, X_new, y, scoring='f1', cv=kfold)
    auc = cross_val_score(model, X_new, y, scoring='roc_auc', cv=kfold)
    mlp_end_time = time()
    consumeTime = mlp_end_time - mlp_start_time
    result = {}
    result[0] = accuracy.mean()
    result[1] = precision.mean()
    result[2] = recall.mean()
    result[3] = f1_score.mean()
    result[4] = auc.mean()
    result[5] = consumeTime
    result[6] = model
    return result

# if __name__ == '__main__':
#     path = '../data/service_discovery_dataset_generation_0501_0704.csv'
#     X_minMax, y = sddp.data_procession(path)
#     X_minMax = pd.DataFrame(X_minMax)
#     filter_feature = {22, 23, 51, 37, 9, 4, 50, 47, 3, 49, 8, 48, 24, 33, 46, 38, 27, 45}
#     X_new = sddp.feature_selected_data(filter_feature, X_minMax)
#     mlp = mlp_cls(X_new, y)
