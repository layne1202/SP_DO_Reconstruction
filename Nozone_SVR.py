# -*- coding: utf-8 -*-
import csv
import pandas as pd
import numpy as np
import pickle
from sklearn import metrics  # 模型结果指标库
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
import gc
import os


def Adjust_param(X_train,y_train):
    # 设置参数范围
    cv_params = {
        'C': [0.1, 1, 10, 100],
        'gamma': [0.001, 0.01, 0.1, 1],
    }

    # 使用默认参数定义SVR模型
    cat_model_ = SVR(kernel='rbf')

    # 使用GridSearchCV进行参数调优
    cat_search = GridSearchCV(cat_model_,
                              param_grid=cv_params ,
                              scoring='neg_mean_absolute_error',
                              cv=5)

    cat_search.fit(X_train, y_train)

    # 输出最优参数和最优得分
    print("最优参数：", cat_search.best_params_)
    print("最优得分：", -cat_search.best_score_)

    # 返回最优参数和最优得分
    return cat_search.best_params_, -cat_search.best_score_

def splitzone(trainfile, CatBoostsave, depth, month,remarknote,Predict_features, num_features):
    print("---开始分区建模---")
    dataset = pd.read_csv(trainfile, encoding="UTF-8")

    SVRModel(dataset, CatBoostsave, depth, month,remarknote, 999, Predict_features,num_features)

    del dataset

def save_scaler(scaler, save_path):
    with open(save_path, 'wb') as f:
        pickle.dump(scaler, f)

def SVRModel(dataset, SVRSave, depth, month,remarknote, temp_classinformation,Predict_features,num_features):
    # 数据准备
    datfile = SVRSave
    X = dataset.loc[:, Predict_features].values
    y = dataset.loc[:, ['DOXY']].values.ravel()

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)

    # 数据标准化
    scaler_X = StandardScaler()
    X_train = scaler_X.fit_transform(X_train)
    X_test = scaler_X.transform(X_test)
    save_scaler(scaler_X, datfile + str(temp_classinformation) + "X.pkl")

    # 调用Adjust_param函数获取最优参数
    best_params, best_score = Adjust_param(X_train, y_train)

    svr = SVR(kernel='rbf', **best_params)
    print("网格搜索法最佳得分为：{}".format(best_score))

    # 训练模型
    svr.fit(X_train, y_train)

    # 测试集预测
    y_pre = svr.predict(X_test)


    # 保存模型
    SVRSave = datfile + str(temp_classinformation) + ".dat"
    pickle.dump(svr, open(SVRSave, "wb"))


    # 测试精度评价
    mse = metrics.mean_squared_error(y_test, y_pre)
    rmse = metrics.mean_squared_error(y_test, y_pre)**0.5
    mae = metrics.mean_absolute_error(y_test, y_pre)
    mape = mape0(y_test, y_pre)
    R2 = metrics.r2_score(y_test, y_pre)
    Adjusted_R2 = 1-((1-R2)*(len(y_test)-1))/(len(y_test)-num_features-1)#校正决定系数


    # 评价结果保存txt
    txtfile = datfile + "999.txt"
    with open(txtfile, 'a') as f:  # 设置文件对象
        print("---------------------------", file = f)
        print("区域{0}模型训练并保存完毕！".format(temp_classinformation), file = f)
        print("网格搜索法最优参数：{0}".format(best_params),file = f)
        print("网格搜索法最优得分： %.2f" % best_score,file = f)
        print("均方误差MSE: %.2f" % mse, file = f)
        print("均方根误差RMSE: %.2f" % rmse, file = f)
        print("平均绝对误差MAE: %.2f" % mae, file = f)
        print("平均绝对百分比误差MAPE: %.2f" % mape, file = f)
        print("决定系数R2: %.2f" % R2, file = f)
        print("校正决定系数Adjusted_R2: %.2f" % Adjusted_R2, file = f)

    row = [remarknote, depth, month, mse, rmse, mae, mape, R2]
    accfile = "D:\\Data\\Argo_Data\\dissolved_oxygen\\result\\accuracy.csv"
    accfilebeifen = "D:\\Data\\Argo_Data\\dissolved_oxygen\\result\\accuracy1.csv"
    try:
        with open(accfile, 'a', newline='', encoding='UTF-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(row)
            print("模型精度写入csv完成！")
    except:
        print("模型精度写入csv失败，即将写入备份文件中！")
        try:
            with open(accfilebeifen, 'a', newline='', encoding='UTF-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(row)
                print("模型精度写入csv完成！")
        except:
            print("无法写入备份文件!")

    # 模型训练完毕
    print("区域{0}模型训练并测试完毕！".format(temp_classinformation))
    del X_train
    del X_test
    del y_train
    del y_test
    gc.collect()

def mape0(y_test, y_pre):
    mask = y_test > 1
    return np.mean(np.abs((y_pre[mask] - y_test[mask]) / y_test[mask])) * 100

def relative_error0(y_test, y_pre):
    mask = y_test > 1
    ree =  np.abs(y_test[mask] - y_pre[mask]) / y_test[mask] * 100
    return ree[ree < 100]