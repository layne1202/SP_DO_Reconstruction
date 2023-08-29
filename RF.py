# -*- coding: utf-8 -*-
import csv
import pandas as pd
import numpy as np
import pickle
from sklearn import metrics  # 模型结果指标库
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.ensemble import RandomForestRegressor
import gc
import os

def Adjust_param(X_train,y_train):# 调参

    cv_params  = {
        'max_depth': [6,8,10,12,14,16],

    }

    other_params = {
        'n_estimators': 500,
    }

    cat_model_ = RandomForestRegressor(**other_params)
    cat_search = GridSearchCV(cat_model_,
                              param_grid=cv_params ,
                              scoring='neg_mean_absolute_error',
                              cv=5)

    cat_search.fit(X_train, y_train)
    print(cat_search.best_params_)
    print(cat_search.best_score_)

    return cat_search.best_params_, -cat_search.best_score_


def splitzone(trainfile, CatBoostsave, depth, month,remarknote,Predict_features, num_features):
    print("---开始分区建模---")
    dataset = pd.read_csv(trainfile, encoding="UTF-8")

    classinformation = dataset["zone"].unique()
    #计算总的均方误差MSE，均方根误差RMSE，平均绝对误差MAE，平均绝对百分比误差MAPE，决定系数R2
    total_MSE= np.array([])
    total_MAE= []
    total_MAPE= []
    total_R2 = []
    total_test= []
    total_pre= []

    # 根据数值大小对classinformation列表进行排序
    numeric_list = [int(item) for item in classinformation]
    sorted_list = [item for item in sorted(numeric_list)]
    #根据ZONE中的数值划分数值
    for temp_classinformation in sorted_list:
        temp_data = dataset[dataset["zone"].isin([temp_classinformation])].copy()
        #利用三倍标准差剔除分区内异常值
        m1 = temp_data['DOXY'].mean()
        v1 = temp_data['DOXY'].std()
        t11 = m1 - 2*v1
        t21 = m1 + 2*v1
        temp_data =temp_data[(temp_data['DOXY'] > t11) & (temp_data['DOXY'] < t21)]

        try:
            temp_classinformation = int(temp_classinformation)
        except:
            temp_classinformation = temp_classinformation
            print(f'训练时int({temp_classinformation})出现错误！')#CatBoostMode
        total_MSE, total_MAE, total_MAPE, total_R2,total_test,total_pre = RandomForestModel(temp_data, CatBoostsave, temp_classinformation, Predict_features,num_features, total_MSE, total_MAE, total_MAPE, total_R2,total_test,total_pre)
        del temp_data
        gc.collect()

    #保存各分区测试集数据
    data = {'test': total_test, 'predict': total_pre}
    df = pd.DataFrame(data)
    if os.path.exists(CatBoostsave + 'TestSet.csv'):
        os.remove(CatBoostsave + 'TestSet.csv')
    try:
        existing_df = pd.read_csv(CatBoostsave + 'TestSet.csv')
        combined_df = pd.concat([existing_df, df], axis=1)
    except FileNotFoundError:
        combined_df = df
    combined_df.to_csv(CatBoostsave + 'TestSet.csv', index=False)
    total_MSE_weighted = metrics.mean_squared_error(total_test, total_pre)  # 计算总MSE
    total_RMSE_weighted = total_MSE_weighted**0.5  # 计算总RMSE
    total_MAE_weighted = metrics.mean_absolute_error(total_test, total_pre)  # 计算总MAE
    total_MAPE_weighted = mape0(total_test, total_pre)  # 计算总MAPE
    total_R2_weighted = metrics.r2_score(total_test, total_pre)  # 计算总R2

    txtfile = CatBoostsave + ".txt"
    with open(txtfile, 'a') as f:  # 设置文件对象
        print("---------------------------", file = f)
        print("总体模型精度:", file = f)
        print("总均方误差MSE: %.2f" % total_MSE_weighted, file = f)
        print("总均方根误差RMSE: %.2f" % total_RMSE_weighted, file = f)
        print("总平均绝对误差MAE: %.2f" % total_MAE_weighted, file = f)
        print("总平均绝对百分比误差MAPE: %.2f" % total_MAPE_weighted, file = f)
        print("总决定系数R2: %.2f" % total_R2_weighted, file = f)

    row = [remarknote, depth, month, total_MSE_weighted, total_RMSE_weighted, total_MAE_weighted, total_MAPE_weighted, total_R2_weighted]
    for i in range(len(total_MSE)):  # 对于每个分区
        row.extend([total_MSE[i], total_MSE[i] ** 0.5, total_MAE[i], total_MAPE[i], total_R2[i]])
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


    del dataset
def RandomForestModel(dataset, RandomForestSave, temp_classinformation,Predict_features,num_features, total_MSE, total_MAE, total_MAPE, total_R2,total_test,total_pre):# 随机森林回归建模
    # 数据准备
    datfile = RandomForestSave
    X = dataset.loc[:, Predict_features].values
    y = dataset.loc[:, ['DOXY']].values.ravel()

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)

    # 调用Adjust_param函数获取最优参数
    best_params, best_score = Adjust_param(X_train, y_train)

    # 定义随机森林回归模型
    rf = RandomForestRegressor(n_estimators=500, **best_params, random_state=100, n_jobs=-1)

    # 训练模型
    rf.fit(X_train, y_train)

    # 测试集预测
    y_pre = rf.predict(X_test)

    # 保存模型
    RandomForestSave = datfile + str(temp_classinformation) + ".dat"
    pickle.dump(rf, open(RandomForestSave, "wb"))



    # 测试精度评价
    mse = metrics.mean_squared_error(y_test, y_pre)
    rmse = metrics.mean_squared_error(y_test, y_pre)**0.5
    mae = metrics.mean_absolute_error(y_test, y_pre)
    mape = mape0(y_test, y_pre)
    R2 = metrics.r2_score(y_test, y_pre)
    Adjusted_R2 = 1-((1-R2)*(len(y_test)-1))/(len(y_test)-num_features-1)#校正决定系数

    #记录
    total_MSE = np.append(total_MSE, mse)
    total_MAE = np.append(total_MAE, mae)
    total_MAPE = np.append(total_MAPE, mape)
    total_R2 = np.append(total_R2, R2)
    total_R2_test = np.append(total_test, y_test)
    total_R2_pre = np.append(total_pre, y_pre)


    # 评价结果保存txt
    txtfile = datfile + ".txt"
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

    # 模型训练完毕
    print("区域{0}模型训练并测试完毕！".format(temp_classinformation))
    del X_train
    del X_test
    del y_train
    del y_test
    return total_MSE, total_MAE, total_MAPE, total_R2,total_R2_test,total_R2_pre
    gc.collect()

def mape0(y_test, y_pre):
    mask = y_test > 1
    return np.mean(np.abs((y_pre[mask] - y_test[mask]) / y_test[mask])) * 100

def relative_error0(y_test, y_pre):
    mask = y_test > 1
    ree =  np.abs(y_test[mask] - y_pre[mask]) / y_test[mask] * 100
    return ree[ree < 100]