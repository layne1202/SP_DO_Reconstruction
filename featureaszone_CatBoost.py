# -*- coding: utf-8 -*-
import csv
import pandas as pd
import numpy as np
import pickle
from sklearn import metrics  # 模型结果指标库
from sklearn.model_selection import train_test_split,GridSearchCV
from catboost import CatBoostRegressor
import gc
import matplotlib.pyplot as plt  # 画图


def Adjust_param(X_train,y_train,cat_features):# 调参

    cv_params  = {
        'depth': [12,14,16],
        #'learning_rate': [0.05,0.1,0.2],
        #'l2_leaf_reg': [0.1,1],  # L2正则化系数
        #'loss_function': ['MAE','RMSE']

    }

    other_params = {
        'learning_rate': 0.1,
        'l2_leaf_reg': 3,  # L2正则化系数
        'iterations': 500,
        'loss_function': 'RMSE',
        'eval_metric': 'RMSE',
        'thread_count': -1,#训练时所用的cpu/gpu核数
        'verbose': False,
        'random_seed': 100,
        'cat_features': cat_features
    }

    cat_model_ = CatBoostRegressor(**other_params)
    cat_search = GridSearchCV(cat_model_,
                              param_grid=cv_params ,
                              scoring='neg_mean_absolute_error',
                              cv=3)

    cat_search.fit(X_train, y_train, eval_set=(X_train, y_train), early_stopping_rounds=50, use_best_model=True)
    print(cat_search.best_params_)
    print(cat_search.best_score_)

    return cat_search.best_params_, -cat_search.best_score_

def splitzone(trainfile, CatBoostsave, depth, month,remarknote,Predict_features,num_features):
    print("---开始建模---")
    dataset = pd.read_csv(trainfile, encoding="UTF-8")

    CatBoostMode(dataset, CatBoostsave, depth, month,remarknote, 789, Predict_features,num_features)

    del dataset

def CatBoostMode(dataset, CatBoostsave, depth, month,remarknote, temp_classinformation,Predict_features,num_features):# Catboost建模
    # 数据准备
    datfile = CatBoostsave
    X = dataset.loc[:, Predict_features].values
    y = dataset.loc[:, ['DOXY']].values.ravel()
    #X = preprocessing.scale(X, axis=0)  #将数据转化为标准数据

    try:
        # 获取类别型特征相对于 Predict_features 的列索引
        cat_feature_index = Predict_features.index('EVENT')
        # 指定类别型特征的列索引
        cat_features = [cat_feature_index]
    except:
        cat_features = []


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)


    '''scaler = StandardScaler()
    X_train= scaler.fit_transform(X_train)
    X_test= scaler.transform(X_test)
    # 保存标准化参数
    scaler_filename = datfile + str(temp_classinformation) + ".save"
    pickle.dump(scaler, open(scaler_filename, "wb"))#pickle'''


    # 网格搜索法调参
    #best_params, best_score = Adjust_param(X_train, y_train, cat_features)


    params = {
        #'depth': best_params['depth'],
        'depth':12,
        #'learning_rate': best_params['learning_rate'],
        'learning_rate': 0.1,
        #'l2_leaf_reg': best_params['l2_leaf_reg'],
        'l2_leaf_reg': 3,   # L2正则化系数，用于控制模型复杂度，防止过拟合，但过大的正则化系数可能会导致欠拟合。
        #'iterations': best_params['iterations'],
        'iterations': 500,
        #'iterations': best_params['iterations'],
        'loss_function': 'RMSE', #损失函数，用于衡量模型的预测误差，MAE更适合偏态数据集
        #'loss_function': best_params['loss_function'],
        'eval_metric': 'MAE', #评价指标，用于评估模型在验证集上的性能
        'thread_count': -1,#训练时所用的cpu/gpu核数
        'verbose': False, #是否输出训练过程信息
        'random_seed': 100, #随机种子，用于控制随机化过程的起点，保证每次训练的结果是可重复的。
        'cat_features': cat_features
    }

    # 定义模型
    clf = CatBoostRegressor(**params)
    # 训练模型
    clf.fit(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=50, use_best_model=True, verbose = False)#early_stopping_rounds=50，即如果模型在验证集上的性能连续50次迭代都没有提升，就会提前停止训练，以避免过拟合。

    #计算回归模型的测试集决定系数
    score = clf.score(X_test, y_test)#用于计算模型在测试集上的性能表现，可以作为评估模型的重要指标之一，同时也可以用于比较不同模型的性能。
    #测试集预测
    y_pre = clf.predict(X_test)

    # 保存模型
    CatBoostsave = datfile + str(temp_classinformation) + ".dat"
    pickle.dump(clf, open(CatBoostsave, "wb"))#pickle

    '''# 绘制折线图
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.arange(len(y_pre)), y=y_test, mode='lines+markers', name='true value', line=dict(color='green')))
    fig.add_trace(go.Scatter(x=np.arange(len(y_pre)), y=y_pre, mode='lines+markers', name='predict value', line=dict(color='red')))
    fig.update_layout(title='%s score: %f' % ("catboost", score), xaxis_title='Serial number', yaxis_title='Oxygen Value(μmol/kg)')
    fig.show()

    # 绘制散点图
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[min(y_test)-10, max(y_test)+10], y=[min(y_test)-10, max(y_test)+10], mode='lines', name='1:1 reference', line=dict(color='black', width=1, dash='dot')))
    fig.add_trace(go.Scatter(x=y_test, y=y_pre, mode='markers', name='data points', marker=dict(color='red', size=8, line=dict(color='black', width=0.5))))
    fig.update_layout(title='Catboost Scatter Plot', xaxis_title='TrueValue(μmol/kg)', yaxis_title='PredictValue(μmol/kg)', xaxis_range=[min(y_test)-10, max(y_test)+10], yaxis_range=[min(y_test)-10, max(y_test)+10])
    fig.show()

    # 绘制相对误差图
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(1, len(relative_error)+1)), y=sorted(relative_error), mode='markers', name='relative error', marker=dict(color='green', size=8, line=dict(color='black', width=0.5))))
    fig.add_trace(go.Scatter(x=[0, len(y_pre)+3], y=[10, 10], mode='lines', name='10% threshold', line=dict(color='black', width=1, dash='dot')))
    fig.add_trace(go.Scatter(x=[0, len(y_pre)+3], y=[2, 2], mode='lines', name='2% threshold', line=dict(color='black', width=1, dash='dot')))
    fig.add_trace(go.Scatter(x=[0, len(y_pre)+3], y=[5, 5], mode='lines', name='5% threshold', line=dict(color='black', width=1, dash='dot')))
    fig.update_layout(title='Relative error Scatter Plot', xaxis_title='Serial number', yaxis_title='Relative error(%)', xaxis_range=[0, len(y_pre)+3], yaxis_range=[0, max(sorted(relative_error))+5])
    fig.show()'''

    #树可视化
    '''tree.plot_tree(CatBoost.estimators_[0])
    plt.savefig('fix.jpg', dpi=1200)
    plt.show()'''


    # 作折线图
    plt.figure(figsize=(15, 10))
    plt.subplot(2,2,1)
    plt.plot(np.arange(len(y_pre)), y_test, 'go-', label='Observed Value')
    plt.plot(np.arange(len(y_pre)), y_pre, 'ro-', label='Predicted Value')
    plt.title('%s score: %f' % ("Catboost", score))
    plt.xlabel('ID')
    plt.ylabel('Dissolved Oxygen (μmol/kg)')
    plt.legend()

    #生成散点图
    xxx = [min(y_test)-10, max(y_test)+10]
    yyy = [min(y_test)-10, max(y_test)+10]#设置参考的1：1虚线参数
    plt.subplot(2,2,2)
    plt.plot(xxx, yyy, c='0', linewidth=1, linestyle=':', marker='.', alpha=0.3)#绘制虚线
    plt.scatter(y_test, y_pre, s=20, c='r', edgecolors='k', marker='o', alpha=0.8)#绘制散点图，横轴是真实值，竖轴是预测值
    plt.xlim((min(y_test)-10, max(y_test)+10))   #设置坐标轴范围
    plt.ylim((min(y_test)-10, max(y_test)+10))
    plt.title('Catboost Scatter Plot')
    plt.xlabel('Observed Values')
    plt.ylabel('Reconstructed Values')

    #相对误差图
    xxx1 = [0, len(y_pre)+3]
    yyy1 = [20, 20]
    yyy2 = [10, 10]
    yyy3 = [5, 5]
    plt.subplot(2,2,3)
    plt.plot(xxx1, yyy1, c='0', linewidth=1, linestyle=':', marker='.', alpha=0.3)#绘制虚线
    plt.plot(xxx1, yyy2, c='0', linewidth=1, linestyle=':', marker='.', alpha=0.3)#绘制虚线
    plt.plot(xxx1, yyy3, c='0', linewidth=1, linestyle=':', marker='.', alpha=0.3)#绘制虚线
    relative_error = relative_error0(y_test, y_pre)
    x = list(range(1, len(relative_error)+1))
    plt.plot(x, sorted(relative_error), 'go')
    plt.xlim((0, len(y_pre)+3))#设置坐标轴范围
    plt.title('Relative error Scatter Plot')
    plt.xlabel('ID')
    plt.ylabel('Relative error(%)')

    # 重要性评估图
    feature_importance = clf.feature_importances_ # make importances relative to max importance
    plt.subplot(2, 2, 4)
    feat_importances = pd.Series(clf.feature_importances_, index=pd.DataFrame(dataset.loc[:, Predict_features]).columns)
    feat_importances.nlargest(num_features).plot(kind='barh')
    plt.title('Top 4 Most Important Features')
    plt.xlabel('Relative Importance')
    plt.ylabel('Features')
    pathpicture = datfile + str(temp_classinformation) +".png"
    plt.savefig(pathpicture, dpi=600, bbox_inches='tight')

    # 测试精度评价
    mse = metrics.mean_squared_error(y_test, y_pre)
    rmse = metrics.mean_squared_error(y_test, y_pre)**0.5
    mae = metrics.mean_absolute_error(y_test, y_pre)
    mape = mape0(y_test, y_pre)
    R2 = metrics.r2_score(y_test, y_pre)
    Adjusted_R2 = 1-((1-R2)*(len(y_test)-1))/(len(y_test)-num_features-1)#校正决定系数


    # 评价结果保存txt
    txtfile = datfile + "789.txt"
    with open(txtfile, 'a') as f:  # 设置文件对象
        print("---------------------------", file = f)
        print("区域{0}模型训练并保存完毕！".format(temp_classinformation), file = f)
        #print("网格搜索法最优参数：{0}".format(best_params),file = f)
        #print("网格搜索法最优得分： %.2f" % best_score,file = f)
        print("均方误差MSE: %.2f" % mse, file = f)
        print("均方根误差RMSE: %.2f" % rmse, file = f)
        print("平均绝对误差MAE: %.2f" % mae, file = f)
        print("平均绝对百分比误差MAPE: %.2f" % mape, file = f)
        print("决定系数R2: %.2f" % R2, file = f)
        print("校正决定系数Adjusted_R2: %.2f" % Adjusted_R2, file = f)
        print("特征重要性：{0}".format(feature_importance), file = f)

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