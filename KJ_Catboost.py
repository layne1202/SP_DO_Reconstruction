# -*- coding: utf-8 -*-
import shap
import csv
import os
import pandas as pd
import numpy as np
import pickle
from sklearn import metrics  # 模型结果指标库
from sklearn.model_selection import train_test_split, GridSearchCV
from catboost import CatBoostRegressor
import gc
import matplotlib.pyplot as plt  # 画图
'''import sys
sys.path.append(r'D:\Document\catboost-master\catboost-master\catboost')'''


def adjust_param(x_train, y_train, cat_features):  # 调参
    print("开始调参！")
    cv_params = {
        'depth': [6, 8, 10, 12, 14],
        # 'learning_rate': [0.05,0.1,0.2],
        # 'l2_leaf_reg': [0.1,1],  # L2正则化系数
        # 'loss_function': ['MAE','RMSE']

    }

    other_params = {
        'learning_rate': 0.1,
        'l2_leaf_reg': 3,  # L2正则化系数
        'iterations': 800,
        'loss_function': 'RMSE',
        'eval_metric': 'RMSE',
        'thread_count': -1,  # 训练时所用的cpu/gpu核数
        'verbose': False,
        'random_seed': 100,
        'cat_features': cat_features,
    }

    cat_model_ = CatBoostRegressor(**other_params)
    cat_search = GridSearchCV(cat_model_,
                              param_grid=cv_params,
                              scoring='neg_mean_absolute_error',
                              cv=5)

    cat_search.fit(x_train, y_train, eval_set=(x_train, y_train), early_stopping_rounds=50, use_best_model=True)
    print(cat_search.best_params_)
    print(cat_search.best_score_)

    return cat_search.best_params_, -cat_search.best_score_

def splitzone(trainfile, CatBoostsave, depth, month, remarknote, Predict_features, num_features):
    print("---开始分区建模---")
    dataset = pd.read_csv(trainfile, encoding="UTF-8")
    # 使用 replace 函数替换特定值
    #dataset['EVENT'] = dataset['EVENT'].replace({'LaNina': 1, 'Normal': 2, 'ENSO': 3})

    classinformation = dataset["zone"].unique()
    #计算总的均方误差MSE，均方根误差RMSE，平均绝对误差MAE，平均绝对百分比误差MAPE，决定系数R2
    total_test= []
    total_pre= []
    total_MSE= []
    total_MAE= []
    total_MAPE= []
    total_R2 = []
    all_shap_values = []  # 在循环外部初始化空列表

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
        #exec("df%s = temp_data"%temp_classinformation)
        try:
            temp_classinformation = int(temp_classinformation)
        except:
            pass

        total_MSE, total_MAE, total_MAPE, total_R2, total_test, total_pre,all_shap_values = CatBoostMode(temp_data, CatBoostsave, temp_classinformation, Predict_features,num_features, total_MSE, total_MAE, total_MAPE, total_R2, total_test, total_pre,all_shap_values)
        del temp_data
        gc.collect()

    #保存各分区测试集数据
    df = pd.DataFrame({'test': total_test, 'predict': total_pre})

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


    '''# 绘制SHAP（为每个分区设置颜色）
    colors = plt.cm.turbo(np.linspace(0, 1, len(all_shap_values)))
    fig, ax = plt.subplots(figsize=(10, 10))
    # 循环绘制每个分区的Shap值
    for i, shap_values in enumerate(all_shap_values):
        shap.summary_plot(shap_values.values, None, feature_names=Predict_features, color=colors[i], plot_type='dot', show=False, axis_color="#333333")
'''

    #绘制SHAP
    # 合并 .values 属性
    concatenated_values = np.vstack([s.values for s in all_shap_values])
    # 合并 .base_values 属性
    concatenated_base_values = np.hstack([s.base_values for s in all_shap_values])
    # 合并 .data 属性
    concatenated_data = np.vstack([s.data for s in all_shap_values])
    # 创建一个新的 shap.Explanation 对象
    merged_shap_values = shap.Explanation(
        values=concatenated_values,
        base_values=concatenated_base_values,
        data=concatenated_data
    )
    # 设置图表大小
    fig, ax = plt.subplots(figsize=(10, 10))
    # 设置全局字体大小
    plt.rcParams.update({'font.size': 15})
    # 设置全局字体为新罗马
    plt.rcParams["font.family"] = "Times New Roman"
    # 绘制SHAP图
    shap.summary_plot(merged_shap_values, None, feature_names=Predict_features, plot_type='dot', show=False)
    # 设置绘图属性
    ax.set_title('')
    # 设置横坐标轴范围
    ax.set_xlim(-30, 30)
    #ax.set_xlabel('SHAP value(impact on model output)', fontsize=14)
    #ax.set_ylabel('Input variables in CatBoost model', fontsize=14)
    # 增加y轴刻度标签的字体大小
    ax.yaxis.set_tick_params(labelsize=12)
    # 显示并保存图像
    plt.tight_layout() # 确保所有元素在图像中可见
    #plt.show()
    fig.savefig(CatBoostsave + 'SHAP2.png', dpi=600)

    #写入文件
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
    del merged_shap_values
    del total_test
    del total_pre
    del total_MSE
    del total_MAE
    del total_MAPE
    del total_R2
    del all_shap_values


def CatBoostMode(dataset, CatBoostsave, temp_classinformation,Predict_features,num_features, total_MSE, total_MAE, total_MAPE, total_R2,total_test,total_pre,all_shap_values):# 随机森林回归建模
    # 数据准备
    datfile = CatBoostsave
    #feature_names = dataset.loc[:, Predict_features].columns.tolist()
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
    best_params, best_score = adjust_param(X_train, y_train, cat_features)


    params = {
        'depth': best_params['depth'],
        #'depth':10,
        #'learning_rate': best_params['learning_rate'],
        'learning_rate': 0.1,
        #'l2_leaf_reg': best_params['l2_leaf_reg'],
        'l2_leaf_reg': 3,   # L2正则化系数，用于控制模型复杂度，防止过拟合，但过大的正则化系数可能会导致欠拟合。
        #'iterations': best_params['iterations'],
        'iterations': 800,
        #'iterations': best_params['iterations'],
        'loss_function': 'RMSE', #损失函数，用于衡量模型的预测误差，MAE更适合偏态数据集
        #'loss_function': best_params['loss_function'],
        'eval_metric': 'MAE', #评价指标，用于评估模型在验证集上的性能
        'thread_count': -1,#训练时所用的cpu/gpu核数
        'verbose': False, #是否输出训练过程信息
        'random_seed': 100, #随机种子，用于控制随机化过程的起点，保证每次训练的结果是可重复的。
        'cat_features': cat_features,
    }

    # 定义模型
    clf = CatBoostRegressor(**params)

    # 训练模型
    clf.fit(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=50, use_best_model=True, verbose = False)#early_stopping_rounds=50，即如果模型在验证集上的性能连续50次迭代都没有提升，就会提前停止训练，以避免过拟合。

    #计算回归模型的测试集决定系数
    score = clf.score(X_test, y_test)#用于计算模型在测试集上的性能表现，可以作为评估模型的重要指标之一，同时也可以用于比较不同模型的性能。
    #测试集预测
    y_pre = clf.predict(X_test)

    # 创建解释器对象
    explainer = shap.Explainer(clf)
    # 计算测试集的Shap值
    shap_values = explainer(X_test)
    all_shap_values.append(shap_values)

    # 保存模型
    CatBoostsave = datfile + str(temp_classinformation) + ".dat"
    pickle.dump(clf, open(CatBoostsave, "wb"))#pickle


    '''#shap
    plt.figure(figsize=(15, 10))
    # 创建解释器对象
    explainer = shap.Explainer(clf)

    # 计算测试集的Shap值
    shap_values = explainer(X_test)
    shap.summary_plot(shap_values, X_test, feature_names=Predict_features)
    plt.plot()'''

    #树可视化
    '''last_tree_idx = clf.tree_count_ - 1 #获取最后一棵树的索引

    pool = Pool(X_train, y_train, cat_features=cat_features, feature_names=feature_names)
    tree_dot = clf.plot_tree(
        tree_idx=last_tree_idx,
        pool=pool
    )
    tree_str = tree_dot.source  # 获取源代码字符串
    graph = graphviz.Source(tree_str)
    graph.render(datfile + str(temp_classinformation) + "_plot", format='png', cleanup=True)'''


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

    #记录
    #记录
    total_MSE = np.append(total_MSE, mse)
    total_MAE = np.append(total_MAE, mae)
    total_MAPE = np.append(total_MAPE, mape)
    total_R2 = np.append(total_R2, R2)
    total_test = np.append(total_test, y_test)
    total_pre = np.append(total_pre, y_pre)

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
        print("特征重要性：{0}".format(feature_importance), file = f)

    # 模型训练完毕
    print("区域{0}模型训练并测试完毕！".format(temp_classinformation))
    del X_train
    del X_test
    del y_train
    del y_test
    gc.collect()
    return total_MSE, total_MAE, total_MAPE, total_R2, total_test, total_pre,all_shap_values



def mape0(y_test, y_pre):
    mask = y_test > 1
    return np.mean(np.abs((y_pre[mask] - y_test[mask]) / y_test[mask])) * 100

def relative_error0(y_test, y_pre):
    mask = y_test > 1
    ree =  np.abs(y_test[mask] - y_pre[mask]) / y_test[mask] * 100
    return ree[ree < 100]