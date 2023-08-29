# -*- coding: utf-8 -*-
import os
import csv
import time
import math
import math

import warnings
import jenkspy
import threading
import queue
import datetime
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import silhouette_score
import shutil
import pickle
import gc
from osgeo import gdal
from kneed import KneeLocator
import itertools
import shutil
from sklearn.exceptions import ConvergenceWarning

from tqdm import tqdm
import arcpy

from arcpy import env
from arcpy.sa import *
import matplotlib
import matplotlib.pyplot as plt  # 画图
import sklearn
from sklearn.cluster import KMeans
from sklearn.cluster import OPTICS
import gdal
from sklearn.cluster import DBSCAN
from scipy.interpolate import griddata
import geopandas as gpd
import rasterio
from rasterio.plot import show
from sklearn.cluster import DBSCAN
from sklearn import metrics  # 模型结果指标库
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split,GridSearchCV
from catboost import CatBoostRegressor
from sklearn import preprocessing
import plotly.graph_objects as go
'''import warnings0.
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")'''

import RF
import Nozone_RF
import SVR
#import NN
import KJ_Catboost
import featureaszone_CatBoost
import Nozone_CatBoost
import featureasinput_RF
import Nozone_SVR
import featureasinput_SVR

matplotlib.rcParams['font.sans-serif'] = ['Arial']
matplotlib.rcParams['font.family'] = 'SimHei'
matplotlib.rcParams['font.size'] = 10
matplotlib.rcParams['axes.unicode_minus']=False

def input_with_timeout(prompt, timeout):
    result = []

    def get_input():
        result.append(input(prompt))

    t = threading.Thread(target=get_input)
    t.daemon = True  # 将线程设置为守护线程
    t.start()
    t.join(timeout)

    if t.is_alive():
        print("\n输入超时，将备注信息设置为 'none'")
        return 'none'

    return result[0]


def defineoutpath(Depthfilepre, stdepth, month):
    if float(stdepth) >= 8 and float(stdepth) < 13:
        standarddepth = "10"
    elif float(stdepth) >= 18 and float(stdepth) < 23:
        standarddepth = "20"
    elif float(stdepth) >= 28 and float(stdepth) < 33:
        standarddepth = "30"
    elif float(stdepth) >= 38 and float(stdepth) < 43:
        standarddepth = "40"
    elif float(stdepth) >= 48 and float(stdepth) < 53:
        standarddepth = "50"
    elif float(stdepth) >= 73 and float(stdepth) < 78:
        standarddepth = "75"
    elif float(stdepth) >= 98 and float(stdepth) < 103:
        standarddepth = "100"
    elif float(stdepth) >= 123 and float(stdepth) < 128:
        standarddepth = "125"
    elif float(stdepth) >= 148 and float(stdepth) < 153:
        standarddepth = "150"
    elif float(stdepth) >= 195 and float(stdepth) < 205:
        standarddepth = "200"
    elif float(stdepth) >= 245 and float(stdepth) < 255:
        standarddepth = "250"
    elif float(stdepth) >= 295 and float(stdepth) < 305:
        standarddepth = "300"
    elif float(stdepth) >= 395 and float(stdepth) < 405:
        standarddepth = "400"
    elif float(stdepth) >= 495 and float(stdepth) < 505:
        standarddepth = "500"
    elif float(stdepth) >= 595 and float(stdepth) < 605:
        standarddepth = "600"
    elif float(stdepth) >= 695 and float(stdepth) < 705:
        standarddepth = "700"
    elif float(stdepth) >= 795 and float(stdepth) < 805:
        standarddepth = "800"
    elif float(stdepth) >= 895 and float(stdepth) < 905:
        standarddepth = "900"
    elif float(stdepth) >= 995 and float(stdepth) < 1005:
        standarddepth = "1000"
    elif float(stdepth) >= 1095 and float(stdepth) < 1105:
        standarddepth = "1100"
    elif float(stdepth) >= 1195 and float(stdepth) < 1205:
        standarddepth = "1200"
    elif float(stdepth) >= 1295 and float(stdepth) < 1305:
        standarddepth = "1300"
    elif float(stdepth) >= 1395 and float(stdepth) < 1405:
        standarddepth = "1400"
    elif float(stdepth) >= 1495 and float(stdepth) < 1505:
        standarddepth = "1500"
    elif float(stdepth) >= 1745 and float(stdepth) < 1755:
        standarddepth = "1750"
    elif float(stdepth) >= 1995 and float(stdepth) < 2005:
        standarddepth = "2000"
    else:
        standarddepth = "NULL"
    numbers = {
        "01" : Depthfilepre + standarddepth + "dbar\\01\\" + "depth" + standarddepth + "_01.csv",
        "02" : Depthfilepre + standarddepth + "dbar\\02\\" + "depth" + standarddepth + "_02.csv",
        "03" : Depthfilepre + standarddepth + "dbar\\03\\" + "depth" + standarddepth + "_03.csv",
        "04" : Depthfilepre + standarddepth + "dbar\\04\\" + "depth" + standarddepth + "_04.csv",
        "05" : Depthfilepre + standarddepth + "dbar\\05\\" + "depth" + standarddepth + "_05.csv",
        "06" : Depthfilepre + standarddepth + "dbar\\06\\" + "depth" + standarddepth + "_06.csv",
        "07" : Depthfilepre + standarddepth + "dbar\\07\\" + "depth" + standarddepth + "_07.csv",
        "08" : Depthfilepre + standarddepth + "dbar\\08\\" + "depth" + standarddepth + "_08.csv",
        "09" : Depthfilepre + standarddepth + "dbar\\09\\" + "depth" + standarddepth + "_09.csv",
        "10" : Depthfilepre + standarddepth + "dbar\\10\\" + "depth" + standarddepth + "_10.csv",
        "11" : Depthfilepre + standarddepth + "dbar\\11\\" + "depth" + standarddepth + "_11.csv",
        "12" : Depthfilepre + standarddepth + "dbar\\12\\" + "depth" + standarddepth + "_12.csv",

    }
    if standarddepth != "NULL":
        return numbers.get(month, None)
    else:
        return standarddepth
def preprocess(originalfile,Depthfilepre,NO_Depthfilepre):# 数据库数据预处理
    with open(originalfile, mode="r", encoding="gbk") as f:
        reader = csv.DictReader(f)
        for row in reader:
            longitude = row["测量结束经度"]
            latitude = row["测量结束纬度"]
            date = row["测量结束日期"]
            try:
                if date != 99999 and date != 999999 and date != "" and date != "?":
                    #month_day = datetime.datetime.strptime(date, '%Y-%m-%d %H:%M:%S').strftime('%m-%d')
                    datetime_obj = datetime.datetime.strptime(date, '%Y-%m-%d %H:%M:%S')
                    year_month_day = datetime_obj.strftime('%Y-%m-%d')
                    #day_of_year = datetime_obj.timetuple().tm_yday
                    #sin_value = math.sin(2 * math.pi * day_of_year / 365.25)

                    month = datetime.datetime.strptime(date, '%Y-%m-%d %H:%M:%S').strftime('%m')
                    year = datetime.datetime.strptime(date, '%Y-%m-%d %H:%M:%S').strftime('%Y')
                else:
                    continue
            except:
                pass
                print(date)

            if int(year) == 2023 or int(year) < 2004:
                continue
            depth = row["压力"]
            num = (len(depth) - len(depth.replace(",", ""))) + 1  # 元素个数
            depth = depth.split(',')
            depth_qc = row["压力质量标示"]
            depth_qc = depth_qc.split(',')

            salinity = row["盐度"]
            salinity = salinity.split(',')
            salinity_qc = row["盐度质量标示"]
            salinity_qc = salinity_qc.split(',')

            temperature = row["温度"]
            temperature = temperature.split(',')
            temperature_qc = row["温度质量标示"]
            temperature_qc = temperature_qc.split(',')

            dis_oxygen = row["溶解氧"]
            if isinstance(dis_oxygen, int):
                dis_oxygen = str(dis_oxygen)
            if longitude != '99999' and latitude != '99999' and month != '99999' and year != '99999' and longitude != "" and latitude != "" and month != "" and year != "" and longitude != '999999' and latitude != '999999' and month != '999999' and year != '999999':
                if dis_oxygen != "99999" and dis_oxygen != "" and dis_oxygen != "999999":
                    dis_oxygen = dis_oxygen.split(',')
                    dis_oxygen_qc = row["溶解氧质量标示"]
                    dis_oxygen_qc = dis_oxygen_qc.split(',')
                    for i in range(num):
                        try:
                            if depth[i] != '99999' and salinity[i] != '99999' and temperature[i] != '99999' and dis_oxygen[i] != 'inf' \
                                    and depth[i] != "" and salinity[i] != "" and temperature[i] != "" and dis_oxygen[i] != "" and dis_oxygen[i] != '99999' \
                                    and float(depth_qc[i]) < 3.0 and float(salinity_qc[i]) < 3.0 and float(temperature_qc[i]) < 3.0 and float(dis_oxygen_qc[i]) < 3.0 \
                                    and float(depth_qc[i]) > 0.0 and float(salinity_qc[i]) > 0.0 and float(temperature_qc[i]) > 0.0 and float(dis_oxygen_qc[i]) > 0.0 \
                                    and float(salinity[i]) > 0 and float(temperature[i]) != 0 and float(dis_oxygen[i]) > 0 and float(latitude) > -80 and float(latitude) < 80 \
                                    and float(longitude) < 360 and float(longitude) > 0:
                                df = pd.DataFrame(data=[[depth[i], longitude, latitude, year, month, year_month_day, temperature[i], salinity[i], dis_oxygen[i]]])
                                outpath = defineoutpath(Depthfilepre, depth[i], month)
                                if outpath != "NULL":
                                    if not os.path.exists(outpath):
                                        #print(outpath)
                                        df.to_csv(outpath, header=['PRES', 'LONGITUDE', 'LATITUDE', 'YEAR', 'MOTH', 'DATE', 'TEMP', 'PSAL', 'DOXY'], index=False, mode='a', encoding="gbk")
                                    else:
                                        df.to_csv(outpath, header=False, index=False, mode='a', encoding="gbk")
                                        #os.remove(outpath)
                        except:
                            print('浮标{0}第{1}周期第{2}/{3}个数据处理失败'.format(row["浮标ID"], row["周期"], i+1, num))
                else:
                    for i in range(num):
                        try:
                            if depth[i] != '99999' and salinity[i] != '99999' and temperature[i] != '99999' \
                                    and depth[i] != "" and salinity[i] != "" and temperature[i] != "" \
                                    and float(depth_qc[i]) < 3.0 and float(salinity_qc[i]) < 3.0 and float(temperature_qc[i]) < 3.0 \
                                    and float(depth_qc[i]) > 0.0 and float(salinity_qc[i]) > 0.0 and float(temperature_qc[i]) > 0.0 \
                                    and float(salinity[i]) > 0 and float(temperature[i]) != 0 and float(latitude) > -80 and float(latitude) < 80 \
                                    and float(longitude) < 360 and float(longitude) > 0:
                                df = pd.DataFrame(data=[[depth[i], longitude, latitude, year, month, year_month_day, temperature[i], salinity[i]]])
                                outpath = defineoutpath(NO_Depthfilepre, depth[i], month)
                                if outpath != "NULL":
                                    if not os.path.exists(outpath):
                                        df.to_csv(outpath, header=['PRES', 'LONGITUDE', 'LATITUDE', 'YEAR', 'MOTH', 'DATE', 'TEMP', 'PSAL'], index=False, mode='a', encoding="gbk")
                                    else:
                                        df.to_csv(outpath, header=False, index=False, mode='a', encoding="gbk")
                        except:
                            print('浮标{0}第{1}周期第{2}/{3}个数据处理失败'.format(row["浮标ID"], row["周期"], i+1, num))
            else:
                continue
print("数据预处理成功")


def makeseamake(mainseatif,de,seafile,landfile):
    env.workspace = "D:\\Data\\ArcGIS_Data\\temp"
    arcpy.env.overwriteOutput = True
    print("---开始处理深度{}---".format(de))
    intde = int(de)
    temppath = "D:\\Data\\ArcGIS_Data\\temp"
    Racalcu_tif = temppath + "\\Racalcu" + de + ".tif"
    polygon_feature = temppath + "\\polygon" + de + ".shp"
    out_feature_calss = temppath + "\\feature_sea" + de + ".shp"
    out_sea = "outsea" + de + ".shp"
    out_land = "outland" + de + ".shp"
    out_sea_file = temppath + "\\outsea" + de + ".shp"
    out_land_file = temppath + "\\outland" + de + ".shp"

    try:
        # 进行栅格计算，输出结果为0或1的栅格
        expression = "Con(\"{0}\">{1}, 1, 0)".format(mainseatif, -intde)
        arcpy.gp.RasterCalculator_sa(expression, Racalcu_tif)
        print("栅格计算完成")
        # 栅格转面
        arcpy.conversion.RasterToPolygon(Racalcu_tif, polygon_feature, "SIMPLIFY")
        print("栅格转面完成")
        #融合
        arcpy.management.Dissolve(polygon_feature, out_feature_calss, "gridcode")
        print("融合完成")
        #导出要素
        arcpy.conversion.FeatureClassToFeatureClass(out_feature_calss, temppath, out_sea, "gridcode = 0")
        arcpy.conversion.FeatureClassToFeatureClass(out_feature_calss, temppath, out_land, "gridcode = 1")
        print("导出要素完成")
        #投影
        wkt3 = "GEOGCS['WGS 1984_3',DATUM['D_WGS_1984',SPHEROID['WGS_1984',6378137.0,298.257223563]],\
                      PRIMEM['Greenwich',-150.0],UNIT['Degree',0.0174532925199433]];\
                      -400 -400 1000000000;-100000 10000;-100000 10000;8.98315284119522E-09;\
                      0.001;0.001;IsHighPrecision"
        outCS = arcpy.SpatialReference(text = wkt3)
        arcpy.Project_management(out_sea_file, seafile, outCS)
        print("sea投影完成")
        arcpy.Project_management(out_land_file, landfile, outCS)
        print("land投影完成")
    except Exception as ex:
        print(ex.args[0])




def Standardlayerfilter(filename,outfile,element):
    if os.path.exists(outfile):
        os.remove(outfile)
    '''df = pd.read_csv(filename)
    df['presvar'] = (df.PRES-int(dbar)).abs()
    df=df.reset_index()
    df = df.sort_values(by=['LONGITUDE','presvar'], ascending=True)
    df= df.drop_duplicates('LONGITUDE', keep='first')
    df = df.drop(['index','presvar'],axis=1)'''
    # 读取数据文件
    df = pd.read_csv(filename)
    # 将经度范围转换为-180到180
    df['LONGITUDE'] = df['LONGITUDE'].apply(lambda x: x - 360 if x > 180 else x)
    # 按LONGITUDE，LATITUDE，YEAR，MOTH分组
    grouped = df.groupby(['LONGITUDE', 'LATITUDE', 'YEAR', 'MOTH'])

    # 定义异常值剔除函数
    def remove_outlier(group):
        if len(group) >= 3:
            mean = group.mean()
            std = group.std()
            group = group[(group - mean).abs() <= 3 * std]
        return group.mean()

    # 对TEMP，PSAL，DOXY分别进行异常值剔除和均值计算
    result = grouped[element].apply(remove_outlier)
    # 将分组的LONGITUDE，LATITUDE，YEAR，MOTH属性和计算出来的均值合并为一个DataFrame
    result = pd.concat([grouped[['LONGITUDE', 'LATITUDE', 'YEAR', 'MOTH']].first(), result], axis=1)

    # 将结果写入out.csv文件
    result.to_csv(outfile, index=False)

    print("标准层数据筛选完成")


def XY_point(file, out_point_file):
    try:
        in_table = file
        out_feature_class = out_point_file + ".shp"
        x_coords = "LONGITUDE"
        y_coords = "LATITUDE"
        z_coords = ""
        arcpy.management.XYTableToPoint(in_table, out_feature_class,
                                        x_coords, y_coords, z_coords,
                                        arcpy.SpatialReference("WGS 1984"))
        print("XY转Point完成，共{0}个点".format(arcpy.GetCount_management(out_feature_class)))
    except Exception as ex:
        print(ex.args[0])

def Project_1984_3(wkt,out_point_file,out_point_projectname):
    out_point_file = out_point_file + ".shp"
    try:
        dsc = arcpy.Describe(out_point_file)
        if dsc.spatialReference.Name == "Unknown":
            print('坐标系未定义: ' + out_point_file)
        else:
            outfc = os.path.join(outWorkspace, out_point_projectname)
            outCS = arcpy.SpatialReference(text = wkt)
            arcpy.Project_management(out_point_file, outfc, outCS)
            #print(arcpy.GetMessages())
            print("投影完成")
    except Exception as ex:
        print(ex.args[0])

def delete_land_point(wordmap,out_point_projectname,out_point_projecttrue):
    try:
        tempLayer = "pointLayer"
        arcpy.CopyFeatures_management(out_point_projectname, out_point_projecttrue)
        arcpy.MakeFeatureLayer_management(out_point_projecttrue, tempLayer)
        arcpy.SelectLayerByLocation_management(tempLayer, 'WITHIN_CLEMENTINI', wordmap)
        landpointnum = int(arcpy.GetCount_management(tempLayer)[0])
        if landpointnum > 0:
            arcpy.DeleteFeatures_management(tempLayer)
        print("删除陆地上{0}个点完成".format(landpointnum))
    except Exception as ex:
        print(ex.args[0])

def SplineandBarriers(shapefile, Splineout, Splineout_real,maxgrid,mingrid):
    try:
        outSplineBarriers = SplineWithBarriers(shapefile, "DOXY", JHcoastline,1)
        outSplineBarriers.save(Splineout)
        out_rc_minus_raster = RasterCalculator([Splineout], ["X"], "Con(X <= " + mingrid + ", " + mingrid + ", X)")
        out_rc_minus_raster1 = RasterCalculator([out_rc_minus_raster], ["X"], "Con(X > " + maxgrid + ", " + maxgrid + ", X)")
        out_rc_minus_raster1.save(Splineout_real)
        print("插值完成")
    except Exception as ex:
        print(ex.args[0])

def tifmaxmin(Splineout_real):
    # 打开插值后的栅格图像
    raster = gdal.Open(Splineout_real)

    # 获取栅格图像的行数、列数
    x_size = raster.RasterXSize
    y_size = raster.RasterYSize

    # 获取波段
    band = raster.GetRasterBand(1)

    # 获取像元值范围
    min_value = band.GetMinimum()
    max_value = band.GetMaximum()
    if min_value is None or max_value is None:
        (min_value, max_value) = band.ComputeRasterMinMax(1, x_size, y_size)

    print("TIF图像: Min={0}, Max={1}".format(min_value, max_value))
    return min_value,max_value


def calculate_bic(array, breaks):# 计算贝叶斯信息准则（BIC）
    num_classes = len(breaks) - 1
    n = len(array)
    residuals = []

    for i in range(num_classes):
        if i < num_classes - 1:
            class_data = array[(array >= breaks[i]) & (array < breaks[i+1])]
        else:
            class_data = array[(array >= breaks[i]) & (array <= breaks[i+1])]
        class_mean = np.mean(class_data)
        class_residuals = class_data - class_mean
        residuals.extend(class_residuals)

    rss = sum(res ** 2 for res in residuals)
    bic = num_classes * np.log(n) - 2 * np.log(rss)
    return bic

def interbreak(kmeans,input_raster):

    # 获取NoData值
    raster_obj = arcpy.Raster(input_raster)
    no_data_value = raster_obj.noDataValue

    # 将栅格数据转换为NumPy数组
    array = arcpy.RasterToNumPyArray(input_raster)
    array = array[array != no_data_value]


    nclasses = int(kmeans)
    '''gvf = 0.0
    while gvf < .9 and nclasses < 16:#
        gvf = goodness_of_variance_fit(array, nclasses)
        print(gvf)
        nclasses += 1
    nclasses -= 1'''
    best_breaks = jenkspy.jenks_breaks(array, nclasses)

    # 输出结果
    #print("最优分类数：", nclasses)
    print("等值线间隔数组为：", best_breaks)

    return best_breaks

def goodness_of_variance_fit(array, classes):
    # get the break points
    classes = jenkspy.jenks_breaks(array, classes)
    # do the actual classification
    classified = np.array([classify(i, classes) for i in array])
    # max value of zones
    maxz = max(classified)
    # nested list of zone indices
    zone_indices = [[idx for idx, val in enumerate(classified) if zone + 1 == val] for zone in range(maxz)]
    # sum of squared deviations from array mean
    sdam = np.sum((array - array.mean()) ** 2)
    # sorted polygon stats
    array_sort = [np.array([array[index] for index in zone]) for zone in zone_indices]
    # sum of squared deviations of class means
    sdcm = sum([np.sum((classified - classified.mean()) ** 2) for classified in array_sort])
    # goodness of variance fit
    gvf = (sdam - sdcm) / sdam
    return gvf

def classify(value, breaks):
    for i in range(1, len(breaks)):
        if value < breaks[i]:
            return i
    return len(breaks) - 1

def kmeansfun(CatBoostsave,Splineout_real):
    def read_tif(file_path):
        with rasterio.open(file_path) as src:
            image = src.read(1)
            mask = src.read_masks(1)
            profile = src.profile
        return image, mask, profile


    # 寻找肘部点
    def find_elbow_point(sse):
        diff = np.diff(sse)
        diff_ratio = np.divide(diff[:-1], diff[1:])
        elbow_index = np.argmax(diff_ratio) + 1
        elbow_point = elbow_index + 2  # 聚类个数从2开始
        return elbow_point

    def find_optimal_clusters(data, max_clusters):
        sse = []
        range_n_clusters = range(2, max_clusters + 1)

        for n_clusters in range_n_clusters:
            kmeans = KMeans(n_clusters=n_clusters, random_state=0)
            kmeans.fit(data)
            sse.append(kmeans.inertia_)

        plt.figure()
        plt.plot(range_n_clusters, sse, 'bo-')
        plt.xlabel('Number of Clusters')
        plt.ylabel('SSE')
        plt.title('Elbow Method')
        plt.grid(True)
        plt.savefig(CatBoostsave + 'elbow.png')  # 保存肘部法图到文件

        # 寻找肘部点
        elbow_point = find_elbow_point(sse)

        return elbow_point

    # 主函数
    def main(input_tif, output_tif):
        image, mask, profile = read_tif(input_tif)
        flattened_image = image[mask != 0].reshape(-1, 1)

        optimal_clusters = find_optimal_clusters(flattened_image, max_clusters=16)
        print(f"最佳聚类个数: {optimal_clusters}")
        return optimal_clusters

    optimal_clusters = main(Splineout_real, CatBoostsave + 'cluster.tif')
    return optimal_clusters


def Contourarea(Splineout_real, outContours,contourInterval,baseContour):
    try:
        Contour(Splineout_real, outContours, contourInterval, baseContour, contour_type = "CONTOUR_POLYGON")
        print("等值面划分完成")
    except Exception as ex:
        print(ex.args[0])

def Contourbreaks(input_raster,contour_intervals):
    output_feature_class = "contours_merged.shp"
    # 创建一个空的多边形要素类
    arcpy.CreateFeatureclass_management(arcpy.env.workspace, output_feature_class, "POLYGON")

    # 使用等值线间隔数组生成等值面
    for i in range(len(contour_intervals) - 1):
        lower_bound = float(contour_intervals[i])
        upper_bound = float(contour_intervals[i + 1])

        # 创建临时的二值栅格
        temp_raster = Con((input_raster >= lower_bound) & (input_raster < upper_bound), 1)

        # 将二值栅格转换为多边形
        temp_polygon = arcpy.CreateUniqueName("temp_polygon.shp", arcpy.env.workspace)
        arcpy.RasterToPolygon_conversion(temp_raster, temp_polygon, "NO_SIMPLIFY", "Value")

        # 进行融合，将多边形属性合并到一个要素中
        dissolve_polygons = arcpy.CreateUniqueName("dissolve_polygons.shp", arcpy.env.workspace)
        arcpy.Dissolve_management(temp_polygon, dissolve_polygons)

        # 将生成的多边形添加到输出要素类中
        arcpy.Append_management(dissolve_polygons, output_feature_class, "NO_TEST")

        # 删除临时多边形要素类
        arcpy.Delete_management(temp_polygon)
        arcpy.Delete_management(dissolve_polygons)
    return output_feature_class

def Featogrid(outContours,outContourgrid):
    try:
        field = "FID"
        arcpy.FeatureToRaster_conversion(outContours, field, outContourgrid)
        print("等值面转栅格完成")
    except Exception as ex:
        print(ex.args[0])

def Valuetopoint(out_point_projecttrue,outContourgrid,outPointFeatures,PointFeaturestrue,new_field_name):
    try:
        ExtractValuesToPoints(out_point_projecttrue, outContourgrid, outPointFeatures,"NONE", "VALUE_ONLY")
        arcpy.AddField_management(outPointFeatures, new_field_name, "FLOAT")
        arcpy.CalculateField_management(outPointFeatures, new_field_name, "!RASTERVALU!", "PYTHON3")
        # 删除旧字段
        arcpy.DeleteField_management(outPointFeatures, "RASTERVALU")
        tempLayer = "pointLayer1"
        arcpy.CopyFeatures_management(outPointFeatures, PointFeaturestrue)
        arcpy.MakeFeatureLayer_management(PointFeaturestrue, tempLayer)

        arcpy.SelectLayerByAttribute_management(tempLayer, 'NEW_SELECTION', f'{new_field_name} = -9999')
        if int(arcpy.GetCount_management(tempLayer)[0]) > 0:
            arcpy.DeleteFeatures_management(tempLayer)
        print("值提取到点并后处理完成")
    except Exception as ex:
        print(ex.args[0])


def find_tif_file(year, month, directory):
    # 遍历目录下的所有文件
    for filename in os.listdir(directory):
        # 检查文件名是否符合要求
        if filename.endswith(".tif"):
            parts = filename.split("_")
            if parts[2] == str(year):
                # 如果文件名符合要求，返回文件的绝对路径
                return os.path.join(directory, filename)
    # 如果找不到符合条件的文件，返回None
    return None
def valuetonewfield(csv_file,output_csv_file,folder_path,field):
    if os.path.exists(output_csv_file):
        os.remove(output_csv_file)
    with open(csv_file, "r") as f:
        # 使用csv模块读取csv文件
        reader1 = csv.reader(f)
        num_lines = sum(1 for row in reader1)
    with open(csv_file, "r") as f:
        # 使用csv模块读取csv文件
        reader = csv.DictReader(f)
        # 打开输出csv文件
        with open(output_csv_file, "w", newline="") as fout:
            # 定义输出csv文件的字段名
            fieldnames = reader.fieldnames + [field]
            # 使用csv模块写入输出csv文件的头部
            writer = csv.DictWriter(fout, fieldnames=fieldnames)
            writer.writeheader()
            progress_bar = tqdm(reader, desc="Processing rows", total=num_lines)
            for row in progress_bar:
                # 获取year和month
                year = int(float(row["YEAR"]))
                month = int(float(row["MOTH"]))
                if year < 2004:
                    continue
                # 获取对应tif文件路径
                tif_file = find_tif_file(year, month, folder_path)
                # 若对应tif文件不存在，则跳过该行
                if tif_file is None:
                    continue

                out_raster = tif_file
                longitude = float(row['LONGITUDE'])
                latitude = float(row['LATITUDE'])
                # 定义输入和输出空间参考
                in_spatial_ref = arcpy.SpatialReference(4326)  # WGS 1984
                out_spatial_ref = arcpy.SpatialReference()  # 创建一个新的空间参考对象
                out_spatial_ref.loadFromString("GEOGCS['WGS 1984_3',DATUM['D_WGS_1984',SPHEROID['WGS_1984',6378137.0,298.257223563]],\
                      PRIMEM['Greenwich',-150.0],UNIT['Degree',0.0174532925199433]];\
                      -400 -400 1000000000;-100000 10000;-100000 10000;8.98315284119522E-09;\
                      0.001;0.001;IsHighPrecision")  # 使用你的自定义坐标系的WKT字符串

                # 将输入点从WGS 1984坐标系投影到自定义坐标系
                projected_x, projected_y = project_point(longitude, latitude, in_spatial_ref, out_spatial_ref)


                #print(f'x:{x},y:{y}')
                point = str(projected_x) + ' ' + str(projected_y)
                # 查询该点在tif文件中对应的值
                result = arcpy.GetCellValue_management(out_raster, point)
                try:
                    cellvalue = round(float(result.getOutput(0)), 3)
                except:
                    cellvalue = "NoData"
                # 将查询结果填入新字段中
                row[field] = cellvalue
                # 写入新csv文件
                writer.writerow(row)
                progress_bar.update(1)

def project_point(longitude, latitude, in_spatial_ref, out_spatial_ref):
    # 创建输入点要素类
    in_point = arcpy.Point(longitude, latitude)
    in_point_geometry = arcpy.PointGeometry(in_point, in_spatial_ref)

    # 使用 projectAs 方法投影点
    projected_point_geometry = in_point_geometry.projectAs(out_spatial_ref)

    # 提取投影后的坐标并返回
    projected_x = projected_point_geometry.firstPoint.X
    projected_y = projected_point_geometry.firstPoint.Y
    return projected_x, projected_y

def deletenodata(input_csv, output_csv, field):
    # 定义删除行的计数器
    delete_count = 0

    # 打开输入csv文件
    with open(input_csv, 'r') as input_file:
        # 创建csv读取器
        reader = csv.DictReader(input_file)

        # 打开输出csv文件
        with open(output_csv, 'w', newline='') as output_file:
            # 创建csv写入器
            writer = csv.DictWriter(output_file, fieldnames=reader.fieldnames)

            # 写入csv头部信息
            writer.writeheader()
            # 遍历每一行数据
            for row in reader:
                # 如果field属性值为"NoData"，则跳过该行
                if str(row[field]) == 'NoData' or str(row[field]) == '-1':
                    delete_count += 1
                    continue
                # 否则，将该行写入输出csv文件
                writer.writerow(row)
    # 输出删除行的个数
    print(f'共删除{delete_count}行数据')

def Valuetocsv(PointFeaturestrue,outPointcsv,talbename):
    filename = outPointcsv + "\\" + talbename
    if os.path.exists(filename):
        os.remove(filename)
    try:
        arcpy.TableToTable_conversion(PointFeaturestrue, outPointcsv, talbename)
        print("转CSV完成")
    except Exception as ex:
        print(ex.args[0])
    df = pd.read_csv(filename)
    df['zone'] = df['zone'].astype(int)
    df.to_csv(filename, index=False)


def merge_left_right(data_count,i,data,k):
    left = data[i-1]
    right = data[i+1]
    # 如果左分区的数量小于右分区，则将左右合并，否则将当前分区和右分区合并
    if left[1] <= right[1]:
        merged_zone = left[0] + ',' + data[i][0]
        merged_count = left[1] + data[i][1]
        data[i-1] = [merged_zone, merged_count]
        del data[i]
        data_count -= 1
        i -= 2
    elif left[1] > right[1]:
        merged_zone = data[i][0] + ',' + right[0]
        merged_count = data[i][1] + right[1]
        data[i] = [merged_zone,merged_count]
        del data[i+1]
        data_count -= 1
        if i == data_count - 1 and merged_count < k:
            merged_zone = data[i-1][0] + ',' + data[i][0]
            merged_count = data[i-1][1] + data[i][1]
            data[i-1] = [merged_zone,merged_count]
            del data[i]
            data_count -= 1
        if merged_count < k and i < data_count - 1:
            data_count,i,data = merge_left_right(data_count,i,data,k)
        i -= 1
    i= i+1
    return data_count,i,data

def merge_function(data, k):
    # 初始化结果列表
    data = data.copy()
    #data = [['分区1', 170], ['分区2', 150], ['分区3', 180], ['分区4', 200], ['分区5', 20], ['分区6', 20],['分区7', 220]]
    data_count = len(data)
    # 遍历所有分区
    i = 0
    while i < data_count:
        # 如果该分区数据个数大于等于k，直接将其添加到结果列表
        if data[i][1] >= k:
            i=i+1
            continue
        else:
            # 如果该分区是第一个分区，只考虑和右侧相邻分区的合并
            if i == 0:
                merged_zone = data[i][0] + ',' + data[i+1][0]
                merged_count = data[i][1] + data[i+1][1]
                data[i] = [merged_zone,merged_count]
                del data[1]
                data_count = data_count - 1

                # 如果合并后的新分区数据个数仍小于k，则继续向右合并
                while merged_count < k and i+1 < data_count:
                    i += 1
                    merged_zone = merged_zone + ',' + data[i][0]
                    merged_count += data[i][1]
                    data[i-1] = [merged_zone,merged_count]
                    del data[i]
                    i=i-1
                    data_count = data_count - 1

                i = i +1
                del merged_zone
                del merged_count

            elif i != data_count - 1:
                data_count,i,data = merge_left_right(data_count,i,data,k)

            elif i == data_count - 1:
                merged_zone = data[i-1][0] + ',' + data[i][0]
                merged_count = data[i-1][1] + data[i][1]
                data[i-1] = [merged_zone,merged_count]
                del data[i]
                data_count = data_count - 1
                del merged_zone
                del merged_count
                i=i+1


    return data


def merage_zone(file,k,CatBoostsave):
    # 读取csv文件并获取所有数据行
    with open(file) as csvfile:
        reader = csv.DictReader(csvfile)
        rows = [row for row in reader]

    # 统计每个分区的数据行数
    counts = {}
    for row in rows:
        rastervalue = row['zone']
        if rastervalue not in counts:
            counts[rastervalue] = 0
        counts[rastervalue] += 1

    # 将统计结果写入一个列表中，并按照分区名称进行排序
    result = []
    for rastervalue, count in counts.items():
        result.append([rastervalue, count])
    result.sort()
    sorted_lst = sorted(result, key=lambda x: int(x[0]))

    result = merge_function(sorted_lst, k)
    new_result = [[zone] if ',' not in zone else zone.split(',') for zone, count in result]
    txtfile = CatBoostsave + ".txt"
    with open(txtfile, 'a') as f:  # 设置文件对象
        print("各分区数目:", file = f)
        print(counts, file = f)
        print("分区合并结果:", file = f)
        print(result, file = f)
        #print(new_result, file = f)
    return new_result

def meragecsv(resultlist,file):
    # 读取csv文件并将数据存储在一个列表中
    with open(file, newline='') as f:
        reader = csv.DictReader(f)
        data = [row for row in reader]

    # 根据子集元素的数量进行修改操作
    for subset in resultlist:
        if len(subset) > 1:
            value = subset[0]
            new_subset = subset[1:]
            for row in data:
                if row['zone'] in new_subset:
                    row['zone'] = value

    if os.path.exists(file):
        os.remove(file)
    # 将修改后的数据写回csv文件
    with open(file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=reader.fieldnames)
        writer.writeheader()
        writer.writerows(data)
    print("合并分区完成")


def predict(zfile, deletefeatures, Modelsave, predictfile, predicted):
    if os.path.exists(predicted):
        os.remove(predicted)
    datfile = Modelsave

    # Catboost模型预测
    predata = pd.read_csv(predictfile)
    classinformation = predata["zone"].unique()
    #根据zone中的数值划分数值
    for temp_classinformation in classinformation:
        temp_data = predata[predata["zone"].isin([temp_classinformation])].copy()
        try:
            temp_classinformation = int(temp_classinformation)
        except:
            temp_classinformation = temp_classinformation
            print('预测时int(temp_classinformation)出现错误！')
        Modelsave = datfile + str(temp_classinformation) +".dat"

        X_pred = temp_data.loc[:, Predict_features].values

        '''# 加载标准化参数    
        Scaler = pickle.load(datfile + str(temp_classinformation) + ".save")
        X_pred= Scaler.transform(X_pred)'''

        try:
            Model1 = pickle.load(open(Modelsave, "rb"))
        except:
            print("找不到训练好的区域{0}模型！".format(temp_classinformation))
            continue
        prediction = Model1.predict(X_pred)


        temp_data['DOXY'] = prediction
        #删除指定列
        temp_data = temp_data.drop(deletefeatures, axis=1)

        if not os.path.exists(predicted):
            pd.DataFrame(temp_data).to_csv(predicted, header=True,index=False, mode='a')
        else:
            pd.DataFrame(temp_data).to_csv(predicted, header=False, index=False, mode='a')
        print("区域{0}模型预测完毕！".format(temp_classinformation))
        del temp_data
        del X_pred
        gc.collect()

    #将原始有溶解氧数据追加到预测结果文件中
    data_ = pd.read_csv(zfile)
    data_1 = data_.loc[:, ['LONGITUDE', 'LATITUDE', 'YEAR', 'MOTH', 'DOXY']]
    data_1.to_csv(predicted, mode='ab', header=False, index=None)
    # 删除'DOXY'属性值小于等于0的行
    df = pd.read_csv(predicted)
    df = df[df['DOXY'] > 0]
    df.to_csv(predicted, index=False)

def load_scaler(load_path):
    with open(load_path, 'rb') as f:
        scaler = pickle.load(f)
    return scaler

def predictSP(deletefeatures, Modelsave, predictfile, predicted):
    if os.path.exists(predicted):
        os.remove(predicted)
    datfile = Modelsave

    # Catboost模型预测
    predata = pd.read_csv(predictfile)
    classinformation = predata["zone"].unique()
    #根据zone中的数值划分数值
    for temp_classinformation in classinformation:
        temp_data = predata[predata["zone"].isin([temp_classinformation])].copy()
        try:
            temp_classinformation = int(temp_classinformation)
        except:
            temp_classinformation = temp_classinformation
            print('预测时int(temp_classinformation)出现错误！')
        Modelsave = datfile + str(temp_classinformation) +".dat"

        X_pred = temp_data.loc[:, Predict_features].values

        try:
            # 加载标准化参数
            with open(datfile + str(temp_classinformation) + "X.pkl", "rb") as scaler_file:
                Scaler_X = pickle.load(scaler_file)
            X_pred= Scaler_X.transform(X_pred)
            print("SVR标准化完成")
        except:
            print("SVR标准化失败！")

        try:
            Model1 = pickle.load(open(Modelsave, "rb"))
        except:
            print("找不到训练好的区域{0}模型！".format(temp_classinformation))
        prediction = Model1.predict(X_pred)
        temp_data['DOC'] = prediction
        #删除指定列
        temp_data = temp_data.drop(deletefeatures, axis=1)

        if not os.path.exists(predicted):
            pd.DataFrame(temp_data).to_csv(predicted, header=True,index=False, mode='a')
        else:
            pd.DataFrame(temp_data).to_csv(predicted, header=False, index=False, mode='a')
        print("区域{0}模型预测完毕！".format(temp_classinformation))
        del temp_data
        del X_pred
        gc.collect()


def predictNSP(deletefeatures, Modelsave, predictfile, predicted):
    if os.path.exists(predicted):
        os.remove(predicted)
    datfile = Modelsave

    # Catboost模型预测
    temp_data = pd.read_csv(predictfile)
    temp_classinformation = 999
    Modelsave = datfile + str(temp_classinformation) +".dat"
    X_pred = temp_data.loc[:, Predict_features].values

    try:
        # 加载标准化参数
        with open(datfile + str(temp_classinformation) + "X.pkl", "rb") as scaler_file:
            Scaler_X = pickle.load(scaler_file)
        X_pred= Scaler_X.transform(X_pred)
        print("SVR标准化完成")
    except:
        print("SVR标准化失败！")

    try:
        Model1 = pickle.load(open(Modelsave, "rb"))
    except:
        print("找不到训练好的区域{0}模型！".format(temp_classinformation))
    prediction = Model1.predict(X_pred)
    temp_data['DOC'] = prediction
    #删除指定列
    temp_data = temp_data.drop(deletefeatures, axis=1)

    if not os.path.exists(predicted):
        pd.DataFrame(temp_data).to_csv(predicted, header=True,index=False, mode='a')
    else:
        pd.DataFrame(temp_data).to_csv(predicted, header=False, index=False, mode='a')
    print("区域{0}模型预测完毕！".format(temp_classinformation))
    del temp_data
    del X_pred
    gc.collect()


def predictPIP(deletefeatures, Modelsave, predictfile, predicted):
    if os.path.exists(predicted):
        os.remove(predicted)
    datfile = Modelsave

    # Catboost模型预测
    temp_data = pd.read_csv(predictfile)
    temp_classinformation = 789
    Modelsave = datfile + str(temp_classinformation) +".dat"
    X_pred = temp_data.loc[:, ['LONGITUDE', 'LATITUDE', 'TEMP', 'PSAL','YEAR' ,'zone']].values

    try:
        # 加载标准化参数
        with open(datfile + str(temp_classinformation) + "X.pkl", "rb") as scaler_file:
            Scaler_X = pickle.load(scaler_file)
        X_pred= Scaler_X.transform(X_pred)
        print("SVR标准化完成")
    except:
        print("SVR标准化失败！")

    try:
        Model1 = pickle.load(open(Modelsave, "rb"))
    except:
        print("找不到训练好的区域{0}模型！".format(temp_classinformation))
    prediction = Model1.predict(X_pred)
    temp_data['DOC'] = prediction
    #删除指定列
    temp_data = temp_data.drop(deletefeatures, axis=1)

    if not os.path.exists(predicted):
        pd.DataFrame(temp_data).to_csv(predicted, header=True,index=False, mode='a')
    else:
        pd.DataFrame(temp_data).to_csv(predicted, header=False, index=False, mode='a')
    print("区域{0}模型预测完毕！".format(temp_classinformation))
    del temp_data
    del X_pred
    gc.collect()



def predict2023(Modelsave, predictfile, predicted):
    if os.path.exists(predicted):
        os.remove(predicted)
    datfile = Modelsave

    # Catboost模型预测
    predata = pd.read_csv(predictfile)
    classinformation = predata["zone"].unique()
    #根据zone中的数值划分数值
    for temp_classinformation in classinformation:
        temp_data = predata[predata["zone"].isin([temp_classinformation])].copy()
        try:
            temp_classinformation = int(temp_classinformation)
        except:
            temp_classinformation = temp_classinformation
            print('预测时int(temp_classinformation)出现错误！')
        Modelsave = datfile + str(temp_classinformation) +".dat"

        X_pred = temp_data.loc[:, Predict_features].values

        '''# 加载标准化参数    
        Scaler = pickle.load(datfile + str(temp_classinformation) + ".save")
        X_pred= Scaler.transform(X_pred)'''

        try:
            Model1 = pickle.load(open(Modelsave, "rb"))
        except:
            print("找不到训练好的区域{0}模型！".format(temp_classinformation))
            continue
        prediction = Model1.predict(X_pred)


        temp_data['DOC'] = prediction


        if not os.path.exists(predicted):
            pd.DataFrame(temp_data).to_csv(predicted, header=True,index=False, mode='a')
        else:
            pd.DataFrame(temp_data).to_csv(predicted, header=False, index=False, mode='a')
        print("区域{0}模型预测完毕！".format(temp_classinformation))
        del temp_data
        del X_pred
        gc.collect()


def Pickyear(ye,out_point_project,pointbyyear):
    try:
        ye = int(ye)
        experession = '"YEAR" = {}'.format(ye)
        tempLayer = "pointLayer2"
        arcpy.MakeFeatureLayer_management(out_point_project, tempLayer)
        arcpy.SelectLayerByAttribute_management(tempLayer, 'NEW_SELECTION', experession)
        if int(arcpy.GetCount_management(tempLayer)[0]) > 0:
            arcpy.CopyFeatures_management(tempLayer, pointbyyear)
            print("{0}年筛选完成".format(ye))
        else:
            print("{0}年无数据".format(ye))
    except Exception as ex:
        print(ex.args[0])

def SplineandBarrier(shapefile, Splineout, Splineout_real):
    try:
        outSplineBarriers = SplineWithBarriers(shapefile, "DOXY", JHcoastline, 1)
        outSplineBarriers.save(Splineout)
        out_rc_minus_raster = RasterCalculator([Splineout], ["X"], "Con(X <= 0, 0.001, X)")
        out_rc_minus_raster.save(Splineout_real)
        print("插值完成")
    except Exception as ex:
        print(ex.args[0])

def maskextract(Splineout_real3,maskfile,finalout):
    try:
        outExtractByMask = ExtractByMask(Splineout_real3, maskfile)
        outExtractByMask.save(finalout)
        print("按掩膜提取完成")
    except Exception as ex:
        print(ex.args[0])

def deletematchdata(file_a,file_b,file_c):
    if os.path.exists(file_c):
        os.remove(file_c)
    # 读取CSV文件
    a_df = pd.read_csv(file_a)
    b_df = pd.read_csv(file_b)

    # 合并数据集，找到在文件a和b中共有的数据行
    merged_df = a_df.merge(b_df, on=['LONGITUDE', 'LATITUDE', 'YEAR'], how='inner')

    # 从文件a中删除与文件b相匹配的数据行
    matched_indices = a_df.index[a_df[['LONGITUDE', 'LATITUDE', 'YEAR']].apply(tuple, axis=1).isin(merged_df[['LONGITUDE', 'LATITUDE', 'YEAR']].apply(tuple, axis=1))]
    a_df_cleaned = a_df.drop(matched_indices)

    # 将结果保存到新的CSV文件
    a_df_cleaned.to_csv(file_c, index=False)

def parse_date(date_str):
    year, month = map(int, date_str.split('.'))
    return datetime.date(year, month, 1)

def get_periods(periods):
    result = []
    for start, end in periods:
        result.append((parse_date(start), parse_date(end)))
    return result

def is_in_periods(date, periods):
    for start, end in periods:
        if start <= date <= end:
            return True
    return False

def add_event_column(input_csv, output_csv):

    el_nino_periods = [
        ('2002.08', '2003.03'),
        ('2006.08', '2007.01'),
        ('2009.10', '2010.03'),
        ('2015.05', '2016.05')
    ]

    la_nina_periods = [
        ('2000.01', '2001.06'),
        ('2005.10', '2006.04'),
        ('2007.06', '2009.05'),
        ('2010.06', '2012.03'),
        ('2017.07', '2018.06'),
        ('2020.06', '2021.12'),
    ]

    el_nino_periods = get_periods(el_nino_periods)
    la_nina_periods = get_periods(la_nina_periods)

    df = pd.read_csv(input_csv)
    # 判断'EVENT'列是否存在，存在则删除
    if 'EVENT' in df.columns:
        data = df.drop(columns=['EVENT'])

    df['EVENT'] = df.apply(lambda row: 'ENSO' if is_in_periods(datetime.date(int(row['YEAR']), int(row['MOTH']), 1), el_nino_periods)
    else 'LaNina' if is_in_periods(datetime.date(int(row['YEAR']), int(row['MOTH']), 1), la_nina_periods)
    else 'Normal', axis=1)

    if os.path.exists(output_csv):
        os.remove(output_csv)
    df.to_csv(output_csv, index=False)

def add_TS_column(file1, file2):
    # 读取csv表
    data = pd.read_csv(file1)

    # 判断'TS'列是否存在，存在则删除
    if 'TS' in data.columns:
        data = data.drop(columns=['TS'])

    # 计算'TS'列的值
    data['TS'] = data['TEMP'] * data['PSAL']

    # 将结果保存到新的csv文件
    data.to_csv(file2, index=False)

def move_files(source_folder, destination_folder):
    # 获取源文件夹中的所有文件
    files = os.listdir(source_folder)

    for file in files:
        # 拼接源文件的完整路径
        source_file_path = os.path.join(source_folder, file)

        # 拼接目标文件的完整路径
        destination_file_path = os.path.join(destination_folder, file)

        # 将文件从源文件夹剪切到目标文件夹
        shutil.move(source_file_path, destination_file_path)

if __name__ == '__main__':
    __spec__ = None
    # 数据库导出原始（有溶解氧）D:\Data\Argo_Data\dissolved_oxygen\result\alldoxy\200dbar\03\depth200_03zone.csv
    originalfile = "D:\\Data\\Argo_Data\\dissolved_oxygen\\rowcsv\\alldoxy.csv"
    NO_originalfile = "D:\\Data\\Argo_Data\\dissolved_oxygen\\rowcsv\\allnodoxy.csv"
    # 预处理结果按深度划分的数据（有溶解氧）前缀名
    Depthfilepre = "D:\\Data\\Argo_Data\\dissolved_oxygen\\result\\alldoxy\\"
    NO_Depthfilepre = "D:\\Data\\Argo_Data\\dissolved_oxygen\\result\\allnodoxy\\"

    wordmap = "D:\\Data\\ArcGIS_Data\\word\\太平洋中心\\world0_150W.shp"
    JHcoastline = "D:\\Data\\ArcGIS_Data\\word\\太平洋中心\\JHcoastline_150W.shp"
    maskfile = "D:\\Data\\ArcGIS_Data\word\\太平洋中心\\maskocean.shp"
    Chlafolder_path = "D:\\Data\\chla"

    #depthlist = ["10","100","200","1000","2000"]#"10","100","200","1000","2000"
    depthlist = ["10","20","30","40","50","75","100","125","150","200","250","300","400","500","600","700","800","900","1000","1100","1200","1300","1400","1750","1500","2000"]
    #"01","04","07","10"
    monthlist = ["01","02","03","04","05","06","07","08","09","10","11","12"] # "02","03","04","05","06","07","08","09","10","11","12"
    yearlist = ["2004","2005","2006","2007","2008","2009","2010","2011","2012","2013","2014","2015","2016","2017","2018","2019","2020","2021","2022"]#"2004","2005","2006","2007","2008","2009","2010","2011","2012","2013","2014","2015","2016","2017","2018","2019","2020","2021","2022"
    #yearlist = ["2022"]#"2004","2005","2006","2007","2008","2009","2010","2011","2012","2013","2014","2015","2016","2017","2018","2019","2020","2021","2022"

    print("...即将处理的深度层为:{0},即将处理的月份为:{1},共{2}个循环!".format(depthlist,monthlist,len(monthlist)*len(depthlist)))

    #调节备注
    remarknote = input_with_timeout("请输入备注信息：", 60)
    #remarknote = 'False, k: 200'#输入英文

    # 预处理
    #preprocess(originalfile, Depthfilepre, NO_Depthfilepre)
    #preprocess(NO_originalfile, Depthfilepre, NO_Depthfilepre)

    # 各深度层海底陆地掩膜
    '''mainseatif = "D:\\Data\\ArcGIS_Data\\word\\ETOPO_2022_v1_30s_N90W180_bed.tif"
    outmaskpath = "D:\\Dat`a\\ArcGIS_Data\\word\\太平洋中心\\seamask"
    for de in depthlist:
        seafile = outmaskpath + "\\sea" + de + ".shp"
        landfile = outmaskpath + "\\land" + de + ".shp"

        makeseamake(mainseatif,de,seafile,landfile)'''


    Predict_features = ['LONGITUDE', 'LATITUDE', 'TEMP', 'PSAL', 'YEAR']
    num_features = len(Predict_features)
    all_features = ['LONGITUDE', 'LATITUDE', 'YEAR', 'MOTH', 'TEMP', 'PSAL', 'DOXY']#,'EVENT'
    deletefeatures = ['zone', 'OID_', 'TEMP', 'PSAL']
    print("...模型预测因子为{}!".format(Predict_features))



    for de in depthlist:
        '''if de == '0':
            Predict_features = ['LONGITUDE', 'LATITUDE', 'TEMP', 'PSAL', 'CHLA']
            num_features = len(Predict_features)
            all_features = ['LONGITUDE', 'LATITUDE', 'YEAR', 'MOTH', 'TEMP', 'PSAL', 'DOXY','CHLA']
            deletefeatures = ['zone', 'OID_', 'TEMP', 'PSAL']'''

        seamaskfile = "D:\\Data\\ArcGIS_Data\\word\\太平洋中心\\seamask\\sea" + de + ".shp"
        for mo in monthlist:
            # 区域标识
            depth1 = de
            month1 = mo
            print("-----------------处理进度：深度为{0}，月份为{1}--------------------".format(depth1, month1))

            # 标准层筛选
            trafile = "D:\\Data\\Argo_Data\\dissolved_oxygen\\result\\alldoxy\\" + depth1 + "dbar\\" + month1 + "\\depth" + depth1 + "_" + month1 + ".csv"
            traoutfile = "D:\\Data\\Argo_Data\\dissolved_oxygen\\result\\alldoxy\\" + depth1 + "dbar\\" + month1 + "\\depth" + depth1 + "_" + month1 + "_.csv"
            prefile = "D:\\Data\\Argo_Data\\dissolved_oxygen\\result\\allnodoxy\\" + depth1 + "dbar\\" + month1 + "\\depth" + depth1 + "_" + month1 + ".csv"
            preoutfile = "D:\\Data\\Argo_Data\\dissolved_oxygen\\result\\allnodoxy\\" + depth1 + "dbar\\" + month1 + "\\depth" + depth1 + "_" + month1 + "_.csv"
            zonefile = "D:\\Data\\Argo_Data\\dissolved_oxygen\\result\\doxy\\" + depth1 + "dbar\\" + month1 + "\\depth" + depth1 + "_" + month1 + ".csv"
            zoneoutfile = "D:\\Data\\Argo_Data\\dissolved_oxygen\\result\\doxy\\" + depth1 + "dbar\\" + month1 + "\\depth" + depth1 + "_" + month1 + "_.csv"

            #arcpy
            env.workspace = "D:\\Data\\Argo_Data\\dissolved_oxygen\\result\\alldoxy\\" + depth1 + "dbar\\" + month1 #设置环境
            outWorkspace = "D:\\Data\\Argo_Data\\dissolved_oxygen\\result\\alldoxy\\" + depth1 + "dbar\\" + month1
            arcpy.env.overwriteOutput = True



            #xy转点(无后缀)
            out_point_file1 = "D:\\Data\\Argo_Data\\dissolved_oxygen\\result\\alldoxy\\" + depth1 + "dbar\\" + month1 + "\\depth" + depth1 + "_" + month1 + "_xy"
            out_point_file2 = "D:\\Data\\Argo_Data\\dissolved_oxygen\\result\\allnodoxy\\" + depth1 + "dbar\\" + month1 + "\\depth" + depth1 + "_" + month1 + "_xy"
            out_point_file0 = "D:\\Data\\Argo_Data\\dissolved_oxygen\\result\\doxy\\" + depth1 + "dbar\\" + month1 + "\\depth" + depth1 + "_" + month1 + "_xy"

            #点投影
            wkt3 = "GEOGCS['WGS 1984_3',DATUM['D_WGS_1984',SPHEROID['WGS_1984',6378137.0,298.257223563]],\
                      PRIMEM['Greenwich',-150.0],UNIT['Degree',0.0174532925199433]];\
                      -400 -400 1000000000;-100000 10000;-100000 10000;8.98315284119522E-09;\
                      0.001;0.001;IsHighPrecision"
            out_point_project1 = out_point_file1 + "_proj.shp"
            out_point_project2 = out_point_file2 + "_proj.shp"
            out_point_project0 = out_point_file0 + "_proj.shp"

            out_point_projecttrue1 = out_point_file1 + "_projtrue.shp"
            out_point_projecttrue2 = out_point_file2 + "_projtrue.shp"
            out_point_projecttrue0 = out_point_file0 + "_projtrue.shp"
            # 样条插值法
            Splineout = "D:\\Data\\Argo_Data\\dissolved_oxygen\\result\\doxy\\" + depth1 + "dbar\\" + month1 + "\\spline_climate.tif"
            Splineout_real = "D:\\Data\\Argo_Data\\dissolved_oxygen\\result\\doxy\\" + depth1 + "dbar\\" + month1 + "\\spline_climatereal.tif"
            # 等值面
            outContours = "D:\\Data\\Argo_Data\\dissolved_oxygen\\result\\doxy\\" + depth1 + "dbar\\" + month1 + "\\contourarea.shp"
            # 要素转栅格
            outContourgrid = "D:\\Data\\Argo_Data\\dissolved_oxygen\\result\\doxy\\" + depth1 + "dbar\\" + month1 + "\\contourgrid.tif"
            # 值提取到点
            outPointFeatures1 = "D:\\Data\\Argo_Data\\dissolved_oxygen\\result\\alldoxy\\" + depth1 + "dbar\\" + month1 + "\\depth" + depth1 + "_" + month1 + "_area.shp"
            PointFeaturestrue1 =  "D:\\Data\\Argo_Data\\dissolved_oxygen\\result\\alldoxy\\" + depth1 + "dbar\\" + month1 + "\\depth" + depth1 + "_" + month1 + "_areatrue.shp"

            outPointFeatures2 = "D:\\Data\\Argo_Data\\dissolved_oxygen\\result\\allnodoxy\\" + depth1 + "dbar\\" + month1 + "\\depth" + depth1 + "_" + month1 + "_area.shp"
            PointFeaturestrue2 =  "D:\\Data\\Argo_Data\\dissolved_oxygen\\result\\allnodoxy\\" + depth1 + "dbar\\" + month1 + "\\depth" + depth1 + "_" + month1 + "_areatrue.shp"
            # shp转csv
            outPointcsv1 = "D:\\Data\\Argo_Data\\dissolved_oxygen\\result\\alldoxy\\" + depth1 + "dbar\\" + month1
            talbename = "depth" + depth1 + "_" + month1 + "_all.csv"
            outPointcsv2 = "D:\\Data\\Argo_Data\\dissolved_oxygen\\result\\allnodoxy\\" + depth1 + "dbar\\" + month1
            #叶绿素
            # 训练数据（有溶解氧）
            trainfile = "D:\\Data\\Argo_Data\\dissolved_oxygen\\result\\alldoxy\\" + depth1 + "dbar\\" + month1 + "\\depth" + depth1 + "_" + month1 + "_all.csv"
            # Catboost回归模型存储
            CatBoostsave = "D:\\Data\\Argo_Data\\dissolved_oxygen\\result\\alldoxy\\" + depth1 + "dbar\\" + month1 + "\\depth" + depth1 + "_" + month1 + "_Cat"
            RFsave = "D:\\Data\\Argo_Data\\dissolved_oxygen\\result\\alldoxy\\" + depth1 + "dbar\\" + month1 + "\\depth" + depth1 + "_" + month1 + "_RF"
            SVRsave = "D:\\Data\\Argo_Data\\dissolved_oxygen\\result\\alldoxy\\" + depth1 + "dbar\\" + month1 + "\\depth" + depth1 + "_" + month1 + "_SVR"
            NNsave = "D:\\Data\\Argo_Data\\dissolved_oxygen\\result\\alldoxy\\" + depth1 + "dbar\\" + month1 + "\\depth" + depth1 + "_" + month1 + "_NN"

            # 带预测数据（无溶解氧）
            predictfile = "D:\\Data\\Argo_Data\\dissolved_oxygen\\result\\allnodoxy\\" + depth1 + "dbar\\" + month1 + "\\depth" + depth1 + "_" + month1 + "_all.csv"

            # 预测结果数据（有溶解氧预测值）
            predicted = "D:\\Data\\Argo_Data\\dissolved_oxygen\\result\\allnodoxy\\" + depth1 + "dbar\\" + month1 + "\\depth" + depth1 + "_" + month1 + "_pred.csv"
            # 预测结果插值
            out_point_file3 = "D:\\Data\\Argo_Data\\dissolved_oxygen\\result\\allnodoxy\\" + depth1 + "dbar\\" + month1 + "\\depth" + depth1 + "_" + month1 + "_area"
            out_point_project3 = out_point_file3 + "_proj.shp"
            out_point_projecttrue3 = out_point_file3 + "_projtrue.shp"

            #Splineout3 = r"D:\\Data\\Argo_Data\\dissolved_oxygen\\result\\allnodoxy\\" + depth1 + "dbar\\" + month1 + "\\spline.tif"
            #Splineout_real3 = r"D:\\Data\\Argo_Data\\dissolved_oxygen\\result\\allnodoxy\\" + depth1 + "dbar\\" + month1 + "\\spline_real.tif"
            #--------------------------------------------------------------------------------------

            # 标准层筛选
            '''Standardlayerfilter(trafile, traoutfile, ['TEMP', 'PSAL', 'DOXY'])#训练数据
            Standardlayerfilter(prefile, preoutfile, ['TEMP', 'PSAL'])#待预测数据
            Standardlayerfilter(zonefile, zoneoutfile, ['DOXY'])#分区数据'''

            #-----ArcpySTART-----
            # XY转点
            XY_point(traoutfile, out_point_file1)#训练数据
            XY_point(preoutfile, out_point_file2)#待预测数据
            XY_point(zoneoutfile, out_point_file0)#分区数据
            # 投影 1984 → 1984-3
            Project_1984_3(wkt3,out_point_file1,out_point_project1)#训练数据
            Project_1984_3(wkt3,out_point_file2,out_point_project2)#待预测数据
            Project_1984_3(wkt3,out_point_file0,out_point_project0)#分区数据
            # 删除在陆地的点
            delete_land_point(wordmap,out_point_project1,out_point_projecttrue1)#delete_land_point(wordmap,out_point_project2,out_point_projecttrue2)
            delete_land_point(wordmap,out_point_project0,out_point_projecttrue0)#delete_land_point(wordmap,out_point_project2,out_point_projecttrue2)
            # 读取 CSV 文件DOXY 属性的最大值
            dfmax = pd.read_csv(zoneoutfile)
            max_doxy = dfmax['DOXY'].max()
            min_doxy = dfmax['DOXY'].min()
            maxgrid = max_doxy + 10
            mingrid = max(min_doxy - 10, 0.01)
            # 含障碍的样条插值法并删除负值
            SplineandBarriers(out_point_projecttrue0, Splineout, Splineout_real, str(maxgrid), str(mingrid))

            #按掩膜提取
            maskextract(Splineout_real,maskfile,Splineout_real)

            #k均值聚类确定分区个数
            #kmeans = kmeansfun(CatBoostsave,Splineout_real)*2#！！！！！！！！！！！！！！！！！！！！！！！！！！
            kmeans = 10

            #min_value,max_value = tifmaxmin(Splineout_real)
            #contourInterval = [0,46.929546, 100.56255, 144.6962, 176.87785, 202.01434, 224.99207, 251.6839, 279.7301, 308.9094, 486.852]
            #baseContour = int(min_value) #输入起始等值线
            #sumcount = len(open(traoutfile).readlines())
            #k = max(200, int(sumcount / kmeansK))


            contourInterval = interbreak(kmeans,Splineout_real)
            #contourInterval = [0.01, 46.929546, 100.56255, 144.6962, 176.87785, 202.01434, 224.99207, 251.6839, 279.7301, 308.9094, 486.852]

            txtfile = CatBoostsave + ".txt"
            with open(txtfile, 'a') as f:  # 设置文件对象
                print("----------------------------------------------------------------------分割线----------------------------------------------------------------------", file = f)
                print("备注：{0}".format(remarknote), file = f)
                print("等值线间隔: %s" % contourInterval, file=f)
                print("最佳簇个数: %d" % kmeans, file=f)

            outContours = Contourbreaks(Raster(Splineout_real),contourInterval)
            # 要素转栅格
            Featogrid(outContours, outContourgrid)
            # 值提取到点
            Valuetopoint(out_point_projecttrue1, outContourgrid, outPointFeatures1,PointFeaturestrue1,"zone")
            Valuetopoint(out_point_project2, outContourgrid, outPointFeatures2,PointFeaturestrue2,"zone")

            # 提取到表
            Valuetocsv(PointFeaturestrue1, outPointcsv1, talbename)
            Valuetocsv(PointFeaturestrue2, outPointcsv2, talbename)

            k = 250 #经验法则
            txtfile = CatBoostsave + ".txt"
            with open(txtfile, 'a') as f:  # 设置文件对象
                print("分区合并阈值: %d" % k, file = f)
            #分区合并
            resultlist = merage_zone(trainfile,k,CatBoostsave)
            meragecsv(resultlist, trainfile)
            meragecsv(resultlist, predictfile)
            #-----ArcpyEND-----

            # 添加TS
            #add_TS_column(trainfile, trainfile)
            #add_TS_column(predictfile, predictfile)

            # 添加enso事件特征
            #add_event_column(trainfile, trainfile)
            #add_event_column(predictfile, predictfile)


            # 模型训练----------------------
            start = time.time()
            KJ_Catboost.splitzone(trainfile, CatBoostsave, depth1, month1,remarknote,Predict_features,num_features)#CatBoostMode(trainfile, CatBoostsave)
            #Nozone_CatBoost.splitzone(trainfile, CatBoostsave, depth1, month1,remarknote, Predict_features, num_features)
            #featureaszone_CatBoost.splitzone(trainfile, CatBoostsave, depth1, month1,remarknote,['LONGITUDE', 'LATITUDE', 'TEMP', 'PSAL','YEAR' ,'zone'],6)

            #RF.splitzone(trainfile, RFsave, depth1, month1,remarknote,Predict_features,num_features)#CatBoostMode(trainfile, CatBoostsave)
            #Nozone_RF.splitzone(trainfile, RFsave, depth1, month1,remarknote, Predict_features,num_features)+
            #featureasinput_RF.splitzone(trainfile, RFsave, depth1, month1,remarknote,['LONGITUDE', 'LATITUDE', 'TEMP', 'PSAL','YEAR' ,'zone'],6)

            #SVR.splitzone(trainfile, SVRsave, depth1, month1,remarknote,Predict_features,num_features)#CatBoostMode(trainfile, CatBoostsave)
            #Nozone_SVR.splitzone(trainfile, SVRsave, depth1, month1,remarknote, Predict_features, num_features)
            #featureasinput_SVR.splitzone(trainfile, SVRsave, depth1, month1,remarknote, ['LONGITUDE', 'LATITUDE', 'TEMP', 'PSAL','YEAR' ,'zone'],6)

            #NN.splitzone(trainfile, NNsave, depth1, month1,remarknote,Predict_features,num_features)#CatBoostMode(trainfile, CatBoostsave)

            end = time.time()
            with open(CatBoostsave + ".txt", 'a') as f:  # 设置文件对象
                print("---循环运行时间:%.2f秒---" % (end - start), file = f)
            # 训练结束----------------------


            # 模型预测----------------------
            # 最终catboost预测
            predict(zoneoutfile, deletefeatures, CatBoostsave, predictfile, predicted)#评估用traoutfile,数据集用zoneoutfile

            # catboost
            #predictSP(['OID_'], CatBoostsave, predictfile, predicted)
            #predictNSP(['OID_'], CatBoostsave, predictfile, predicted)
            #predictPIP(['OID_'], CatBoostsave, predictfile, predicted)

            # 随机森林回归
            #predictSP(['OID_'], RFsave, predictfile, predicted)
            #predictNSP(['OID_'], RFsave, predictfile, predicted)
            #predictPIP(['OID_'], RFsave, predictfile, predicted)

            # SVR
            #predictSP(['OID_'], SVRsave, predictfile, predicted)
            #predictNSP(['OID_'], SVRsave, predictfile, predicted)
            #predictPIP(['OID_'], SVRsave, predictfile, predicted)


            # 预测结果csv转点并投影
            XY_point(predicted, out_point_file3)
            Project_1984_3(wkt3, out_point_file3, out_point_project3)
            # 按年份划分
            for ye in yearlist:
                pointtemp = "D:\\Data\\Argo_Data\\dissolved_oxygen\\result\\allnodoxy\\" + depth1 + "dbar\\" + month1 +"\\" + ye +"_temp.shp"
                pointbyyear = "D:\\Data\\Argo_Data\\dissolved_oxygen\\result\\allnodoxy\\" + depth1 + "dbar\\" + month1 +"\\" + ye + "_" + month1 + "_" + depth1 + ".shp"
                splinetemp = "D:\\Data\\Argo_Data\\dissolved_oxygen\\result\\allnodoxy\\" + depth1 + "dbar\\" + month1 +"\\" + ye +"_temp.tif"
                splinebyyear = "D:\\Data\\Argo_Data\\dissolved_oxygen\\result\\allnodoxy\\" + depth1 + "dbar\\" + month1 +"\\" + ye + "_" + month1 + "_" + depth1 + ".tif"
                mediaout = "D:\\Data\\Argo_Data\\dissolved_oxygen\\result\\allnodoxy\\" + depth1 + "dbar\\" + month1 +"\\" + ye + "_" + month1 + "_" + depth1 + "m.tif"
                finalout = "D:\\Data\\Argo_Data\\dissolved_oxygen\\result\\tif\\" + "DOC_MTH_" + ye + "_M" + month1 + "_D" + str(depth1).zfill(4) + ".tif" #DOC_MTH_XXXX_MXX_DXXXX
                # 筛选
                Pickyear(ye,out_point_project3,pointtemp)
                # 删除路面点
                delete_land_point(wordmap,pointtemp,pointbyyear)#删除陆面点
                # 插值出图
                SplineandBarrier(pointbyyear, splinetemp, splinebyyear)
                try:
                    #按掩膜提取去除北冰洋
                    maskextract(splinebyyear,maskfile,mediaout)#mediaout
                    #按掩膜提取考虑地形
                    maskextract(mediaout,seamaskfile,finalout)#研制数据集采用
                except Exception as ex:
                    print(ex.args[0])
                del pointtemp
                del pointbyyear
                del splinetemp
                del splinebyyear
                del finalout
                gc.collect()
            print("标准层{0}深度{1}月份处理完成！".format(de, mo))
            print("--------------------------------------------------\n")

            # 预测结束----------------------

            # 模型评估----------------------
            '''selectmodel = "CatBoost"#!!!!!!!!!!!!!! CatBoost RF
            zonefile_a = "D:\\Data\\Argo_Data\\dissolved_oxygen\\result\\doxy\\" + depth1 + "dbar\\" + month1 + "\\depth" + depth1 + "_" + month1 + "_.csv"
            zonefile_b = "D:\\Data\\Argo_Data\\dissolved_oxygen\\result\\alldoxy\\" + depth1 + "dbar\\" + month1 + "\\depth" + depth1 + "_" + month1 + "_all.csv"
            zonefile_c = "D:\\Data\\Argo_Data\\dissolved_oxygen\\result\\doxy\\" + depth1 + "dbar\\" + month1 + "\\depth" + depth1 + "_" + month1 + "_c.csv"
            zonefile_d = "D:\\Data\\Argo_Data\\dissolved_oxygen\\result\\doxy\\" + depth1 + "dbar\\" + month1 + "\\depth" + depth1 + "_" + month1 + "_" + selectmodel + ".csv"
            zonefile_path = "D:\\Data\\Argo_Data\\dissolved_oxygen\\result\\Es_tif\\temp" + depth1 + "_" + month1 + "\\" + selectmodel

            #deletematchdata(zonefile_a, zonefile_b, zonefile_c)#删除匹配数据行
            #valuetonewfield(zonefile_c, zonefile_d, zonefile_path, 'DOC')#a:doxy_;b:alldoxy_all; c：删除之后的自己生成的 d:最终自己生成的
            '''
            # 评估结束----------------------

            # test2023
            preoutfile2023 = "D:\\Data\\Argo_Data\\dissolved_oxygen\\result\\alldoxy2023\\" + depth1 + "dbar\\" + month1 + "\\depth" + depth1 + "_" + month1 + "_.csv"
            out_point_file2023 = "D:\\Data\\Argo_Data\\dissolved_oxygen\\result\\alldoxy2023\\" + depth1 + "dbar\\" + month1 + "\\depth" + depth1 + "_" + month1 + "_xy"
            out_point_project2023 = out_point_file2023 + "_proj.shp"
            outPointFeatures2023 = "D:\\Data\\Argo_Data\\dissolved_oxygen\\result\\alldoxy2023\\" + depth1 + "dbar\\" + month1 + "\\depth" + depth1 + "_" + month1 + "_area.shp"
            PointFeaturestrue2023 =  "D:\\Data\\Argo_Data\\dissolved_oxygen\\result\\alldoxy2023\\" + depth1 + "dbar\\" + month1 + "\\depth" + depth1 + "_" + month1 + "_areatrue.shp"
            outPointcsv2023 = "D:\\Data\\Argo_Data\\dissolved_oxygen\\result\\alldoxy2023\\" + depth1 + "dbar\\" + month1
            predicted2023 = "D:\\Data\\Argo_Data\\dissolved_oxygen\\result\\alldoxy2023\\" + depth1 + "dbar\\" + month1 + "\\depth" + depth1 + "_" + month1 + "_pred"
            predictfile2023 = "D:\\Data\\Argo_Data\\dissolved_oxygen\\result\\alldoxy2023\\" + depth1 + "dbar\\" + month1 + "\\depth" + depth1 + "_" + month1 + "_all.csv"

            '''XY_point(preoutfile2023, out_point_file2023)#2023数据
            Project_1984_3(wkt3,out_point_file2023,out_point_project2023)#待预测数据
            Valuetopoint(out_point_project2023, outContourgrid, outPointFeatures2023,PointFeaturestrue2023,"zone")
            Valuetocsv(PointFeaturestrue2023, outPointcsv2023, talbename)#值提取到点
            resultlist = merage_zone(trainfile,120,CatBoostsave)#分区合并'''

            # catboost
            '''predictSP(['zone', 'OID_'], CatBoostsave, predictfile2023, predicted2023 + "cat_SP.csv")
            predictNSP(['zone', 'OID_'], CatBoostsave, predictfile2023, predicted2023 + "cat_NSP.csv")
            predictPIP(['zone', 'OID_'], CatBoostsave, predictfile2023, predicted2023 + "cat_PIP.csv")

            # 随机森林回归
            predictSP(['zone', 'OID_'], RFsave, predictfile2023, predicted2023 + "RF_SP.csv")
            predictNSP(['zone', 'OID_'], RFsave, predictfile2023, predicted2023 + "RF_NSP.csv")
            predictPIP(['zone', 'OID_'], RFsave, predictfile2023, predicted2023 + "RF_PIP.csv")

            # SVR
            predictSP(['zone', 'OID_'], SVRsave, predictfile2023, predicted2023 + "SVR_SP.csv")
            predictNSP(['zone', 'OID_'], SVRsave, predictfile2023, predicted2023 + "SVR_NSP.csv")
            predictPIP(['zone', 'OID_'], SVRsave, predictfile2023, predicted2023 + "SVR_PIP.csv")'''

            del depth1
            del month1
            del trafile
            del traoutfile
            del prefile
            del preoutfile
            del outWorkspace
            del out_point_file1
            del out_point_file2
            del wkt3
            del out_point_project1
            del out_point_project2
            del out_point_projecttrue1
            del out_point_projecttrue2
            del Splineout
            del Splineout_real
            del outContours
            del outContourgrid
            del outPointFeatures1
            del PointFeaturestrue1
            del outPointFeatures2
            del PointFeaturestrue2
            del outPointcsv1
            del talbename
            del outPointcsv2
            del trainfile
            del CatBoostsave
            del predictfile
            del predicted
            del out_point_file3
            del out_point_project3
            del out_point_projecttrue3
            gc.collect()

        print("!!!标准层{0}深度各个月份处理完成!!！".format(de))
        print("\n")








