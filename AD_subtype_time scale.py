# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 22:41:06 2022

@author: Administrator
"""
import copy
import csv
from sklearn import preprocessing
import numpy as np
import heapq
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
import mpl_toolkits.mplot3d
import random
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier ,  AdaBoostClassifier,ExtraTreesClassifier
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import NuSVC , SVC
from sklearn import linear_model
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix, precision_score, accuracy_score,recall_score, f1_score,roc_auc_score
from sklearn.model_selection import KFold

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cluster import SpectralClustering

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestRegressor

from scipy.spatial.distance import pdist,squareform
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.cluster import AffinityPropagation
from time import time



choose_feature_name_cluse0=['ST2SV_UCSFFSX_11_02_15_UCSFFSX51_08_01_16']

choose_feature_name_cluse1=['ST14TS_UCSFFSX_11_02_15_UCSFFSX51_08_01_16']

choose_feature_name_cluse2=['ST60CV_UCSFFSX_11_02_15_UCSFFSX51_08_01_16']

choose_feature_name_cluse3=['ST24TS_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16']


for k in range(len(choose_feature_name_cluse3)):
    NName=choose_feature_name_cluse3[k]
     
    data_or0=[]
 
    for i in range(len(data_done_first)):
        if data_done_first[i][0] in cluse_data_name3:
            data_or0.append(data_done_first[i])

    
    factor_name=NName
    name_index = data_name_haha.index(factor_name)
    
    data_time=['bl','m06','m12','m18','m24','m30','m36','m42','m48','m54','m60','m66','m72','m78','m84','m90','m96','m102','m108','m114','m120']
    
    data_len=len(data_time)
    
    data_data0=[]

    
    for i in range(len(data_or0)):
        demo_data=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        data_data=data_or0[i][1]
        for j in range(len(data_data)):
            if data_data[j][2]=='m03':
                continue
            
            dd_index = data_time.index(data_data[j][2])
            if data_data[j][name_index]==' ' or data_data[j][name_index]=='':
                continue
            demo_data[dd_index]= float(data_data[j][name_index])
        data_data0.append(demo_data)
    
    
    DDDD=[['Time']]
    data_avg=[]
    for i in range(len(demo_data)):
        data_num_count=0
        data_full=0
        
        dd0=[data_time[i]]
        for j in range(len(data_data0)):
            if data_data0[j][i]!=0:
                data_num_count=data_num_count+1
                data_full=data_full+data_data0[j][i]
                dd0.append(data_data0[j][i])
            else:
                dd0.append('')

        if data_num_count==0:
            dd0.append('')
            data_avg.append(0)
        else:
            dd0.append(data_full/data_num_count)
            data_avg.append(data_full/data_num_count)


        DDDD.append(dd0)
    

    with open('cluse4_top_Trend'+'/'+'cluse4_'+factor_name+'.csv', 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        for ll in DDDD:
            csv_writer.writerow(ll)    
    
                


       
    plt.figure(figsize=(12,5)) 
    for i in range(len(data_data0)):
        
        x=[]
        y=[]
        for j in range(len(data_data0[i])):
            if data_data0[i][j]!=0:
                x.append(j)
                y.append(data_data0[i][j])
    
            plt.plot(x,y,"o-",color='grey')
    
    
    x=[]
    y=[]
    for j in range(len(data_avg)):
        if data_avg[j]!=0:
            x.append(j)
            y.append(data_avg[j])

    plt.plot(x,y,"o-",color='red')
    
    plt.title('The trends of first cluster '+'('+factor_name+')')
    
    plt.xticks(range(len(data_time)),data_time)
#    plt.bar(range(len(data_time)), data_time,tick_label=data_time)
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.0, 1), loc=1, borderaxespad=0.)
    plt.savefig('cluse4_top_Trend'+'/'+NName+'4'+'.jpg')
    
    
    









