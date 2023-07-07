# -*- coding: utf-8 -*-
"""
Created on Sun May  8 13:39:05 2022

@author: Administrator
"""

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


random.shuffle(cluse_data_name0)
random.shuffle(cluse_data_name1)
random.shuffle(cluse_data_name2)
random.shuffle(cluse_data_name3)





target_data=[]
target_label=[]
background_data=[]
data_lengh=1
#life=[0,4,8,12]

time_dely=10 #六个月一个步长

LLabel=[]
for i in range(len(Label)):
    aaa=[]
    for j in range(len(Label[i])):
        if Label[i][j]=='NL':
            aaa.append(0)
        if Label[i][j]=='NL to MCI':
            aaa.append(0.5)
        if Label[i][j]=='MCI':
            aaa.append(1)
        if Label[i][j]=='MCI to Dementia':
            aaa.append(1.5)
        if Label[i][j]=='NL to Dementia':
            aaa.append(1.75)
        if Label[i][j]=='Dementia':
            aaa.append(2)
        if Label[i][j]=='MCI to NL':
            aaa.append(-1)
        if Label[i][j]=='Dementia to MCI':
            aaa.append(-1)
    LLabel.append(aaa)
            
        
        
good_change=[]
bad_change=[]
no_change=[]   
good_change_name=[]
bad_change_name=[]
no_change_name=[]      


for i in range(len(data_done_forth)):
    if data_done_forth[i][0]<data_lengh:
        continue
    for j in range(data_done_forth[i][0]-time_dely-data_lengh):
        if LLabel[i][j+data_lengh+time_dely-1] - LLabel[i][j+data_lengh-1]>0:
            aa=data_done_forth[i][2][j:j+data_lengh]
#            target_data.append(aa)
#            target_label.append(-1)
            bad_change.append(aa)
            bad_change_name.append(data_done_forth[i][1])
        if LLabel[i][j+data_lengh+time_dely-1] - LLabel[i][j+data_lengh-1]<0:
            aa=data_done_forth[i][2][j:j+data_lengh]
#            target_data.append(aa)
#            target_label.append(1)
            good_change.append(aa)
            good_change_name.append(data_done_forth[i][1])
        if LLabel[i][j+data_lengh+time_dely-1] - LLabel[i][j+data_lengh-1]==0 and LLabel[i][j+data_lengh+time_dely-1]!=0 and LLabel[i][j+data_lengh-1]!=0:
            aa=data_done_forth[i][2][j:j+data_lengh]
#            target_data.append(aa)
#            target_label.append(0)
            no_change.append(aa)
            no_change_name.append(data_done_forth[i][1])
        if sum(LLabel[i][j:j+data_lengh+time_dely])==0:
            aa=data_done_forth[i][2][j:j+data_lengh]
            background_data.append(aa)
#            target_label.append(0)
        
#hee = len(bad_change) + len(good_change)
#zero_choose = random.sample(range(len(no_change)),hee)
data_change_name=[]
for i in range(len(bad_change)):
    target_data.append(bad_change[i])
    target_label.append(-1)
    data_change_name.append(bad_change_name[i])
for i in range(len(good_change)):   
    target_data.append(good_change[i])
    target_label.append(1)     
    data_change_name.append(good_change_name[i])
for i in range(len(no_change)):   
    target_data.append(no_change[i])
    target_label.append(0) 
    data_change_name.append(no_change_name[i])



data_data_name=[]
TTTT=[]
TTTL=[]
TTTN=[]
abc=list(range(len(target_label)))
random.shuffle(abc)
for i in range(len(abc)):
    TTTT.append(target_data[abc[i]])
    TTTL.append(target_label[abc[i]])
    TTTN.append(data_change_name[abc[i]])

target_data=TTTT
target_label=TTTL
data_data_name=TTTN

or_target_data=target_data
or_target_label=target_label




    
##diyilei

dddddata=[]
lllllabel=[]
for i in range(len(cluse_data_name0)):
    d_name=cluse_data_name0[i]
    for j in range(len(data_data_name)):
        if data_data_name[j]==d_name:
            dddddata.append(target_data[j][0])
            lllllabel.append(target_label[j])


random_forest_model=RandomForestRegressor(n_estimators=200)
random_forest_model.fit(dddddata,lllllabel)
random_forest_importance=list(random_forest_model.feature_importances_)

max_number0=[]
importance_num=10
max_number00 = heapq.nlargest(importance_num, random_forest_importance) 
RF_max_index = map(random_forest_importance.index, heapq.nlargest(importance_num, random_forest_importance)) 
indx=list(set(RF_max_index))
for i in range(len(indx)):
    max_number0.append(random_forest_importance[indx[i]])

choose_feature_name0=[]
for i in range(len(indx)):
    choose_feature_name0.append(data_name[indx[i]])


##diyilei

dddddata=[]
lllllabel=[]
for i in range(len(cluse_data_name1)):
    d_name=cluse_data_name1[i]
    for j in range(len(data_data_name)):
        if data_data_name[j]==d_name:
            dddddata.append(target_data[j][0])
            lllllabel.append(target_label[j])
            
random_forest_model=RandomForestRegressor(n_estimators=200)
random_forest_model.fit(dddddata,lllllabel)
random_forest_importance=list(random_forest_model.feature_importances_)

max_number1=[]
importance_num=10
max_number11 = heapq.nlargest(importance_num, random_forest_importance) 
RF_max_index = map(random_forest_importance.index, heapq.nlargest(importance_num, random_forest_importance)) 
indx=list(set(RF_max_index))
for i in range(len(indx)):
    max_number1.append(random_forest_importance[indx[i]])
    

choose_feature_name1=[]
for i in range(len(indx)):
    choose_feature_name1.append(data_name[indx[i]])



##diyilei

dddddata=[]
lllllabel=[]
for i in range(len(cluse_data_name2)):
    d_name=cluse_data_name2[i]
    for j in range(len(data_data_name)):
        if data_data_name[j]==d_name:
            dddddata.append(target_data[j][0])
            lllllabel.append(target_label[j])

random_forest_model=RandomForestRegressor(n_estimators=200)
random_forest_model.fit(dddddata,lllllabel)
random_forest_importance=list(random_forest_model.feature_importances_)

max_number2=[]
importance_num=10
max_number22 = heapq.nlargest(importance_num, random_forest_importance) 
RF_max_index = map(random_forest_importance.index, heapq.nlargest(importance_num, random_forest_importance)) 
indx=list(set(RF_max_index))
for i in range(len(indx)):
    max_number2.append(random_forest_importance[indx[i]])

choose_feature_name2=[]
for i in range(len(indx)):
    choose_feature_name2.append(data_name[indx[i]])






##diyilei

dddddata=[]
lllllabel=[]
for i in range(len(cluse_data_name3)):
    d_name=cluse_data_name3[i]
    for j in range(len(data_data_name)):
        if data_data_name[j]==d_name:
            dddddata.append(target_data[j][0])
            lllllabel.append(target_label[j])

random_forest_model=RandomForestRegressor(n_estimators=200)
random_forest_model.fit(dddddata,lllllabel)
random_forest_importance=list(random_forest_model.feature_importances_)

max_number3=[]
importance_num=10
max_number33 = heapq.nlargest(importance_num, random_forest_importance) 
RF_max_index = map(random_forest_importance.index, heapq.nlargest(importance_num, random_forest_importance)) 
indx=list(set(RF_max_index))
for i in range(len(indx)):
    max_number3.append(random_forest_importance[indx[i]])

choose_feature_name3=[]
for i in range(len(indx)):
    choose_feature_name3.append(data_name[indx[i]])





