# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 12:34:29 2019

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
 



m=0
data_or = []
with open('TADPOLE_D1_D2_del_del_del.csv', 'r', newline='') as csv_file:
    csv_writer = csv.reader(csv_file)
    for ll in csv_writer:
        if ll[0]=='RID':
            data_name=ll
            continue
        data_or.append(ll)
data_p=[]
for i in range(len(data_or)):
    data_p.append(data_or[i][0])
data_p=list(set(data_p))   

data_done_first=[]
for i in range(len(data_p)):
    bbb=[]
    for j in range(len(data_or)):
        if data_or[j][0]==data_p[i]:
            bbb.append(data_or[j])
    data_done_first.append([data_p[i],bbb])
data_num=['bl','m03','m06','m12','m18','m24','m30','m36','m42','m48','m54','m60','m66','m72','m78','m84','m90','m96','m102','m108','m114','m120']

data_done_second=[]
for i in range(len(data_done_first)):
    ccc=[]
    data_demo=data_done_first[i][1]
    for j in range(len(data_num)):
        f=0
        for k in range(len(data_demo)):
            if data_demo[k][2]==data_num[j]:
                ccc.append(data_demo[k])
                f=1
        if f==0:
            ccc.append(ccc[-1])
        
    data_done_second.append([data_done_first[i][0],ccc])                
       

data_done_third=[]
for i in range(len(data_done_second)):
    data_data=data_done_second[i][1]
    for j in range(len(data_data)-1):
        if '' in data_data[j] or ' ' in data_data[j]:
            continue
        for k in range(len(data_data[0])):
            if data_data[j+1][k]=='' or data_data[j+1][k]==' ':
                data_data[j+1][k]=data_data[j][k]
    data_done_third.append([data_done_second[i][0],data_data])

# for i in range(len(data_done_third)):
#     if data_done_third[i][0]=='23':
#         aa=data_done_third[i]
# mm=[]
# for i in range(len(data_done_third[61][1][9])):
#     if data_done_third[61][1][9]=='' or data_done_third[61][1][9]==' ':
#         mm.append([i,data_name[i]])
# mm=[] 
# poi=[]   
# for i in range(len(data_done_third)):
#     aa=data_done_third[i]
#     for j in range(len(aa[1])):
#         aaaa=aa[1][j]
#         mmmm=0
#         for k in range(len(aaaa)):
#             if aaaa[k]=='' or aaaa[k]==' ':
#                 mmmm=mmmm+1
#         mm.append(mmmm)
#         poi.append([i,j,aaaa])
#     print(i)

    
    
    
    
    
    
    
    
data_done_forth=[]            
for i in range(len(data_done_third)):
    ccc=[]
    data_data=data_done_third[i][1]
    for j in range(len(data_data)):
        if '' in data_data[j] or ' ' in data_data[j]:
            continue
        else:
            ccc.append(data_data[j])
    if len(ccc)<1:
        continue
    data_done_forth.append([len(ccc),data_done_third[i][0],ccc])    


Label=[]    
for i in range(len(data_done_forth)):
    aaa=[]
    for j in range(len(data_done_forth[i][2])):
        if data_done_forth[i][2][j][8]=='Male':
            data_done_forth[i][2][j][8]=1
        else:
            data_done_forth[i][2][j][8]=0

        if data_done_forth[i][2][j][10]=='Not Hisp/Latino':
            data_done_forth[i][2][j][10]=0
        else:
            data_done_forth[i][2][j][10]=1

        if data_done_forth[i][2][j][11]=='White':
            data_done_forth[i][2][j][11]=1
        elif data_done_forth[i][2][j][11]=='More than one':
            data_done_forth[i][2][j][11]=0.75
        elif data_done_forth[i][2][j][11]=='Asian':
            data_done_forth[i][2][j][11]=0.25
        else:
            data_done_forth[i][2][j][11]=0

        if data_done_forth[i][2][j][12]=='Married':
            data_done_forth[i][2][j][12]=1
        elif data_done_forth[i][2][j][12]=='Widowed':
            data_done_forth[i][2][j][12]=0.5
        else:
            data_done_forth[i][2][j][12]=0      
        aaa.append(data_done_forth[i][2][j][-4])
        data_done_forth[i][2][j]=data_done_forth[i][2][j][7:-4]
        
    Label.append(aaa)
data_name=data_name[7:-4]        


feature_max_min=[]    
for i in range(len(data_done_forth[0][2][0])):
    data_demo=[]
    for j in range(len(data_done_forth)):
        for k in range(len(data_done_forth[j][2])):
            data_demo.append(float(data_done_forth[j][2][k][i])) 
    feature_max_min.append([min(data_demo),max(data_demo)])
for i in range(len(data_done_forth)):
    for j in range(len(data_done_forth[i][2])):
        for k in range(len(data_done_forth[i][2][0])):
            data_done_forth[i][2][j][k]=(float(data_done_forth[i][2][j][k])-feature_max_min[k][0])/(feature_max_min[k][1]-feature_max_min[k][0])



