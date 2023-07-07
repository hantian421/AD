# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 19:13:39 2022

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
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestRegressor

from scipy.spatial.distance import pdist,squareform
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.cluster import AffinityPropagation


#data_or = []
#with open('TADPOLE_D1_D2_del.csv', 'r', newline='') as csv_file:
#    csv_writer = csv.reader(csv_file)
#    for ll in csv_writer:
#        if ll[0]=='RID':
#            data_name_haha=ll
#            continue
#        data_or.append(ll)
#data_p=[]
#for i in range(len(data_or)):
#    data_p.append(data_or[i][0])
#data_p=list(set(data_p))   
#
#data_done_first=[]
#for i in range(len(data_p)):
#    bbb=[]
#    for j in range(len(data_or)):
#        if data_or[j][0]==data_p[i]:
#            bbb.append(data_or[j])
#    data_done_first.append([data_p[i],bbb])



    
#target_data=[]
#target_label=[]
#background_data=[]
#data_change_name=[]
#data_lengh=8
#
#for i in range(len(data_done_forth)):
#    if data_done_forth[i][0]<data_lengh:
#        continue
#    if 'NL to MCI' in Label[i]: 
#        for j in range(len(data_done_forth[i][2])-data_lengh):
#            aa=data_done_forth[i][2][j:j+data_lengh]
#            target_data.append(aa)
#            data_change_name.append(data_done_forth[i][1])
#            target_label.append(-1)
#
#    if 'MCI to Dementia' in Label[i]: 
#        for j in range(len(data_done_forth[i][2])-data_lengh):
#            aa=data_done_forth[i][2][j:j+data_lengh]
#            target_data.append(aa)
#            data_change_name.append(data_done_forth[i][1])
#            target_label.append(-1)   
#    if 'MCI to NL' in Label[i]: 
#        for j in range(len(data_done_forth[i][2])-data_lengh):
#            aa=data_done_forth[i][2][j:j+data_lengh]
#            target_data.append(aa)
#            data_change_name.append(data_done_forth[i][1])
#            target_label.append(1)
#    if 'Dementia to MCI' in Label[i]: 
#        for j in range(len(data_done_forth[i][2])-data_lengh):
#            aa=data_done_forth[i][2][j:j+data_lengh]
#            target_data.append(aa)
#            data_change_name.append(data_done_forth[i][1])
#            target_label.append(1)
#            
#    if 'NL to MCI' not in Label[i] and 'MCI to Dementia' not in Label[i] and 'MCI to NL' not in Label[i] and 'Dementia to MCI' not in Label[i] and 'MCI' in Label[i]: 
#        for j in range(len(data_done_forth[i][2])-data_lengh):
#            aa=data_done_forth[i][2][j:j+data_lengh]
#            target_data.append(aa)
#            data_change_name.append(data_done_forth[i][1])
#            target_label.append(0)
#    if 'NL to MCI' not in Label[i] and 'MCI to Dementia' not in Label[i] and 'MCI to NL' not in Label[i] and 'Dementia to MCI' not in Label[i] and 'Dementia' in Label[i]: 
#        for j in range(len(data_done_forth[i][2])-data_lengh):
#            aa=data_done_forth[i][2][j:j+data_lengh]
#            target_data.append(aa)
#            data_change_name.append(data_done_forth[i][1])
#            target_label.append(0)
#    if 'NL to MCI' not in Label[i] and 'MCI to Dementia' not in Label[i] and 'MCI to NL' not in Label[i] and 'Dementia to MCI' not in Label[i] and 'NL' in Label[i]: 
#        for j in range(len(data_done_forth[i][2])-data_lengh):
#            aa=data_done_forth[i][2][j:j+data_lengh]
#            background_data.append(aa)


data_change_name=[]
target_data=[]
target_label=[]
background_data=[]
data_lengh=8
#life=[0,4,8,12]

time_dely=2 #六个月一个步长

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
data_change_name=TTTN

or_target_data=target_data
or_target_label=target_label





##数据处理已完毕_下面开始pca
#CPCA
for i in range(len(target_data)):
    target_data[i]=np.array(target_data[i])
for i in range(len(background_data)):
    background_data[i]=np.array(background_data[i])
#特征目标数据PCA
data_average=sum(target_data)/len(target_data)
G=np.zeros((target_data[0].shape[1],target_data[0].shape[1]))
for i in range(len(target_data)):
    G=G+np.matmul((target_data[i].T-data_average.T),(target_data[i]-data_average))
G_feature_target=G/len(target_data)
Target_eig,Target_vec = np.linalg.eig(G_feature_target)
#特征背景数据PCA
data_average=sum(background_data)/len(background_data)
G=np.zeros((background_data[0].shape[1],background_data[0].shape[1]))
for i in range(len(background_data)):
    G=G+np.matmul((background_data[i].T-data_average.T),(background_data[i]-data_average))
G_feature_background=G/len(background_data)
background_eig,background_vec=np.linalg.eig(G_feature_background)


#时间目标数据PCA
data_average=sum(target_data)/len(target_data)
G=np.zeros((target_data[0].shape[0],target_data[0].shape[0]))
for i in range(len(target_data)):
    G=G+np.matmul((target_data[i]-data_average),(target_data[i].T-data_average.T))
G_time_target=G/len(target_data)
Target_eig,Target_vec = np.linalg.eig(G_time_target)
#时间背景数据PCA
data_average=sum(background_data)/len(background_data)
G=np.zeros((background_data[0].shape[0],background_data[0].shape[0]))
for i in range(len(background_data)):
    G=G+np.matmul((background_data[i]-data_average),(background_data[i].T-data_average.T))
G_time_background=G/len(background_data)
background_eig,background_vec=np.linalg.eig(G_time_background)




alpha_feature = 0.344209915319041#0.3442#0.5363071422980711#0.5364999459959141#131#205.2299
alpha_time = 0.340293381551894#0.3403#1.189518403928483#1.189582328741842#131
down_D_rate=0.85



# C_PCA_feature
changed_Target=G_feature_target-alpha_feature*G_feature_background
Changed_eig,Changed_vec = np.linalg.eig(changed_Target)
He=sum(Changed_eig)
for i in range(len(Changed_eig)-1):
    AAA=heapq.nlargest(i, list(Changed_eig))
    if sum(AAA)/He > down_D_rate:
        down_D=i
        break
max_index = map(list(Changed_eig).index, heapq.nlargest(down_D, list(Changed_eig))) 
indx=list(set(max_index))
change_feature_metrix=[]
cpca_changed_target_data_feature=[]
for i in range(len(indx)):
    change_feature_metrix.append(Changed_vec[:,indx[i]])
change_feature_metrix=np.array(change_feature_metrix)
for i in range(len(target_data)):
    cpca_changed_target_data_feature.append(np.matmul(target_data[i],change_feature_metrix.T))



# C_PCA_time
changed_Target=G_time_target-alpha_time*G_time_background
Changed_eig,Changed_vec = np.linalg.eig(changed_Target)
He=sum(Changed_eig)
for i in range(len(Changed_eig)-1):
    AAA=heapq.nlargest(i, list(Changed_eig))
    if sum(AAA)/He > down_D_rate:
        down_D=i
        break
max_index = map(list(Changed_eig).index, heapq.nlargest(down_D, list(Changed_eig))) 
indx=list(set(max_index))
change_time_metrix=[]
cpca_changed_target_data_time=[]
for i in range(len(indx)):
    change_time_metrix.append(Changed_vec[:,indx[i]])
change_time_metrix=np.array(change_time_metrix)
for i in range(len(target_data)):
    cpca_changed_target_data_time.append(np.matmul(change_time_metrix,target_data[i]))


# C_PCA_time_feature
cpca_changed_target_data_feature_time=[]
for i in range(len(target_data)):
    ddddd=np.matmul(change_time_metrix,target_data[i])
    ddddd=np.matmul(ddddd,change_feature_metrix.T)
    cpca_changed_target_data_feature_time.append(ddddd)




data_new=[]
for i in range(len(cpca_changed_target_data_feature_time)):
    data0=cpca_changed_target_data_feature_time[i]
    data1=[]
    for j in range(len(data0)):
        data1=data1+list(data0[j])
    data_new.append(data1)

PPP=[]
for kk in [4]:
    
    dbscan = AgglomerativeClustering(n_clusters=kk,linkage='complete').fit(data_new)#SpectralClustering(n_clusters=3).fit_predict(data_new)#AgglomerativeClustering(n_clusters=3).fit(data_new)#DBSCAN(eps=10.5, min_samples=20).fit(data_new)
    #estimator = KMeans(n_clusters=4)  # 构造聚类器  # 聚类
    label_pred = dbscan.labels_  # 获取聚类标签
    label_pred=list(label_pred)
    
    n_clusters_ = len(set(label_pred)) - (1 if -1 in label_pred else 0)
    
    
    print(n_clusters_)
    
    
    cluse_data_name0=[]
    cluse_data_name1=[]
    cluse_data_name2=[]
    cluse_data_name3=[]
    cluse_data_name_nan=[]
    
    
    for i in range(len(label_pred)):
        if label_pred[i]==0:
            cluse_data_name0.append(data_change_name[i])
        if label_pred[i]==1:
            cluse_data_name1.append(data_change_name[i])
        if label_pred[i]==2:
            cluse_data_name2.append(data_change_name[i])
        if label_pred[i]==3:
            cluse_data_name3.append(data_change_name[i])
        if label_pred[i]==-1:
            cluse_data_name_nan.append(data_change_name[i])
            
        
    cluse_data_name0=list(set(cluse_data_name0))  
    cluse_data_name1=list(set(cluse_data_name1))  
    cluse_data_name2=list(set(cluse_data_name2))
    cluse_data_name3=list(set(cluse_data_name3)) 
    cluse_data_name_nan=list(set(cluse_data_name_nan))  
    
    
    p_metrix=metrics.silhouette_score(data_new, label_pred, metric='correlation')
    PPP.append(p_metrix)
    print(p_metrix)
    print('_______________')


#with open('silhouette_score.csv', 'w', newline='') as csv_file:
#    csv_writer = csv.writer(csv_file)
#    for ll in PPP:
#        csv_writer.writerow([ll])    

#print(len(cluse_data_name0))
#print(len(cluse_data_name1))
#print(len(cluse_data_name2))
#print(len(cluse_data_name_nan))
#
#
#DData=[]
#LLabel=[]
#dataaa_name=[]
#for i in range(len(data_done_forth)):
#    dataaa_name.append(data_done_forth[i][1])
#    
#for i in range(len(cluse_data_name0)):
#    cluse_data_name=cluse_data_name0[i]
#    name_index=dataaa_name.index(cluse_data_name)
#    for j in range(len(data_done_forth[name_index][2])):
#        DData.append(data_done_forth[name_index][2][j])
#        LLabel.append(0)
#        
#for i in range(len(cluse_data_name1)):
#    cluse_data_name=cluse_data_name0[i]
#    name_index=dataaa_name.index(cluse_data_name)
#    for j in range(len(data_done_forth[name_index][2])):
#        DData.append(data_done_forth[name_index][2][j])
#        LLabel.append(1)        
#
#for i in range(len(cluse_data_name2)):
#    cluse_data_name=cluse_data_name0[i]
#    name_index=dataaa_name.index(cluse_data_name)
#    for j in range(len(data_done_forth[name_index][2])):
#        DData.append(data_done_forth[name_index][2][j])
#        LLabel.append(2)   
#
#
#
#random_forest_model=RandomForestRegressor(n_estimators=200)
#random_forest_model.fit(DData,LLabel)
#random_forest_importance=list(random_forest_model.feature_importances_)
#
#importance_num=10
#max_number = heapq.nlargest(importance_num, random_forest_importance) 
#RF_max_index = map(random_forest_importance.index, heapq.nlargest(importance_num, random_forest_importance)) 
#indx=list(set(RF_max_index))
#
#choose_feature_name=[]
#for i in range(len(indx)):
#    choose_feature_name.append(data_name[indx[i]])





#for k in range(len(choose_feature_name)):
#    NName=choose_feature_name[k]
#    
#    
#    data_or0=[]
#    data_or1=[]
#    data_or2=[]
#    
#    
#    for i in range(len(data_done_first)):
#        if data_done_first[i][0] in cluse_data_name0:
#            data_or0.append(data_done_first[i])
#        if data_done_first[i][0] in cluse_data_name1:
#            data_or1.append(data_done_first[i])
#        if data_done_first[i][0] in cluse_data_name2:
#            data_or2.append(data_done_first[i])
#    
#    factor_name=NName
#    name_index = data_name_haha.index(factor_name)
#    
#    data_time=['bl','m03','m06','m12','m18','m24','m30','m36','m42','m48','m54','m60','m66','m72','m78','m84','m90','m96','m102','m108','m114','m120']
#    
#    data_len=len(data_time)
#    
#    data_data0=[]
#    data_data1=[]
#    data_data2=[]
#    
#    for i in range(len(data_or0)):
#        demo_data=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
#        data_data=data_or0[i][1]
#        for j in range(len(data_data)):
#            dd_index = data_time.index(data_data[j][2])
#            if data_data[j][name_index]==' ' or data_data[j][name_index]=='':
#                continue
#            demo_data[dd_index]= float(data_data[j][name_index])
#        data_data0.append(demo_data)
#    
#    
#    
#    for i in range(len(data_or1)):
#        demo_data=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
#        data_data=data_or1[i][1]
#        for j in range(len(data_data)):
#            dd_index = data_time.index(data_data[j][2])
#            if data_data[j][name_index]==' ' or data_data[j][name_index]=='':
#                continue
#            demo_data[dd_index]= float(data_data[j][name_index])
#        data_data1.append(demo_data)
#    
#    
#    
#    for i in range(len(data_or2)):
#        demo_data=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
#        data_data=data_or2[i][1]
#        for j in range(len(data_data)):
#            dd_index = data_time.index(data_data[j][2])
#            if data_data[j][name_index]==' ' or data_data[j][name_index]=='':
#                continue
#            demo_data[dd_index]= float(data_data[j][name_index])
#        data_data2.append(demo_data)
#    
#    
#    
#    
#    plt.figure(figsize=(5,5)) 
#    for i in range(len(data_data0)):
#        
#        x=[]
#        y=[]
#        for j in range(len(data_data0[i])):
#            if data_data0[i][j]!=0:
#                x.append(j)
#                y.append(data_data0[i][j])
#    
#            plt.plot(x,y,"o-")
#    plt.title('The Trends of first cluster '+'('+factor_name+')')
#    plt.grid(True)
#    plt.legend(bbox_to_anchor=(1.0, 1), loc=1, borderaxespad=0.)
#    plt.savefig('cluster_compare'+'/'+NName+'_cluster0'+'.jpg')
#    
#    
#    
#    plt.figure(figsize=(5,5)) 
#    for i in range(len(data_data1)):
#        x=[]
#        y=[]
#        for j in range(len(data_data1[i])):
#            if data_data1[i][j]!=0:
#                x.append(j)
#                y.append(data_data1[i][j])
#    
#            plt.plot(x,y,"o-")
#    plt.title('The Trends of second cluster '+'('+factor_name+')')
#    plt.grid(True)
#    plt.legend(bbox_to_anchor=(1.0, 1), loc=1, borderaxespad=0.)
#    plt.savefig('cluster_compare'+'/'+NName+'_cluster1'+'.jpg')
#    
#    
#    
#    
#    
#    plt.figure(figsize=(5,5))
#    for i in range(len(data_data2)):
#    
#        x=[]
#        y=[]
#        for j in range(len(data_data2[i])):
#            if data_data2[i][j]!=0:
#                x.append(j)
#                y.append(data_data2[i][j])
#    
#            plt.plot(x,y,"o-")
#    plt.title('The Trends of third cluster '+'('+factor_name+')')
#    plt.grid(True)
#    plt.legend(bbox_to_anchor=(1.0, 1), loc=1, borderaxespad=0.)
#    plt.savefig('cluster_compare'+'/'+NName+'_cluster2'+'.jpg')
#    
#    print(k)











#data_or0=[]
#data_or1=[]
#data_or2=[]
#
#
#for i in range(len(data_done_first)):
#    if data_done_first[i][0] in cluse_data_name0:
#        data_or0.append(data_done_first[i])
#    if data_done_first[i][0] in cluse_data_name1:
#        data_or1.append(data_done_first[i])
#    if data_done_first[i][0] in cluse_data_name2:
#        data_or2.append(data_done_first[i])
#
#factor_name='ADAS11'
#name_index = data_name_haha.index(factor_name)
#
#data_time=['bl','m03','m06','m12','m18','m24','m30','m36','m42','m48','m54','m60','m66','m72','m78','m84','m90','m96','m102','m108','m114','m120']
#
#data_len=len(data_time)
#
#data_data0=[]
#data_data1=[]
#data_data2=[]
#
#for i in range(len(data_or0)):
#    demo_data=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
#    data_data=data_or0[i][1]
#    for j in range(len(data_data)):
#        dd_index = data_time.index(data_data[j][2])
#        if data_data[j][name_index]==' ' or data_data[j][name_index]=='':
#            continue
#        demo_data[dd_index]= float(data_data[j][name_index])
#    data_data0.append(demo_data)
#
#
#
#for i in range(len(data_or1)):
#    demo_data=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
#    data_data=data_or1[i][1]
#    for j in range(len(data_data)):
#        dd_index = data_time.index(data_data[j][2])
#        if data_data[j][name_index]==' ' or data_data[j][name_index]=='':
#            continue
#        demo_data[dd_index]= float(data_data[j][name_index])
#    data_data1.append(demo_data)
#
#
#
#for i in range(len(data_or2)):
#    demo_data=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
#    data_data=data_or2[i][1]
#    for j in range(len(data_data)):
#        dd_index = data_time.index(data_data[j][2])
#        if data_data[j][name_index]==' ' or data_data[j][name_index]=='':
#            continue
#        demo_data[dd_index]= float(data_data[j][name_index])
#    data_data2.append(demo_data)
#
#
#
#
#plt.figure(figsize=(5,5)) 
#for i in range(len(data_data0)):
#    
#    x=[]
#    y=[]
#    for j in range(len(data_data0[i])):
#        if data_data0[i][j]!=0:
#            x.append(j)
#            y.append(data_data0[i][j])
#
#        plt.plot(x,y,"o-")
#plt.title('The Trends of first cluster '+'('+factor_name+')')
#plt.grid(True)
#plt.legend(bbox_to_anchor=(1.0, 1), loc=1, borderaxespad=0.)
#
#
#
#plt.figure(figsize=(5,5)) 
#for i in range(len(data_data1)):
#    x=[]
#    y=[]
#    for j in range(len(data_data1[i])):
#        if data_data1[i][j]!=0:
#            x.append(j)
#            y.append(data_data1[i][j])
#
#        plt.plot(x,y,"o-")
#plt.title('The Trends of second cluster '+'('+factor_name+')')
#plt.grid(True)
#plt.legend(bbox_to_anchor=(1.0, 1), loc=1, borderaxespad=0.)
#
#
#
#
#
#plt.figure(figsize=(5,5))
#for i in range(len(data_data2)):
#
#    x=[]
#    y=[]
#    for j in range(len(data_data2[i])):
#        if data_data2[i][j]!=0:
#            x.append(j)
#            y.append(data_data2[i][j])
#
#        plt.plot(x,y,"o-")
#plt.title('The Trends of third cluster '+'('+factor_name+')')
#plt.grid(True)
#plt.legend(bbox_to_anchor=(1.0, 1), loc=1, borderaxespad=0.)











