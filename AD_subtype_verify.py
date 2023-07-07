# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 16:54:09 2022

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





data_change_name=[]
target_data=[]
target_label=[]
background_data=[]
data_lengh=8
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




alpha_feature = 0.296819278439159#0.5363071422980711#0.5364999459959141#131#205.2299
alpha_time = 1.24097708294546#1.189518403928483#1.189582328741842#131
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
    
    
    
##diyilei
cluseter1_acc=[]
cluseter1_recall=[]
cluseter1_f1=[]



dddddata=[]
lllllabel=[]
for i in range(len(cluse_data_name0)):
    d_name=cluse_data_name0[i]
    for j in range(len(data_data_name)):
        if data_data_name[j]==d_name:
            dddddata.append(cpca_changed_target_data_feature_time[j])
            lllllabel.append(target_label[j])

kf = KFold(n_splits=3,shuffle=True)
for train_index , test_index in kf.split(dddddata):

    ##orpca_changed_target_data    
    
    train_data=[]
    train_label=[]
    test_data=[]
    test_label=[]
    for i in range(len(train_index)):
        train_data.append(dddddata[train_index[i]])
        train_label.append(lllllabel[train_index[i]])
    for i in range(len(test_index)):
        test_data.append(dddddata[test_index[i]])
        test_label.append(lllllabel[test_index[i]])
    for i in range(len(train_data)):
        data0=train_data[i]
        data1=[]
        for j in range(len(data0)):
            data1=data1+list(data0[j])
        train_data[i]=data1
    for i in range(len(test_data)):
        data0=test_data[i]
        data1=[]
        for j in range(len(data0)):
            data1=data1+list(data0[j])
        test_data[i]=data1
    clf = KNeighborsClassifier(n_neighbors=4,metric="euclidean")#RandomForestClassifier(n_estimators=3,criterion="gini",max_features="sqrt",max_depth=20,min_samples_split=10 , min_samples_leaf=5)#MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(60,80,40),activation='tanh')#KNeighborsClassifier(n_neighbors=4,metric="euclidean")#RandomForestClassifier(n_estimators=20,criterion="gini",max_features="sqrt",max_depth=20,min_samples_split=10 , min_samples_leaf=5)#DecisionTreeClassifier(criterion='gini', splitter='random', max_depth=10,random_state=1)#SVC(C=3.5, cache_size=800, degree=15, kernel='rbf',decision_function_shape='ovo', probability=True)#AdaBoostClassifier()#GradientBoostingClassifier(n_estimators=200)#KNeighborsClassifier()#DecisionTreeClassifier(criterion='entropy', splitter='random', max_depth=10,random_state=1)
    clf = clf.fit(train_data, train_label)
    TTT=clf.predict(test_data)
    
    cluseter1_acc.append(clf.score(test_data,test_label))
    cluseter1_recall.append(metrics.recall_score(np.array(test_label), TTT, average='macro'))
    cluseter1_f1.append(metrics.f1_score(np.array(test_label), TTT, average='macro'))
cluseter1_len=len(dddddata)
print("cluseter1_acc:"+str(np.mean(cluseter1_acc)))#clf.score(test_data,test_label)  metrics.f1_score(test_label, TTT, average='micro')
print("cluseter1_recall:"+str(np.mean(cluseter1_recall)))#clf.score(test_data,test_label)  metrics.f1_score(test_label, TTT, average='micro')
print("cluseter1_f1:"+str(np.mean(cluseter1_f1)))#clf.score(test_data,test_label)  metrics.f1_score(test_label, TTT, average='micro')    
    
    
    
    
    
    
    
##diyilei
cluseter2_acc=[]
cluseter2_recall=[]
cluseter2_f1=[]



dddddata=[]
lllllabel=[]
for i in range(len(cluse_data_name1)):
    d_name=cluse_data_name1[i]
    for j in range(len(data_data_name)):
        if data_data_name[j]==d_name:
            dddddata.append(cpca_changed_target_data_feature_time[j])
            lllllabel.append(target_label[j])

kf = KFold(n_splits=3,shuffle=True)
for train_index , test_index in kf.split(dddddata):

    ##orpca_changed_target_data    
    
    train_data=[]
    train_label=[]
    test_data=[]
    test_label=[]
    for i in range(len(train_index)):
        train_data.append(dddddata[train_index[i]])
        train_label.append(lllllabel[train_index[i]])
    for i in range(len(test_index)):
        test_data.append(dddddata[test_index[i]])
        test_label.append(lllllabel[test_index[i]])
    for i in range(len(train_data)):
        data0=train_data[i]
        data1=[]
        for j in range(len(data0)):
            data1=data1+list(data0[j])
        train_data[i]=data1
    for i in range(len(test_data)):
        data0=test_data[i]
        data1=[]
        for j in range(len(data0)):
            data1=data1+list(data0[j])
        test_data[i]=data1
    clf = KNeighborsClassifier(n_neighbors=4,metric="euclidean")#RandomForestClassifier(n_estimators=3,criterion="gini",max_features="sqrt",max_depth=20,min_samples_split=10 , min_samples_leaf=5)#MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(60,80,40),activation='tanh')#KNeighborsClassifier(n_neighbors=4,metric="euclidean")#RandomForestClassifier(n_estimators=20,criterion="gini",max_features="sqrt", max_depth=20,min_samples_split=10 , min_samples_leaf=5)#DecisionTreeClassifier(criterion='gini', splitter='random', max_depth=10,random_state=1)#KNeighborsClassifier(n_neighbors=15, weights='distance')#SVC(C=3.5, cache_size=800, degree=15, kernel='rbf',decision_function_shape='ovo', probability=True)#AdaBoostClassifier()#GradientBoostingClassifier(n_estimators=200)#KNeighborsClassifier()#DecisionTreeClassifier(criterion='entropy', splitter='random', max_depth=10,random_state=1)
    clf = clf.fit(train_data, train_label)
    TTT=clf.predict(test_data)
    
    cluseter2_acc.append(clf.score(test_data,test_label))
    cluseter2_recall.append(metrics.recall_score(np.array(test_label), TTT, average='macro'))
    cluseter2_f1.append(metrics.f1_score(np.array(test_label), TTT, average='macro'))
cluseter2_len=len(dddddata)
print("cluseter2_acc:"+str(np.mean(cluseter2_acc)))#clf.score(test_data,test_label)  metrics.f1_score(test_label, TTT, average='micro')
print("cluseter2_recall:"+str(np.mean(cluseter2_recall)))#clf.score(test_data,test_label)  metrics.f1_score(test_label, TTT, average='micro')
print("cluseter2_f1:"+str(np.mean(cluseter2_f1)))#clf.score(test_data,test_label)  metrics.f1_score(test_label, TTT, average='micro')    
       
    
    
    
    
    
    
##diyilei
cluseter3_acc=[]
cluseter3_recall=[]
cluseter3_f1=[]

ddddddddd0=[]

dddddata=[]
lllllabel=[]
for i in range(len(cluse_data_name2)):
    d_name=cluse_data_name2[i]
    for j in range(len(data_data_name)):
        if data_data_name[j]==d_name:
            dddddata.append(cpca_changed_target_data_feature_time[j])
            lllllabel.append(target_label[j])

kf = KFold(n_splits=3,shuffle=True)
for train_index , test_index in kf.split(dddddata):

    ##orpca_changed_target_data    
    
    train_data=[]
    train_label=[]
    test_data=[]
    test_label=[]
    for i in range(len(train_index)):
        train_data.append(dddddata[train_index[i]])
        train_label.append(lllllabel[train_index[i]])
    for i in range(len(test_index)):
        test_data.append(dddddata[test_index[i]])
        test_label.append(lllllabel[test_index[i]])
    for i in range(len(train_data)):
        data0=train_data[i]
        data1=[]
        for j in range(len(data0)):
            data1=data1+list(data0[j])
        train_data[i]=data1
    for i in range(len(test_data)):
        data0=test_data[i]
        data1=[]
        for j in range(len(data0)):
            data1=data1+list(data0[j])
        test_data[i]=data1
    clf = KNeighborsClassifier(n_neighbors=4,metric="euclidean")#RandomForestClassifier(n_estimators=3,criterion="gini",max_features="sqrt",max_depth=20,min_samples_split=10 , min_samples_leaf=5)#MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(60,80,40),activation='tanh')#KNeighborsClassifier(n_neighbors=4,metric="euclidean")#RandomForestClassifier(n_estimators=20,criterion="gini",max_features="sqrt", max_depth=20,min_samples_split=10 , min_samples_leaf=5)#DecisionTreeClassifier(criterion='gini', splitter='random', max_depth=10,random_state=1)#KNeighborsClassifier(n_neighbors=15, weights='distance')#SVC(C=3.5, cache_size=800, degree=15, kernel='rbf',decision_function_shape='ovo', probability=True)#AdaBoostClassifier()#GradientBoostingClassifier(n_estimators=200)#KNeighborsClassifier()#DecisionTreeClassifier(criterion='entropy', splitter='random', max_depth=10,random_state=1)
    clf = clf.fit(train_data, train_label)
    ddddddddd0.append(train_label)
    TTT=clf.predict(test_data)
    
    cluseter3_acc.append(clf.score(test_data,test_label))
    cluseter3_recall.append(metrics.recall_score(np.array(test_label), TTT, average='macro'))
    cluseter3_f1.append(metrics.f1_score(np.array(test_label), TTT, average='macro'))
cluseter3_len=len(dddddata)
print("cluseter3_acc:"+str(np.mean(cluseter3_acc)))#clf.score(test_data,test_label)  metrics.f1_score(test_label, TTT, average='micro')
print("cluseter3_recall:"+str(np.mean(cluseter3_recall)))#clf.score(test_data,test_label)  metrics.f1_score(test_label, TTT, average='micro')
print("cluseter3_f1:"+str(np.mean(cluseter3_f1)))#clf.score(test_data,test_label)  metrics.f1_score(test_label, TTT, average='micro')    
           

##diyilei
cluseter4_acc=[]
cluseter4_recall=[]
cluseter4_f1=[]

ddddddddd0=[]

dddddata=[]
lllllabel=[]
for i in range(len(cluse_data_name3)):
    d_name=cluse_data_name3[i]
    for j in range(len(data_data_name)):
        if data_data_name[j]==d_name:
            dddddata.append(cpca_changed_target_data_feature_time[j])
            lllllabel.append(target_label[j])

kf = KFold(n_splits=3,shuffle=True)
for train_index , test_index in kf.split(dddddata):

    ##orpca_changed_target_data    
    
    train_data=[]
    train_label=[]
    test_data=[]
    test_label=[]
    for i in range(len(train_index)):
        train_data.append(dddddata[train_index[i]])
        train_label.append(lllllabel[train_index[i]])
    for i in range(len(test_index)):
        test_data.append(dddddata[test_index[i]])
        test_label.append(lllllabel[test_index[i]])
    for i in range(len(train_data)):
        data0=train_data[i]
        data1=[]
        for j in range(len(data0)):
            data1=data1+list(data0[j])
        train_data[i]=data1
    for i in range(len(test_data)):
        data0=test_data[i]
        data1=[]
        for j in range(len(data0)):
            data1=data1+list(data0[j])
        test_data[i]=data1
    clf = KNeighborsClassifier(n_neighbors=4,metric="euclidean")#RandomForestClassifier(n_estimators=3,criterion="gini",max_features="sqrt",max_depth=20,min_samples_split=10 , min_samples_leaf=5)#MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(60,80,40),activation='tanh')#KNeighborsClassifier(n_neighbors=4,metric="euclidean")#RandomForestClassifier(n_estimators=20,criterion="gini",max_features="sqrt", max_depth=20,min_samples_split=10 , min_samples_leaf=5)#DecisionTreeClassifier(criterion='gini', splitter='random', max_depth=10,random_state=1)#KNeighborsClassifier(n_neighbors=15, weights='distance')#SVC(C=3.5, cache_size=800, degree=15, kernel='rbf',decision_function_shape='ovo', probability=True)#AdaBoostClassifier()#GradientBoostingClassifier(n_estimators=200)#KNeighborsClassifier()#DecisionTreeClassifier(criterion='entropy', splitter='random', max_depth=10,random_state=1)
    clf = clf.fit(train_data, train_label)
    ddddddddd0.append(train_label)
    TTT=clf.predict(test_data)
    
    cluseter4_acc.append(clf.score(test_data,test_label))
    cluseter4_recall.append(metrics.recall_score(np.array(test_label), TTT, average='macro'))
    cluseter4_f1.append(metrics.f1_score(np.array(test_label), TTT, average='macro'))
cluseter4_len=len(dddddata)
print("cluseter4_acc:"+str(np.mean(cluseter4_acc)))#clf.score(test_data,test_label)  metrics.f1_score(test_label, TTT, average='micro')
print("cluseter4_recall:"+str(np.mean(cluseter4_recall)))#clf.score(test_data,test_label)  metrics.f1_score(test_label, TTT, average='micro')
print("cluseter4_f1:"+str(np.mean(cluseter4_f1)))#clf.score(test_data,test_label)  metrics.f1_score(test_label, TTT, average='micro')    





print((np.mean(cluseter1_acc)+np.mean(cluseter2_acc)+np.mean(cluseter3_acc)+np.mean(cluseter4_acc))/4)
print((np.mean(cluseter1_recall)+np.mean(cluseter2_recall)+np.mean(cluseter3_recall)+np.mean(cluseter4_recall))/4)
print((np.mean(cluseter1_f1)+np.mean(cluseter2_f1)+np.mean(cluseter3_f1)+np.mean(cluseter4_f1))/4)


#final_acc=(np.mean(cluseter1_acc)*cluseter1_len+np.mean(cluseter2_acc)*cluseter2_len+np.mean(cluseter3_acc)*cluseter3_len)/(cluseter1_len+cluseter2_len+cluseter3_len)
#print(final_acc)
#    
#final_acc=(np.mean(cluseter1_recall)*cluseter1_len+np.mean(cluseter2_recall)*cluseter2_len+np.mean(cluseter3_recall)*cluseter3_len)/(cluseter1_len+cluseter2_len+cluseter3_len)
#print(final_acc)
#
#final_acc=(np.mean(cluseter1_f1)*cluseter1_len+np.mean(cluseter2_f1)*cluseter2_len+np.mean(cluseter3_f1)*cluseter3_len)/(cluseter1_len+cluseter2_len+cluseter3_len)
#print(final_acc)

