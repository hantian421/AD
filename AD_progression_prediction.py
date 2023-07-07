# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 13:48:12 2022

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


from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.kernel_ridge import KernelRidge

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
import copy
from sklearn.svm import SVC
from scipy.spatial.distance import pdist, squareform
from scipy import exp
from scipy.linalg import eigh
import math
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import NearestCentroid


def RQ(x):# ccccc 
    aaa = 1-x/(x+2)
    return aaa

def Multiquadric(x):# ccccc 
    aaa = np.sqrt(x+math.pow(2,2))
    return aaa



def Polynomial(x):# ccccc a,c,d
    aaa = np.power(5*x+8,4)
    return aaa

def tanh(x):# ccccc a,c,d
    aaa = np.tan(-0.05*x+0.002)
    return aaa


def rbf_kernel_pca(X, gamma, down_D_rate):
    """
    RBF kernel PCA implementation.

    Parameters
    __________
    X: {Numpy ndarray}, shape = [n_samples, n_features]

    gamma: float
        Tuning parameters of the RBF kernel

    n_components: int
        Number of principal components to return

    Returns
    __________
    X_pc: {Numpy ndarray}, shape = [n_samples, k_features]
        Projected dataset

    """
    # Calculate pairwise squared Euclidean distances
    # in the MxN dimensional dataset.
    sq_dist = pdist(X, 'sqeuclidean')

    # Convert pairwise distances into a square matrix.
    mat_sq_dists = squareform(sq_dist)

    # Compute the symmetric kernel matrix.
    K = exp(-gamma * mat_sq_dists)

    # Center the kernel matrix.
    N = K.shape[0]
    one_n = np.ones((N, N)) / N
    K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)
    print("finish")

    # Obtaining eigenpairs from the centered kernel matrix
    # scipy.linalg.eigh returns them in ascending order
    eigvals, eigvecs = eigh(K)
    for i in range(len(eigvals)):
        if eigvals[i]<=0:
            eigvals[i]=0

    n_components=0
    He=sum(eigvals)
    for i in range(len(eigvals)-1):
        AAA=heapq.nlargest(i, list(eigvals))
        if sum(AAA)/He > down_D_rate:
            n_components=i
            break
    print(n_components)
    # Collect the top K eigenvectors (projected samples)
    X_pc = np.column_stack((eigvecs[:, -i] for i in range(1, n_components + 1)))

    return X_pc



def RQ_kernel_pca(X, down_D_rate):

    # Calculate pairwise squared Euclidean distances
    # in the MxN dimensional dataset.
    sq_dist = pdist(X, 'sqeuclidean', p=1)

    # Convert pairwise distances into a square matrix.
    mat_sq_dists = squareform(sq_dist)

    # Compute the symmetric kernel matrix.
#    for i in range(len(mat_sq_dists)):
#        for j in range(len(mat_sq_dists[0])):
#            mat_sq_dists[i][j]=1-(mat_sq_dists[i][j]/(mat_sq_dists[i][j]+c))
#        print(i)
    mat_sq_dists=list(map(RQ,mat_sq_dists))
    mat_sq_dists=np.array(mat_sq_dists)
    K=mat_sq_dists
    # Center the kernel matrix.
    N = K.shape[0]
    one_n = np.ones((N, N)) / N
    K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)

    # Obtaining eigenpairs from the centered kernel matrix
    # scipy.linalg.eigh returns them in ascending order
    eigvals, eigvecs = eigh(K)
    for i in range(len(eigvals)):
        if eigvals[i]<=0:
            eigvals[i]=0

    n_components=0
    He=sum(eigvals)
    for i in range(len(eigvals)-1):
        AAA=heapq.nlargest(i, list(eigvals))
        if sum(AAA)/He > down_D_rate:
            n_components=i
            break
    print(n_components)
    # Collect the top K eigenvectors (projected samples)
    X_pc = np.column_stack((eigvecs[:, -i] for i in range(1, n_components + 1)))

    return X_pc





def Multiquadric_pca(X, down_D_rate):

    # Calculate pairwise squared Euclidean distances
    # in the MxN dimensional dataset.
    sq_dist = pdist(X, 'sqeuclidean', p=1)

    # Convert pairwise distances into a square matrix.
    mat_sq_dists = squareform(sq_dist)

    # Compute the symmetric kernel matrix.
    mat_sq_dists=list(map(Multiquadric,mat_sq_dists))
    mat_sq_dists=np.array(mat_sq_dists)
    K=mat_sq_dists
    # Center the kernel matrix.
    N = K.shape[0]
    one_n = np.ones((N, N)) / N
    K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)

    # Obtaining eigenpairs from the centered kernel matrix
    # scipy.linalg.eigh returns them in ascending order
    eigvals, eigvecs = eigh(K)
    for i in range(len(eigvals)):
        if eigvals[i]<=0:
            eigvals[i]=0

    n_components=0
    He=sum(eigvals)
    for i in range(len(eigvals)-1):
        AAA=heapq.nlargest(i, list(eigvals))
        if sum(AAA)/He > down_D_rate:
            n_components=i
            break
    print(n_components)
    # Collect the top K eigenvectors (projected samples)
    X_pc = np.column_stack((eigvecs[:, -i] for i in range(1, n_components + 1)))

    return X_pc



def Linear_pca(X,a, c, down_D_rate):

    # Calculate pairwise squared Euclidean distances
    # in the MxN dimensional dataset.
    X=np.array(X)
    mat_sq_dists=np.dot(X,X.T)
    mat_sq_dists=a*mat_sq_dists+c*np.ones((len(X),len(X)))
    K=mat_sq_dists
    # Center the kernel matrix.
    N = K.shape[0]
    one_n = np.ones((N, N)) / N
    K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)

    # Obtaining eigenpairs from the centered kernel matrix
    # scipy.linalg.eigh returns them in ascending order
    eigvals, eigvecs = eigh(K)
    for i in range(len(eigvals)):
        if eigvals[i]<=0:
            eigvals[i]=0

    n_components=0
    He=sum(eigvals)
    for i in range(len(eigvals)-1):
        AAA=heapq.nlargest(i, list(eigvals))
        if sum(AAA)/He > down_D_rate:
            n_components=i
            break
    print(n_components)   

    # Collect the top K eigenvectors (projected samples)
    X_pc = np.column_stack((eigvecs[:, -i] for i in range(1, n_components + 1)))

    return X_pc




def Polynomial_pca(X, down_D_rate):

    # Calculate pairwise squared Euclidean distances  5 10 5
    # in the MxN dimensional dataset.
    X=np.array(X)
    mat_sq_dists=np.dot(X,X.T)
    
  
    mat_sq_dists=list(map(Polynomial,mat_sq_dists))
    mat_sq_dists=np.array(mat_sq_dists)
    K=mat_sq_dists
    # Center the kernel matrix.
    N = K.shape[0]
    one_n = np.ones((N, N)) / N
    K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)
    print("!!!!")

    # Obtaining eigenpairs from the centered kernel matrix
    # scipy.linalg.eigh returns them in ascending order
    eigvals, eigvecs = eigh(K)
    for i in range(len(eigvals)):
        if eigvals[i]<=0:
            eigvals[i]=0

    n_components=0
    He=sum(eigvals)
    for i in range(len(eigvals)-1):
        AAA=heapq.nlargest(i, list(eigvals))
        if sum(AAA)/He > down_D_rate:
            n_components=i
            break
    print(n_components)

    # Collect the top K eigenvectors (projected samples)
    X_pc = np.column_stack((eigvecs[:, -i] for i in range(1, n_components + 1)))

    return X_pc



def Sigmoid_pca(X,  down_D_rate):

    # Calculate pairwise squared Euclidean distances  1.5 0.1
    # in the MxN dimensional dataset.

        
    X=np.array(X)
    mat_sq_dists=np.dot(X,X.T)
    
  
    mat_sq_dists=list(map(tanh,mat_sq_dists))
    mat_sq_dists=np.array(mat_sq_dists)  
    
    K=mat_sq_dists
    # Center the kernel matrix.
    N = K.shape[0]
    one_n = np.ones((N, N)) / N
    K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)

    # Obtaining eigenpairs from the centered kernel matrix
    # scipy.linalg.eigh returns them in ascending order
    eigvals, eigvecs = eigh(K)
    for i in range(len(eigvals)):
        if eigvals[i]<=0:
            eigvals[i]=0

    n_components=0
    He=sum(eigvals)
    for i in range(len(eigvals)-1):
        AAA=heapq.nlargest(i, list(eigvals))
        if sum(AAA)/He > down_D_rate:
            n_components=i
            break
    print(n_components)
    # Collect the top K eigenvectors (projected samples)
    X_pc = np.column_stack((eigvecs[:, -i] for i in range(1, n_components + 1)))

    return X_pc




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


for i in range(len(data_done_forth)):
    if data_done_forth[i][0]<data_lengh:
        continue
    for j in range(data_done_forth[i][0]-time_dely-data_lengh):
        if LLabel[i][j+data_lengh+time_dely-1] - LLabel[i][j+data_lengh-1]>0:
            aa=data_done_forth[i][2][j:j+data_lengh]
#            target_data.append(aa)
#            target_label.append(-1)
            bad_change.append(aa)
        if LLabel[i][j+data_lengh+time_dely-1] - LLabel[i][j+data_lengh-1]<0:
            aa=data_done_forth[i][2][j:j+data_lengh]
#            target_data.append(aa)
#            target_label.append(1)
            good_change.append(aa)
        if LLabel[i][j+data_lengh+time_dely-1] - LLabel[i][j+data_lengh-1]==0 and LLabel[i][j+data_lengh+time_dely-1]!=0 and LLabel[i][j+data_lengh-1]!=0:
            aa=data_done_forth[i][2][j:j+data_lengh]
#            target_data.append(aa)
#            target_label.append(0)
            no_change.append(aa)
        if sum(LLabel[i][j:j+data_lengh+time_dely])==0:
            aa=data_done_forth[i][2][j:j+data_lengh]
            background_data.append(aa)
#            target_label.append(0)
        
hee = 1*(len(bad_change) + len(good_change))
zero_choose = random.sample(range(len(no_change)),340)    
for i in range(len(bad_change)):
    target_data.append(bad_change[i])
    target_label.append(2)
for i in range(len(good_change)):   
    target_data.append(good_change[i])
    target_label.append(1)     
for i in range(len(zero_choose)):   
    target_data.append(no_change[zero_choose[i]])
    target_label.append(0) 


TTTT=[]
TTTL=[]
abc=list(range(len(target_label)))
random.shuffle(abc)
for i in range(len(abc)):
    TTTT.append(target_data[abc[i]])
    TTTL.append(target_label[abc[i]])

target_data=TTTT
target_label=TTTL

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




alpha_feature = 0.29681927843915#0.5368103812305062 # 0.5363071422980711#0.5364999459959141#131#205.2299
alpha_time = 1.24097708294546#1.1919837539060254 #1.189518403928483#1.189582328741842#131
down_D_rate=0.85

# changed_Target_eig=np.array(Target_eig)-alpha*np.array(background_eig)
# max_index = map(list(changed_Target_eig).index, heapq.nlargest(down_D, list(changed_Target_eig))) 
# indx=list(set(max_index))

# change_metrix=[]
# for i in range(len(indx)):
#     change_metrix.append(Target_vec[indx[i]])
# change_metrix=np.array(change_metrix)
# for i in range(len(target_data)):
#     target_data[i]=np.matmul(target_data[i],change_metrix.T)

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




full_data=[]
for i in range(len(target_data)):
    full_data.append(target_data[i])
#for i in range(len(background_data)):
#    full_data.append(background_data[i])
    
 
    
# Or_PCA_feature    
data_average=sum(full_data)/len(full_data)
G=np.zeros((full_data[0].shape[1],full_data[0].shape[1]))
for i in range(len(full_data)):
    G=G+np.matmul((full_data[i].T-data_average.T),(full_data[i]-data_average))
G_full_data=G/len(full_data)
full_data_eig,full_data_vec=np.linalg.eig(G_full_data)
He=sum(full_data_eig)
for i in range(len(full_data_eig)-1):
    AAA=heapq.nlargest(i, list(full_data_eig))
    if sum(AAA)/He > down_D_rate:
        down_D=i
        break
max_index = map(list(full_data_eig).index, heapq.nlargest(down_D, list(full_data_eig))) 
indx=list(set(max_index))
change_metrix=[]
orpca_changed_target_data=[]
for i in range(len(indx)):
    change_metrix.append(full_data_vec[:,indx[i]])
change_metrix=np.array(change_metrix)
for i in range(len(target_data)):
    orpca_changed_target_data.append(np.matmul(target_data[i],change_metrix.T))



# kernal_PCA(多项式)   
for i in range(len(full_data)):
    data0=[]
    for j in range(len(full_data[i])):
        data0=data0+list(full_data[i][j])
    full_data[i] = copy.deepcopy(data0)
    print(str(i)+"!!!!!")
 
full_data=np.array(full_data)
    
RBF_pca_changed_target_data = rbf_kernel_pca(full_data, gamma=2, down_D_rate=0.85)
RQ_pca_changed_target_data = RQ_kernel_pca(full_data,  down_D_rate=0.85)
#Multiquadric_pca_changed_target_data = Multiquadric_pca(full_data,  down_D_rate=0.85)
Linear_pca_changed_target_data = Linear_pca(full_data, a=2,c=1, down_D_rate=0.85)
Polynomial_pca_changed_target_data = Polynomial_pca(full_data,  down_D_rate=0.85)
Sigmoid_pca_changed_target_data = Sigmoid_pca(full_data, down_D_rate=0.85)
    
   


##准备
###测试
Orpca_acc=[]
Orpca_recall=[]
Orpca_f1=[]
Kpca_RBF_acc=[]
Kpca_RBF_recall=[]
Kpca_RBF_f1=[]
Kpca_RQ_acc=[]
Kpca_RQ_recall=[]
Kpca_RQ_f1=[]
Kpca_Multiquadric_acc=[]
Kpca_Multiquadric_recall=[]
Kpca_Multiquadric_f1=[]
Kpca_Linear_acc=[]
Kpca_Linear_recall=[]
Kpca_Linear_f1=[]
Kpca_Polynomial_acc=[]
Kpca_Polynomial_recall=[]
Kpca_Polynomial_f1=[]
Kpca_Sigmoid_acc=[]
Kpca_Sigmoid_recall=[]
Kpca_Sigmoid_f1=[]
BEST_Cpca_feature_acc=[]
BEST_Cpca_feature_recall=[]
BEST_Cpca_feature_f1=[]
BEST_Cpca_time_acc=[]
BEST_Cpca_time_recall=[]
BEST_Cpca_time_f1=[]
BEST_Cpca_feature_time_acc=[]
BEST_Cpca_feature_time_recall=[]
BEST_Cpca_feature_time_f1=[]





kf = KFold(n_splits=3,shuffle=True)
for train_index , test_index in kf.split(orpca_changed_target_data):

    ##orpca_changed_target_data    
    
    train_data=[]
    train_label=[]
    test_data=[]
    test_label=[]
    for i in range(len(train_index)):
        train_data.append(orpca_changed_target_data[train_index[i]])
        train_label.append(target_label[train_index[i]])
    for i in range(len(test_index)):
        test_data.append(orpca_changed_target_data[test_index[i]])
        test_label.append(target_label[test_index[i]])
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
    clf = RandomForestClassifier(n_estimators=8,criterion="entropy",max_features="log2",max_samples=0.8, max_depth=4,min_samples_split=10 , min_samples_leaf=30)#KNeighborsClassifier(n_neighbors=4,metric="euclidean")#SVC(C=1, cache_size=500,degree=5,  kernel='poly',decision_function_shape='ovr', probability=True)#KNeighborsClassifier(n_neighbors=4,metric="euclidean")#RandomForestClassifier(n_estimators=3,criterion="gini",max_features="sqrt",max_samples=0.8, max_depth=20,min_samples_split=10 , min_samples_leaf=5)#KNeighborsClassifier(n_neighbors=20,metric="euclidean")#SVC(C=3.5, cache_size=800, degree=15, kernel='rbf',decision_function_shape='ovo', probability=True)##KNeighborsClassifier(n_neighbors=20,metric="euclidean")#RandomForestClassifier(n_estimators=3,criterion="gini",max_features="sqrt",max_samples=0.8, max_depth=20,min_samples_split=10 , min_samples_leaf=5)#SVC(C=0.5, cache_size=500,degree=5,  kernel='poly',decision_function_shape='ovr', probability=True)
    clf = clf.fit(train_data, train_label)
    TTT=clf.predict(test_data)#GaussianNB() KNeighborsClassifier(n_neighbors=10)
    
    Orpca_acc.append(clf.score(test_data,test_label))
    Orpca_recall.append(metrics.recall_score(np.array(test_label), TTT, average='macro'))
    Orpca_f1.append(metrics.f1_score(np.array(test_label), TTT, average='macro'))
    print("Orpca_acc:"+str(clf.score(test_data,test_label)))#clf.score(test_data,test_label)  metrics.f1_score(test_label, TTT, average='micro')
    print("Orpca_recall:"+str(metrics.recall_score(np.array(test_label), TTT, average='macro')))#clf.score(test_data,test_label)  metrics.f1_score(test_label, TTT, average='micro')
    print("Orpca_f1:"+str(metrics.f1_score(np.array(test_label), TTT, average='macro')))#clf.score(test_data,test_label)  metrics.f1_score(test_label, TTT, average='micro')

    
    
    ##RBF_pca_changed_target_data
    train_data=[]
    train_label=[]
    test_data=[]
    test_label=[]
    for i in range(len(train_index)):
        train_data.append(RBF_pca_changed_target_data[train_index[i]])
        train_label.append(target_label[train_index[i]])
    for i in range(len(test_index)):
        test_data.append(RBF_pca_changed_target_data[test_index[i]])
        test_label.append(target_label[test_index[i]])

    clf = RandomForestClassifier(n_estimators=8,criterion="entropy",max_features="log2",max_samples=0.8, max_depth=4,min_samples_split=10 , min_samples_leaf=30)#KNeighborsClassifier(n_neighbors=4,metric="euclidean")#SVC(C=1, cache_size=500,degree=5,  kernel='poly',decision_function_shape='ovr', probability=True)#KNeighborsClassifier(n_neighbors=4,metric="euclidean")#RandomForestClassifier(n_estimators=3,criterion="gini",max_features="sqrt",max_samples=0.8, max_depth=20,min_samples_split=10 , min_samples_leaf=5)#KNeighborsClassifier(n_neighbors=20,metric="euclidean")#SVC(C=3.5, cache_size=800, degree=15, kernel='rbf',decision_function_shape='ovo', probability=True)##KNeighborsClassifier(n_neighbors=20,metric="euclidean")#RandomForestClassifier(n_estimators=3,criterion="gini",max_features="sqrt",max_samples=0.8, max_depth=20,min_samples_split=10 , min_samples_leaf=5)#SVC(C=0.5, cache_size=500,degree=5,  kernel='poly',decision_function_shape='ovr', probability=True)
    clf = clf.fit(train_data, train_label)
    TTT=clf.predict(test_data)
    
    Kpca_RBF_acc.append(clf.score(test_data,test_label))
    Kpca_RBF_recall.append(metrics.recall_score(np.array(test_label), TTT, average='macro'))
    Kpca_RBF_f1.append(metrics.f1_score(np.array(test_label), TTT, average='macro'))
    print("Kpca_RBF_acc:"+str(clf.score(test_data,test_label)))#clf.score(test_data,test_label)  metrics.f1_score(test_label, TTT, average='micro')
    print("Kpca_RBF_recall:"+str(metrics.recall_score(np.array(test_label), TTT, average='macro')))#clf.score(test_data,test_label)  metrics.f1_score(test_label, TTT, average='micro')
    print("Kpca_RBF_f1:"+str(metrics.f1_score(np.array(test_label), TTT, average='macro')))#clf.score(test_data,test_label)  metrics.f1_score(test_label, TTT, average='micro')



    ##RQ_pca_changed_target_data
    train_data=[]
    train_label=[]
    test_data=[]
    test_label=[]
    for i in range(len(train_index)):
        train_data.append(RQ_pca_changed_target_data[train_index[i]])
        train_label.append(target_label[train_index[i]])
    for i in range(len(test_index)):
        test_data.append(RQ_pca_changed_target_data[test_index[i]])
        test_label.append(target_label[test_index[i]])
    
    clf = RandomForestClassifier(n_estimators=8,criterion="entropy",max_features="log2",max_samples=0.8, max_depth=4,min_samples_split=10 , min_samples_leaf=30)#KNeighborsClassifier(n_neighbors=4,metric="euclidean")#SVC(C=1, cache_size=500,degree=5,  kernel='poly',decision_function_shape='ovr', probability=True)#KNeighborsClassifier(n_neighbors=4,metric="euclidean")#RandomForestClassifier(n_estimators=3,criterion="gini",max_features="sqrt",max_samples=0.8, max_depth=20,min_samples_split=10 , min_samples_leaf=5)#KNeighborsClassifier(n_neighbors=20,metric="euclidean")#SVC(C=3.5, cache_size=800, degree=15, kernel='rbf',decision_function_shape='ovo', probability=True)##KNeighborsClassifier(n_neighbors=20,metric="euclidean")#RandomForestClassifier(n_estimators=3,criterion="gini",max_features="sqrt",max_samples=0.8, max_depth=20,min_samples_split=10 , min_samples_leaf=5)#SVC(C=0.5, cache_size=500,degree=5,  kernel='poly',decision_function_shape='ovr', probability=True)
    clf = clf.fit(train_data, train_label)
    TTT=clf.predict(test_data)
    
    Kpca_RQ_acc.append(clf.score(test_data,test_label))
    Kpca_RQ_recall.append(metrics.recall_score(np.array(test_label), TTT, average='macro'))
    Kpca_RQ_f1.append(metrics.f1_score(np.array(test_label), TTT, average='macro'))
    print("Kpca_RQ_acc:"+str(clf.score(test_data,test_label)))#clf.score(test_data,test_label)  metrics.f1_score(test_label, TTT, average='micro')
    print("Kpca_RQ_recall:"+str(metrics.recall_score(np.array(test_label), TTT, average='macro')))#clf.score(test_data,test_label)  metrics.f1_score(test_label, TTT, average='micro')
    print("Kpca_RQ_f1:"+str(metrics.f1_score(np.array(test_label), TTT, average='macro')))#clf.score(test_data,test_label)  metrics.f1_score(test_label, TTT, average='micro')





#    ##Multiquadric_pca_changed_target_data
#    train_data=[]
#    train_label=[]
#    test_data=[]
#    test_label=[]
#    for i in range(len(train_index)):
#        train_data.append(Multiquadric_pca_changed_target_data[train_index[i]])
#        train_label.append(target_label[train_index[i]])
#    for i in range(len(test_index)):
#        test_data.append(Multiquadric_pca_changed_target_data[test_index[i]])
#        test_label.append(target_label[test_index[i]])
#    
#    clf = RandomForestClassifier(n_estimators=3,criterion="gini",max_features="sqrt",max_samples=0.8, max_depth=20,min_samples_split=10 , min_samples_leaf=5)#KNeighborsClassifier(n_neighbors=20,metric="euclidean")#SVC(C=3.5, cache_size=800, degree=15, kernel='rbf',decision_function_shape='ovo', probability=True)##KNeighborsClassifier(n_neighbors=20,metric="euclidean")#RandomForestClassifier(n_estimators=3,criterion="gini",max_features="sqrt",max_samples=0.8, max_depth=20,min_samples_split=10 , min_samples_leaf=5)#SVC(C=0.5, cache_size=500,degree=5,  kernel='poly',decision_function_shape='ovr', probability=True)
#    clf = clf.fit(train_data, train_label)
#    TTT=clf.predict(test_data)
#    
#    Kpca_Multiquadric_acc.append(clf.score(test_data,test_label))
#    Kpca_Multiquadric_recall.append(metrics.recall_score(np.array(test_label), TTT, average='macro'))
#    Kpca_Multiquadric_f1.append(metrics.f1_score(np.array(test_label), TTT, average='macro'))
#    print("Kpca_Multiquadric_acc:"+str(clf.score(test_data,test_label)))#clf.score(test_data,test_label)  metrics.f1_score(test_label, TTT, average='micro')
#    print("Kpca_Multiquadric_recall:"+str(metrics.recall_score(np.array(test_label), TTT, average='macro')))#clf.score(test_data,test_label)  metrics.f1_score(test_label, TTT, average='micro')
#    print("Kpca_Multiquadric_f1:"+str(metrics.f1_score(np.array(test_label), TTT, average='macro')))#clf.score(test_data,test_label)  metrics.f1_score(test_label, TTT, average='micro')




    ##Linear_pca_changed_target_data
    train_data=[]
    train_label=[]
    test_data=[]
    test_label=[]
    for i in range(len(train_index)):
        train_data.append(Linear_pca_changed_target_data[train_index[i]])
        train_label.append(target_label[train_index[i]])
    for i in range(len(test_index)):
        test_data.append(Linear_pca_changed_target_data[test_index[i]])
        test_label.append(target_label[test_index[i]])
    
    clf = RandomForestClassifier(n_estimators=8,criterion="entropy",max_features="log2",max_samples=0.8, max_depth=4,min_samples_split=10, min_samples_leaf=30)#KNeighborsClassifier(n_neighbors=4,metric="euclidean")#SVC(C=1, cache_size=500,degree=5,  kernel='poly',decision_function_shape='ovr', probability=True)#KNeighborsClassifier(n_neighbors=4,metric="euclidean")#RandomForestClassifier(n_estimators=3,criterion="gini",max_features="sqrt",max_samples=0.8, max_depth=20,min_samples_split=10 , min_samples_leaf=5)#KNeighborsClassifier(n_neighbors=20,metric="euclidean")#SVC(C=3.5, cache_size=800, degree=15, kernel='rbf',decision_function_shape='ovo', probability=True)##KNeighborsClassifier(n_neighbors=20,metric="euclidean")#RandomForestClassifier(n_estimators=3,criterion="gini",max_features="sqrt",max_samples=0.8, max_depth=20,min_samples_split=10 , min_samples_leaf=5)#SVC(C=0.5, cache_size=500,degree=5,  kernel='poly',decision_function_shape='ovr', probability=True)
    clf = clf.fit(train_data, train_label)
    TTT=clf.predict(test_data)
    
    Kpca_Linear_acc.append(clf.score(test_data,test_label))
    Kpca_Linear_recall.append(metrics.recall_score(np.array(test_label), TTT, average='macro'))
    Kpca_Linear_f1.append(metrics.f1_score(np.array(test_label), TTT, average='macro'))
    print("Kpca_Linear_acc:"+str(clf.score(test_data,test_label)))#clf.score(test_data,test_label)  metrics.f1_score(test_label, TTT, average='micro')
    print("Kpca_Linear_recall:"+str(metrics.recall_score(np.array(test_label), TTT, average='macro')))#clf.score(test_data,test_label)  metrics.f1_score(test_label, TTT, average='micro')
    print("Kpca_Linear_f1:"+str(metrics.f1_score(np.array(test_label), TTT, average='macro')))#clf.score(test_data,test_label)  metrics.f1_score(test_label, TTT, average='micro')





    ##Polynomial_pca_changed_target_data
    train_data=[]
    train_label=[]
    test_data=[]
    test_label=[]
    for i in range(len(train_index)):
        train_data.append(Polynomial_pca_changed_target_data[train_index[i]])
        train_label.append(target_label[train_index[i]])
    for i in range(len(test_index)):
        test_data.append(Polynomial_pca_changed_target_data[test_index[i]])
        test_label.append(target_label[test_index[i]])
    
    clf = RandomForestClassifier(n_estimators=8,criterion="entropy",max_features="log2",max_samples=0.8, max_depth=4,min_samples_split=10 , min_samples_leaf=30)#KNeighborsClassifier(n_neighbors=4,metric="euclidean")#SVC(C=1, cache_size=500,degree=5,  kernel='poly',decision_function_shape='ovr', probability=True)#KNeighborsClassifier(n_neighbors=4,metric="euclidean")#RandomForestClassifier(n_estimators=3,criterion="gini",max_features="sqrt",max_samples=0.8, max_depth=20,min_samples_split=10 , min_samples_leaf=5)#KNeighborsClassifier(n_neighbors=20,metric="euclidean")##KNeighborsClassifier(n_neighbors=20,metric="euclidean")#RandomForestClassifier(n_estimators=3,criterion="gini",max_features="sqrt",max_samples=0.8, max_depth=20,min_samples_split=10 , min_samples_leaf=5)#SVC(C=0.5, cache_size=500,degree=5,  kernel='poly',decision_function_shape='ovr', probability=True)
    clf = clf.fit(train_data, train_label)
    TTT=clf.predict(test_data)
    
    Kpca_Polynomial_acc.append(clf.score(test_data,test_label))
    Kpca_Polynomial_recall.append(metrics.recall_score(np.array(test_label), TTT, average='macro'))
    Kpca_Polynomial_f1.append(metrics.f1_score(np.array(test_label), TTT, average='macro'))
    print("Kpca_Polynomial_acc:"+str(clf.score(test_data,test_label)))#clf.score(test_data,test_label)  metrics.f1_score(test_label, TTT, average='micro')
    print("Kpca_Polynomial_recall:"+str(metrics.recall_score(np.array(test_label), TTT, average='macro')))#clf.score(test_data,test_label)  metrics.f1_score(test_label, TTT, average='micro')
    print("Kpca_Polynomial_f1:"+str(metrics.f1_score(np.array(test_label), TTT, average='macro')))#clf.score(test_data,test_label)  metrics.f1_score(test_label, TTT, average='micro')







    ##Sigmoid_pca_changed_target_data
    train_data=[]
    train_label=[]
    test_data=[]
    test_label=[]
    for i in range(len(train_index)):
        train_data.append(Sigmoid_pca_changed_target_data[train_index[i]])
        train_label.append(target_label[train_index[i]])
    for i in range(len(test_index)):
        test_data.append(Sigmoid_pca_changed_target_data[test_index[i]])
        test_label.append(target_label[test_index[i]])
    
    clf = RandomForestClassifier(n_estimators=8,criterion="entropy",max_features="log2",max_samples=0.8, max_depth=4,min_samples_split=10 , min_samples_leaf=30)#KNeighborsClassifier(n_neighbors=4,metric="euclidean")#SVC(C=1, cache_size=500,degree=5,  kernel='poly',decision_function_shape='ovr', probability=True)#KNeighborsClassifier(n_neighbors=4,metric="euclidean")#RandomForestClassifier(n_estimators=3,criterion="gini",max_features="sqrt",max_samples=0.8, max_depth=20,min_samples_split=10 , min_samples_leaf=5)#KNeighborsClassifier(n_neighbors=20,metric="euclidean")#SVC(C=3.5, cache_size=800, degree=15, kernel='rbf',decision_function_shape='ovo', probability=True)##KNeighborsClassifier(n_neighbors=20,metric="euclidean")#RandomForestClassifier(n_estimators=3,criterion="gini",max_features="sqrt",max_samples=0.8, max_depth=20,min_samples_split=10 , min_samples_leaf=5)#SVC(C=0.5, cache_size=500,degree=5,  kernel='poly',decision_function_shape='ovr', probability=True)
    clf = clf.fit(train_data, train_label)
    TTT=clf.predict(test_data)
    
    Kpca_Sigmoid_acc.append(clf.score(test_data,test_label))
    Kpca_Sigmoid_recall.append(metrics.recall_score(np.array(test_label), TTT, average='macro'))
    Kpca_Sigmoid_f1.append(metrics.f1_score(np.array(test_label), TTT, average='macro'))
    print("Kpca_Sigmoid_acc:"+str(clf.score(test_data,test_label)))#clf.score(test_data,test_label)  metrics.f1_score(test_label, TTT, average='micro')
    print("Kpca_Sigmoid_recall:"+str(metrics.recall_score(np.array(test_label), TTT, average='macro')))#clf.score(test_data,test_label)  metrics.f1_score(test_label, TTT, average='micro')
    print("Kpca_Sigmoid_f1:"+str(metrics.f1_score(np.array(test_label), TTT, average='macro')))#clf.score(test_data,test_label)  metrics.f1_score(test_label, TTT, average='micro')




    ##cpca_changed_target_data_feature
    train_data=[]
    train_label=[]
    test_data=[]
    test_label=[]
    for i in range(len(train_index)):
        train_data.append(cpca_changed_target_data_feature[train_index[i]])
        train_label.append(target_label[train_index[i]])
    for i in range(len(test_index)):
        test_data.append(cpca_changed_target_data_feature[test_index[i]])
        test_label.append(target_label[test_index[i]])
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
    clf = RandomForestClassifier(n_estimators=8,criterion="entropy",max_features="log2",max_samples=0.8, max_depth=4,min_samples_split=10 , min_samples_leaf=30)#KNeighborsClassifier(n_neighbors=4,metric="euclidean")#SVC(C=1, cache_size=500,degree=5,  kernel='poly',decision_function_shape='ovr', probability=True)#KNeighborsClassifier(n_neighbors=4,metric="euclidean")#RandomForestClassifier(n_estimators=3,criterion="gini",max_features="sqrt",max_samples=0.8, max_depth=20,min_samples_split=10 , min_samples_leaf=5)#KNeighborsClassifier(n_neighbors=20,metric="euclidean")#SVC(C=3.5, cache_size=800, degree=15, kernel='rbf',decision_function_shape='ovo', probability=True)##KNeighborsClassifier(n_neighbors=20,metric="euclidean")#RandomForestClassifier(n_estimators=3,criterion="gini",max_features="sqrt",max_samples=0.8, max_depth=20,min_samples_split=10 , min_samples_leaf=5)#SVC(C=0.5, cache_size=500,degree=5,  kernel='poly',decision_function_shape='ovr', probability=True)
    clf = clf.fit(train_data, train_label)
    TTT=clf.predict(test_data)
    
    BEST_Cpca_feature_acc.append(clf.score(test_data,test_label))
    BEST_Cpca_feature_recall.append(metrics.recall_score(np.array(test_label), TTT, average='macro'))
    BEST_Cpca_feature_f1.append(metrics.f1_score(np.array(test_label), TTT, average='macro'))
    print("BEST_Cpca_feature_acc:"+str(clf.score(test_data,test_label)))#clf.score(test_data,test_label)  metrics.f1_score(test_label, TTT, average='micro')
    print("BEST_Cpca_feature_recall:"+str(metrics.recall_score(np.array(test_label), TTT, average='macro')))#clf.score(test_data,test_label)  metrics.f1_score(test_label, TTT, average='micro')
    print("BEST_Cpca_feature_f1:"+str(metrics.f1_score(np.array(test_label), TTT, average='macro')))#clf.score(test_data,test_label)  metrics.f1_score(test_label, TTT, average='micro')




    ##cpca_changed_target_data_time
    train_data=[]
    train_label=[]
    test_data=[]
    test_label=[]
    for i in range(len(train_index)):
        train_data.append(cpca_changed_target_data_time[train_index[i]])
        train_label.append(target_label[train_index[i]])
    for i in range(len(test_index)):
        test_data.append(cpca_changed_target_data_time[test_index[i]])
        test_label.append(target_label[test_index[i]])
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
    clf = RandomForestClassifier(n_estimators=8,criterion="entropy",max_features="log2",max_samples=0.8, max_depth=4,min_samples_split=10 , min_samples_leaf=30)#KNeighborsClassifier(n_neighbors=4,metric="euclidean")#SVC(C=1, cache_size=500,degree=5,  kernel='poly',decision_function_shape='ovr', probability=True)#KNeighborsClassifier(n_neighbors=4,metric="euclidean")#RandomForestClassifier(n_estimators=3,criterion="gini",max_features="sqrt",max_samples=0.8, max_depth=20,min_samples_split=10 , min_samples_leaf=5)#KNeighborsClassifier(n_neighbors=20,metric="euclidean")#SVC(C=3.5, cache_size=800, degree=15, kernel='rbf',decision_function_shape='ovo', probability=True)##KNeighborsClassifier(n_neighbors=20,metric="euclidean")#RandomForestClassifier(n_estimators=3,criterion="gini",max_features="sqrt",max_samples=0.8, max_depth=20,min_samples_split=10 , min_samples_leaf=5)#SVC(C=0.5, cache_size=500,degree=5,  kernel='poly',decision_function_shape='ovr', probability=True)
    clf = clf.fit(train_data, train_label)
    TTT=clf.predict(test_data)
    
    BEST_Cpca_time_acc.append(clf.score(test_data,test_label))
    BEST_Cpca_time_recall.append(metrics.recall_score(np.array(test_label), TTT, average='macro'))
    BEST_Cpca_time_f1.append(metrics.f1_score(np.array(test_label), TTT, average='macro'))
    print("BEST_Cpca_time prediction_acc:"+str(clf.score(test_data,test_label)))#clf.score(test_data,test_label)  metrics.f1_score(test_label, TTT, average='micro')
    print("BEST_Cpca_time prediction_recall:"+str(metrics.recall_score(np.array(test_label), TTT, average='macro')))#clf.score(test_data,test_label)  metrics.f1_score(test_label, TTT, average='micro')
    print("BEST_Cpca_time prediction_f1:"+str(metrics.f1_score(np.array(test_label), TTT, average='macro')))#clf.score(test_data,test_label)  metrics.f1_score(test_label, TTT, average='micro')




    ##cpca_changed_target_data_feature_time
    train_data=[]
    train_label=[]
    test_data=[]
    test_label=[]
    for i in range(len(train_index)):
        train_data.append(cpca_changed_target_data_feature_time[train_index[i]])
        train_label.append(target_label[train_index[i]])
    for i in range(len(test_index)):
        test_data.append(cpca_changed_target_data_feature_time[test_index[i]])
        test_label.append(target_label[test_index[i]])
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
    clf = RandomForestClassifier(n_estimators=8,criterion="entropy",max_features="log2",max_samples=0.8, max_depth=4,min_samples_split=10 , min_samples_leaf=30)#KNeighborsClassifier(n_neighbors=4,metric="euclidean")#SVC(C=1, cache_size=500,degree=5,  kernel='poly',decision_function_shape='ovr', probability=True)#KNeighborsClassifier(n_neighbors=4,metric="euclidean")#RandomForestClassifier(n_estimators=3,criterion="gini",max_features="sqrt",max_samples=0.8, max_depth=20,min_samples_split=10 , min_samples_leaf=5)#KNeighborsClassifier(n_neighbors=20,metric="euclidean")#SVC(C=3.5, cache_size=800, degree=15, kernel='rbf',decision_function_shape='ovo', probability=True)##KNeighborsClassifier(n_neighbors=20,metric="euclidean")#SGDClassifier(loss="log",learning_rate='adaptive',eta0=0.4, penalty="elasticnet")#RandomForestClassifier(n_estimators=3,criterion="gini",max_features="sqrt",max_samples=0.8, max_depth=20,min_samples_split=10 , min_samples_leaf=5)#SVC(C=0.5, cache_size=500,degree=5,  kernel='poly',decision_function_shape='ovr', probability=True)
    clf = clf.fit(train_data, train_label)
    TTT=clf.predict(test_data)#
    
    BEST_Cpca_feature_time_acc.append(clf.score(test_data,test_label))
    BEST_Cpca_feature_time_recall.append(metrics.recall_score(np.array(test_label), TTT, average='macro'))
    BEST_Cpca_feature_time_f1.append(metrics.f1_score(np.array(test_label), TTT, average='macro'))
    print("BEST_Cpca_feature_time prediction_acc:"+str(clf.score(test_data,test_label)))#clf.score(test_data,test_label)  metrics.f1_score(test_label, TTT, average='micro')
    print("BEST_Cpca_feature_time prediction_recall:"+str(metrics.recall_score(np.array(test_label), TTT, average='macro')))#clf.score(test_data,test_label)  metrics.f1_score(test_label, TTT, average='micro')
    print("BEST_Cpca_feature_time prediction_f1:"+str(metrics.f1_score(np.array(test_label), TTT, average='macro')))#clf.score(test_data,test_label)  metrics.f1_score(test_label, TTT, average='micro')






print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")


print("Orpca_acc:"+str(np.mean(Orpca_acc)))#clf.score(test_data,test_label)  metrics.f1_score(test_label, TTT, average='micro')
print("Orpca_recall:"+str(np.mean(Orpca_recall)))#clf.score(test_data,test_label)  metrics.f1_score(test_label, TTT, average='micro')
print("Orpca_f1:"+str(np.mean(Orpca_f1)))#clf.score(test_data,test_label)  metrics.f1_score(test_label, TTT, average='micro')

print("Kpca_RBF_acc:"+str(np.mean(Kpca_RBF_acc)))#clf.score(test_data,test_label)  metrics.f1_score(test_label, TTT, average='micro')
print("Kpca_RBF_recall:"+str(np.mean(Kpca_RBF_recall)))#clf.score(test_data,test_label)  metrics.f1_score(test_label, TTT, average='micro')
print("Kpca_RBF_f1:"+str(np.mean(Kpca_RBF_f1)))#clf.score(test_data,test_label)  metrics.f1_score(test_label, TTT, average='micro')

print("Kpca_RQ_acc:"+str(np.mean(Kpca_RQ_acc)))#clf.score(test_data,test_label)  metrics.f1_score(test_label, TTT, average='micro')
print("Kpca_RQ_recall:"+str(np.mean(Kpca_RQ_recall)))#clf.score(test_data,test_label)  metrics.f1_score(test_label, TTT, average='micro')
print("Kpca_RQ_f1:"+str(np.mean(Kpca_RQ_f1)))#clf.score(test_data,test_label)  metrics.f1_score(test_label, TTT, average='micro')

#print("Kpca_Multiquadric_acc:"+str(np.mean(Kpca_Multiquadric_acc)))#clf.score(test_data,test_label)  metrics.f1_score(test_label, TTT, average='micro')
#print("Kpca_Multiquadric_recall:"+str(np.mean(Kpca_Multiquadric_recall)))#clf.score(test_data,test_label)  metrics.f1_score(test_label, TTT, average='micro')
#print("Kpca_Multiquadric_f1:"+str(np.mean(Kpca_Multiquadric_f1)))#clf.score(test_data,test_label)  metrics.f1_score(test_label, TTT, average='micro')

print("Kpca_Linear_acc:"+str(np.mean(Kpca_Linear_acc)))#clf.score(test_data,test_label)  metrics.f1_score(test_label, TTT, average='micro')
print("Kpca_Linear_recall:"+str(np.mean(Kpca_Linear_recall)))#clf.score(test_data,test_label)  metrics.f1_score(test_label, TTT, average='micro')
print("Kpca_Linear_f1:"+str(np.mean(Kpca_Linear_f1)))#clf.score(test_data,test_label)  metrics.f1_score(test_label, TTT, average='micro')

print("Kpca_Polynomial_acc:"+str(np.mean(Kpca_Polynomial_acc)))#clf.score(test_data,test_label)  metrics.f1_score(test_label, TTT, average='micro')
print("Kpca_Polynomial_recall:"+str(np.mean(Kpca_Polynomial_recall)))#clf.score(test_data,test_label)  metrics.f1_score(test_label, TTT, average='micro')
print("Kpca_Polynomial_f1:"+str(np.mean(Kpca_Polynomial_f1)))#clf.score(test_data,test_label)  metrics.f1_score(test_label, TTT, average='micro')

print("Kpca_Sigmoid_acc:"+str(np.mean(Kpca_Sigmoid_acc)))#clf.score(test_data,test_label)  metrics.f1_score(test_label, TTT, average='micro')
print("Kpca_Sigmoid_recall:"+str(np.mean(Kpca_Sigmoid_recall)))#clf.score(test_data,test_label)  metrics.f1_score(test_label, TTT, average='micro')
print("Kpca_Sigmoid_f1:"+str(np.mean(Kpca_Sigmoid_f1)))#clf.score(test_data,test_label)  metrics.f1_score(test_label, TTT, average='micro')

print("BEST_Cpca_feature_acc:"+str(np.mean(BEST_Cpca_feature_acc)))#clf.score(test_data,test_label)  metrics.f1_score(test_label, TTT, average='micro')
print("BEST_Cpca_feature_recall:"+str(np.mean(BEST_Cpca_feature_recall)))#clf.score(test_data,test_label)  metrics.f1_score(test_label, TTT, average='micro')
print("BEST_Cpca_feature_f1:"+str(np.mean(BEST_Cpca_feature_f1)))#clf.score(test_data,test_label)  metrics.f1_score(test_label, TTT, average='micro')

print("BEST_Cpca_time_acc:"+str(np.mean(BEST_Cpca_time_acc)))#clf.score(test_data,test_label)  metrics.f1_score(test_label, TTT, average='micro')
print("BEST_Cpca_time_recall:"+str(np.mean(BEST_Cpca_time_recall)))#clf.score(test_data,test_label)  metrics.f1_score(test_label, TTT, average='micro')
print("BEST_Cpca_time_f1:"+str(np.mean(BEST_Cpca_time_f1)))#clf.score(test_data,test_label)  metrics.f1_score(test_label, TTT, average='micro')

print("BEST_Cpca_feature_time_acc:"+str(np.mean(BEST_Cpca_feature_time_acc)))#clf.score(test_data,test_label)  metrics.f1_score(test_label, TTT, average='micro')
print("BEST_Cpca_feature_time_recall:"+str(np.mean(BEST_Cpca_feature_time_recall)))#clf.score(test_data,test_label)  metrics.f1_score(test_label, TTT, average='micro')
print("BEST_Cpca_feature_time_f1:"+str(np.mean(BEST_Cpca_feature_time_f1)))#clf.score(test_data,test_label)  metrics.f1_score(test_label, TTT, average='micro')


