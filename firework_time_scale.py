# -*- coding: utf-8 -*-
"""
Created on Fri Dec 24 15:33:20 2021

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


def Dis_pair_cal(data0,data1):
    data0_change=[]
    data1_change=[]
    for i in range(len(data0)):
        data0_change=data0_change+list(data0[i])
    for i in range(len(data1)):
        data1_change=data1_change+list(data1[i])
        
    data0_change=np.array(data0_change)
    data1_change=np.array(data1_change)
    dis=np.sqrt(np.sum(np.square(data0_change-data1_change)))
    return dis    


def Dis_cal(x,target_label): 
    TL_dis=0
    YL_dis=0
    for i in range(len(x)-1):
        data0=x[i]
        for j in range(len(x)-i-1):
            data1=x[i+j+1]
            if target_label[i]==target_label[i+j+1]:
                TL_dis=TL_dis+Dis_pair_cal(data0,data1)
            else:
                YL_dis=YL_dis+Dis_pair_cal(data0,data1)
#        print(i)
        
    print('finish')
    return TL_dis-YL_dis




    
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
        
hee = len(bad_change) + len(good_change)
zero_choose = random.sample(range(len(no_change)),hee)    
for i in range(len(bad_change)):
    target_data.append(bad_change[i])
    target_label.append(-1)
for i in range(len(good_change)):   
    target_data.append(good_change[i])
    target_label.append(1)     
for i in range(len(zero_choose)):   
    target_data.append(no_change[zero_choose[i]])
    target_label.append(0) 
    


or_target_data=target_data
or_target_label=target_label
for i in range(len(target_data)):
    target_data[i]=np.array(target_data[i])
for i in range(len(background_data)):
    background_data[i]=np.array(background_data[i])
#
##烟花算法
Rate=0.85 ##主成分百分比
group_num=30
group_number=np.random.uniform(0,30,group_num)
itation=35
rate=0.25 #3烟花算法主烟花保留百分比
A=500
M=4
yiboxilong=300
min_alpha=[]
min_lost=[]
data_history=[]
data_lose_history=[]
while itation:

    Dis_dis=[]
    for j in range(len(group_number)):
        alpha=group_number[j]
        ##数据处理已完毕_下面开始pca
        #CPCA
    
        #目标数据PCA
        data_average=sum(target_data)/len(target_data)
        G=np.zeros((target_data[0].shape[0],target_data[0].shape[0]))
        for i in range(len(target_data)):
            G=G+np.matmul((target_data[i]-data_average),(target_data[i].T-data_average.T))
        G_target=G/len(target_data)
        #背景数据PCA
        data_average=sum(background_data)/len(background_data)
        G=np.zeros((background_data[0].shape[0],background_data[0].shape[0]))
        for i in range(len(background_data)):
            G=G+np.matmul((background_data[i]-data_average),(background_data[i].T-data_average.T))
        G_background=G/len(background_data)
        
        
#        down_D=20
    
        changed_Target=G_target-alpha*G_background
        Changed_eig,Changed_vec = np.linalg.eig(changed_Target)
        ##确定下降维数down_D 留下Rate的主成分
        for i in range(len(Changed_eig)):
            Changed_eig[i]=np.abs(Changed_eig[i])
        He=sum(Changed_eig)
        for i in range(len(Changed_eig)-1):
            AAA=heapq.nlargest(i, list(Changed_eig))
            if sum(AAA)/He >Rate:
                down_D=i
                break

        max_index = map(list(Changed_eig).index, heapq.nlargest(down_D, list(Changed_eig))) 
        indx=list(set(max_index))
        change_metrix=[]
        changed_target_data=[]
        for i in range(len(indx)):
            change_metrix.append(Changed_vec[:,indx[i]])
        change_metrix=np.array(change_metrix)
        for i in range(len(target_data)):
            changed_target_data.append(np.matmul(change_metrix, target_data[i]))
        Dis_dis.append(Dis_cal(changed_target_data,target_label))
        print(j)
    max_index = map(list(Dis_dis).index, heapq.nsmallest(int(group_num*rate), list(Dis_dis))) 
    indx=list(set(max_index))
    next_group_num=[]
    data_min=[]
    for i in range(len(indx)):
        next_group_num.append(group_number[indx[i]])
        data_min.append(group_number[indx[i]])
    Dis_sum=0
    for i in range(len(Dis_dis)):
        Dis_sum=Dis_sum+(Dis_dis[i]-min(Dis_dis))
    for i in range(len(data_min)):
        data_lost=Dis_dis[indx[i]]
        data_A=A*((data_lost-min(Dis_dis))+yiboxilong)/(Dis_sum+yiboxilong)
        data_next=np.random.uniform(data_min[i]-data_A,data_min[i]+data_A,5)##
        for k in range(len(data_next)):
            next_group_num.append(data_next[k])
    
    data_add=np.random.uniform(0,30,5)
    for i in range(len(data_add)):
        next_group_num.append(data_add[i])
    
    min_alpha.append(group_number[Dis_dis.index(min(Dis_dis))])
    min_lost.append(min(Dis_dis))
    data_history.append(group_number)
    data_lose_history.append(Dis_dis)
    
    group_number=next_group_num
    itation=itation-1
    
    print(str(itation)+'!!!!!!!!!!!!!!!!!')
    
    
    
