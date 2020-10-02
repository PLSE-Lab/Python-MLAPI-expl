#!/usr/bin/env python
# coding: utf-8

# In[2]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[5]:


X=[None]*4
label=[None]*4
classifier=[None]*4
labels=np.load("../input/sublocalisation-of-protien/novel_pssm_label_multi.npy")
labels.shape


# In[6]:


for i in range(0,4):
    X[i]=list(np.load("../input/novel-fork-of-pssm-label-"+str(i+1)+"/novel_label"+str(i+1)+"_GA.npy"))
    for j in range(0,122):
        X[i][j]=list(X[i][j])
    label[i]=list(labels[:,i])
print(len(X[0]))


# In[7]:


def score(it,prediction_sparse,labels_sparse):
    acc_svm=0
    label_count_multi=0
    correct_pred_multi=0
    label_single=0
    correct_single=0
    for i in range(0,it):
        L=set(prediction_sparse.rows[i])
        M=set(labels_sparse.rows[i])
        if len(M)!=0:
            acc_svm=acc_svm+len(L.intersection(M))/len(L.union(M))
        else:
            print("HOw can it be")
        
        if len(M)==2:
            label_count_multi+=1
            if len(L.intersection(M))==2:
                correct_pred_multi+=1
                
        if len(M)==1:
            label_single+=1
            if len(L.intersection(M))==1:
                correct_single+=1
        
            #print(pred.toarray())
    acc_svm=acc_svm/it
    multi_acc='NaN'
    single_acc='NaN'
    if label_count_multi!=0:
        multi_acc=correct_pred_multi/label_count_multi
    if label_single!=0:
        single_acc=correct_single/label_single
    return [acc_svm,multi_acc,single_acc]


# In[8]:


def label_score(pred,Y_test):
    label_count=[0]*4
    true_pred_count=[0]*4
    for j  in range(4):
        for i in range(len(pred)):
            if Y_test[i][j]==1:
                label_count[j]+=1
            if Y_test[i][j]==1 and pred[i][j]==1:
                true_pred_count[j]+=1
    pred_sum=0
    label_sum=0
    macro_accuracy=[]
    for i in range(4):
        if label_count[i]!=0:
            macro_accuracy.append(true_pred_count[i]/label_count[i])
        else:
            macro_accuracy.append('None')
        pred_sum=pred_sum+true_pred_count[i]
        label_sum=label_sum+label_count[i]
    micro_accuracy=pred_sum/label_sum
    return macro_accuracy,micro_accuracy


# In[9]:


import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import random
from sklearn.svm import SVC


for i in range(0,4):
    classifier[i] = SVC(kernel='rbf', C=10,gamma='auto')

from sklearn.metrics import accuracy_score
acc_svm = [[],[],[],[]]
rand_y= [[],[],[],[]]
rand_x= [[],[],[],[]]
X_dash=[None]*4
label_dash=[None]*4

for i in range(0,4):
    X_dash[i]=X[i].copy()
    label_dash[i]=label[i].copy()
#print("X'",len(X_dash[0])  )
k=0
while len(X_dash[0])!=0:
    k=k+1
    Random=random.randint(0,len(X_dash[0])-1)
    for i in range(0,4):
        rand_x[i].append(X_dash[i].pop(Random))
        rand_y[i].append(label_dash[i].pop(Random))
#print(len(rand_x[1]))
it=len(rand_x[0])//10

for i in range(0,4):
    rand_x[i]=np.array(rand_x[i])
    #print(rand_x.shape)
    rand_y[i]=np.array(rand_y[i])

#print(rand_y.shape)
X_train=np.array([None]*4)
Y_train=np.array([None]*4)
X_test=np.array([None]*4)
Y_test=np.array([None]*4)

pred=np.array([None]*4)


# In[10]:


from scipy.sparse import lil_matrix
len(rand_x[3])
pred
acc_svm=[]
label_measure=[]
it=len(rand_x[0])//10
multi_acc=[]
single_acc=[]
for fold_10 in range(0,10):
    prediction=np.zeros(it*4).reshape(it,4)
    label_mat=np.zeros(it*4).reshape(it,4)
    for i in range(0,4):
        X_train[i] = np.concatenate((rand_x[i][:fold_10*it],rand_x[i][((fold_10+1)*it):]),axis=0)
        Y_train[i]=np.concatenate((rand_y[i][:fold_10*it],rand_y[i][((fold_10+1)*it):]),axis=0)
        X_test[i]=rand_x[i][fold_10*it:(fold_10+1)*it]
        #print(X_test.shape)
        Y_test[i]=rand_y[i][fold_10*it:(fold_10+1)*it]
#         print("X",X_train[i].shape)
#         print("Y",Y_train[i].shape)
        classifier[i].fit(X_train[i], Y_train[i])
        pred[i] = classifier[i].predict(X_test[i])
      
        prediction[:,i]=pred[i]
        label_mat[:,i]=Y_test[i]
    
    label_measure.append(list(label_score(prediction,label_mat)))
    prediction_sparse=lil_matrix(prediction)
    labels_sparse=lil_matrix(label_mat)
    temp=score(it,prediction_sparse,labels_sparse)
    acc_svm.append(temp[0])
    multi_acc.append(temp[1])
    single_acc.append(temp[2])
    
    print("fold ",fold_10+1,"accuracy ",acc_svm[fold_10],'Macro accuracy',label_measure[fold_10][0],"Micro Accuracy",label_measure[fold_10][1])
    print("Multi_acc",multi_acc[fold_10],"Single_acc",single_acc[fold_10])
print("mean accuracy",np.mean(acc_svm))

    

    


# In[11]:


Average_micro_measure=0
for i in range(10):
    Average_micro_measure+=label_measure[i][1]
Average_micro_measure/10   


# In[12]:



from scipy.sparse import lil_matrix
len(rand_x[3])
pred
acc_svm=[]
label_measure=[]
multi_acc=[]
single_acc=[]
it=len(rand_x[0])//5
for fold_10 in range(5):
    prediction=np.zeros(it*4).reshape(it,4)
    label_mat=np.zeros(it*4).reshape(it,4)
    for i in range(0,4):
        X_train[i] = np.concatenate((rand_x[i][:fold_10*it],rand_x[i][((fold_10+1)*it):]),axis=0)
        Y_train[i]=np.concatenate((rand_y[i][:fold_10*it],rand_y[i][((fold_10+1)*it):]),axis=0)
        X_test[i]=rand_x[i][fold_10*it:(fold_10+1)*it]
        #print(X_test.shape)
        Y_test[i]=rand_y[i][fold_10*it:(fold_10+1)*it]
#         print("X",X_train[i].shape)
#         print("Y",Y_train[i].shape)
        classifier[i].fit(X_train[i], Y_train[i])
        pred[i] = classifier[i].predict(X_test[i])
      
        prediction[:,i]=pred[i]
        label_mat[:,i]=Y_test[i]
    
    label_measure.append(list(label_score(prediction,label_mat)))
    prediction_sparse=lil_matrix(prediction)
    labels_sparse=lil_matrix(label_mat)
    temp=score(it,prediction_sparse,labels_sparse)
    acc_svm.append(temp[0])
    multi_acc.append(temp[1])
    single_acc.append(temp[2])
    
    print("fold ",fold_10+1,"accuracy ",acc_svm[fold_10],'Macro accuracy',label_measure[fold_10][0],"Micro Accuracy",label_measure[fold_10][1])
    print("Multi_acc",multi_acc[fold_10],"Single_acc",single_acc[fold_10])
print("mean accuracy",np.mean(acc_svm))

    

Average_micro_measure=0
for i in range(5):
    Average_micro_measure+=label_measure[i][1]
print("Avergae_micro accuracy",Average_micro_measure/5)
    


# In[ ]:





# In[ ]:




