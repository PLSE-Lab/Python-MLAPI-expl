#!/usr/bin/env python
# coding: utf-8

# In[26]:


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


# In[27]:


X=[None]*5
label=[None]*5
classifier=[None]*5
labels=np.load("../input/sublocalisation-of-protien/benchmark_pssm_label_multi.npy")
labels.shape


# In[28]:


for i in range(0,5):
    X[i]=list(np.load("../input/fork-of-pssm-label-"+str(i+1)+"/benchmark_label"+str(i+1)+"_GA.npy"))
    for j in range(0,578):
        X[i][j]=list(X[i][j])
    label[i]=list(labels[:,i])
print(len(X[0]))


# In[29]:


def other_scores(Y_test,prediction):
    n=prediction.shape[0]
    precision_score=0
    recall_score=0
    F1_score=0
    for i in range(n):
        actual_l=set(Y_test.rows[i])
        predict_l=set(prediction.rows[i])
        if len(predict_l)!=0:
            precision_score+=len(actual_l.intersection(predict_l))/len(predict_l)
        recall_score+=len(actual_l.intersection(predict_l))/len(actual_l)
        F1_score+=2*len(actual_l.intersection(predict_l))/(len(actual_l)+len(predict_l))
    
    return precision_score/n,recall_score/n,F1_score/n
    
        


# In[30]:


def label_score(Y_test,pred):
    label_count=[0]*5
    true_pred_count=[0]*5
    for j  in range(5):
        for i in range(len(pred)):
            if Y_test[i][j]==1:
                label_count[j]+=1
            if Y_test[i][j]==1 and pred[i][j]==1:
                true_pred_count[j]+=1
    pred_sum=0
    label_sum=0
    macro_accuracy=[]
    for i in range(5):
        if label_count[i]!=0:
            macro_accuracy.append(true_pred_count[i]/label_count[i])
        else:
            macro_accuracy.append('None')
        pred_sum=pred_sum+true_pred_count[i]
        label_sum=label_sum+label_count[i]
    micro_accuracy=pred_sum/label_sum
    return macro_accuracy,micro_accuracy


# In[31]:


import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import random
from sklearn.svm import SVC
5

for i in range(0,5):
    classifier[i] = SVC(kernel='rbf', C=30,gamma='auto')

from sklearn.metrics import accuracy_score
acc_svm = [[],[],[],[],[]]
rand_y= [[],[],[],[],[]]
rand_x= [[],[],[],[],[]]
X_dash=[None]*5
label_dash=[None]*5

for i in range(0,5):
    X_dash[i]=X[i].copy()
    label_dash[i]=label[i].copy()
#print("X'",len(X_dash[0])  )
k=0
while len(X_dash[0])!=0:
    k=k+1
    Random=random.randint(0,len(X_dash[0])-1)
    for i in range(0,5):
        rand_x[i].append(X_dash[i].pop(Random))
        rand_y[i].append(label_dash[i].pop(Random))
#print(len(rand_x[1]))
it=len(rand_x[0])//10

for i in range(0,5):
    rand_x[i]=np.array(rand_x[i])
    #print(rand_x.shape)
    rand_y[i]=np.array(rand_y[i])

#print(rand_y.shape)
X_train=np.array([None]*5)
Y_train=np.array([None]*5)
X_test=np.array([None]*5)
Y_test=np.array([None]*5)

pred=np.array([None]*5)


# In[34]:


from scipy.sparse import lil_matrix
from sklearn.metrics import accuracy_score,jaccard_similarity_score
acc_svm=[]
label_measure=[]
precision=[]
recall=[]
F1_score=[]
macro_accuracy=[]
micro_accuracy=[]
jac=[]
it=len(rand_x[0])//5

for fold_10 in range(0,5):
    prediction=np.zeros(it*5).reshape(it,5)
    label_mat=np.zeros(it*5).reshape(it,5)
    for i in range(0,5):
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
    
    mac,mic=label_score(label_mat,prediction)
    macro_accuracy.append(mac)
    micro_accuracy.append(mic)
    prediction=lil_matrix(prediction)
    label_mat=lil_matrix(label_mat)
    jac.append(jaccard_similarity_score(label_mat,prediction))
    acc_svm.append(accuracy_score(label_mat,prediction))
    prec,rec,F1=other_scores(label_mat,prediction)
    precision.append(prec)
    recall.append(rec)
    F1_score.append(F1)
  
    

average_macro=[0]*5
macro_accuracy=np.array(macro_accuracy)
for i in range(5):
    average_macro[i]=np.mean(macro_accuracy[:,i]),np.std(macro_accuracy[:,i])  
    
    

    


# In[35]:


print ('\n OAA: ', np.mean(acc_svm),np.std(acc_svm))
print('\n  Acc',np.mean(jac),np.std(jac))
print("\n  Precision",np.mean(precision),np.std(precision))
print("\n  Recall",np.mean(recall),np.std(recall))
print("\n F1 score ",np.mean(F1_score),np.std(F1_score))
print("\n Average micro accuracy",np.mean(micro_accuracy),np.std(micro_accuracy))
print("\n Average macro accuracy",average_macro)



    

