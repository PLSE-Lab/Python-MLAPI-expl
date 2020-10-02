#!/usr/bin/env python
# coding: utf-8

# # <div style=' background-color: #ddffff; border-left: 10px solid #2196F3; padding: 20px;'> Playing with the data! </div>

# ### <div style=' color: white; background-color: #2D93D5; border-left: 10px solid #014F99; padding: 20px;'> Simplified Problem Statement </div>
# - The problem is -
#     - From a labelled dataset of 3662 retina images. Teach / Train a Machine Learning model to correctly identify the severity of diabetic retinopathy on a scale of 0 to 4.
# 

# In[ ]:


import re
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.naive_bayes import MultinomialNB
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
import lightgbm as lgb
import xgboost as xg

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold


# ### <div style=' color: white; background-color: #2D93D5; border-left: 10px solid #014F99; padding: 20px;'> Data Description </div>
# - **Total entries in training set - 3662**
#     - *0 - No DR - 1805 (49.29%)*
#     - *1 - Mild - 999 (27.28%)*
#     - *2 - Moderate - 370 (10.10%)*
#     - *3 - Severe - 295 (8.05%)*
#     - *4 - Proliferative DR - 193 (5.27%)*

# Note - All files are of format .png which are being resized to (512 x 512) dimensions and being read as grayscale images.

# In[ ]:


#Prepping X_train
#Note - 512 x 512 - 7.2 GB

Img_size = 512  
X0 = []
base_pth = '../input/train_images/'

for filename in os.listdir('../input/train_images/'):
    ds = cv2.imread(str(base_pth+filename),0)
    b0 = cv2.resize(ds,(Img_size,Img_size))
    X0.append(b0)


# In[ ]:


#Prepping Y_train

df_train = pd.read_csv('../input/train.csv')
y = []

for filename in os.listdir('../input/train_images/'):
    idz = str(filename)[:-4]
    for idx,each in enumerate(df_train['id_code']):
        if idz == each:
            y.append(int(df_train.iloc[idx,1]))


# In[ ]:


X = np.array([i[0] for i in X0]).reshape((-1,512))
y = np.array(y)


# ### <div style=' color: white; background-color: #2D93D5; border-left: 10px solid #014F99; padding: 20px;'>Survey</div>

# In[ ]:


def conf_matrix(y_pred,y_actual):
        
    cm = confusion_matrix(y_actual, y_pred)
    acc = (y_actual == y_pred).sum()/len(y_actual)

    return cm,acc


# In[ ]:


def print_metrics(cm,acc):
    print("Confusion Matrix - ")
    print(cm)
    print("Accuracy")
    print(acc)


# In[ ]:


def train_test_func(algo_init_model):
    
    #Add k_Fold step here
    k = 5
    kf = KFold(n_splits=k)
    
    avg_cm = np.zeros((5,5))
    avg_acc = 0
    
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        #Training
        y_pred = algo_init_model.fit(X_train, y_train)

        #Prediction
        y_pred_val = y_pred.predict(X_test)

        #Metrics
        cm, acc = conf_matrix(y_pred_val,y_test)
        avg_acc += acc
        avg_cm += cm

    print_metrics(avg_cm/k,avg_acc/k)


# #### <div style=' color: black; background-color: #C7D86F; border-left: 10px solid #F7C407; padding: 20px;'>Multi-Nomial Naive Bayes</div>

# In[ ]:


#Trying on Naive Bayes
gnb = MultinomialNB()
train_test_func(gnb)


# #### <div style=' color: black; background-color: #C7D86F; border-left: 10px solid #F7C407; padding: 20px;'>Decision Tree</div>

# In[ ]:


#Trying on Decision Tree
clf = tree.DecisionTreeClassifier()
train_test_func(clf)


# #### <div style=' color: black; background-color: #C7D86F; border-left: 10px solid #F7C407; padding: 20px;'>Logistic Regression</div>

# In[ ]:


#Trying on Logistic Regression
clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial')
train_test_func(clf)


# #### <div style=' color: black; background-color: #C7D86F; border-left: 10px solid #F7C407; padding: 20px;'>Random Forest</div>

# In[ ]:


#Trying on Random Forest
clf = RandomForestClassifier(n_estimators=100, max_depth=2,random_state=0)
train_test_func(clf)


# #### <div style=' color: black; background-color: #C7D86F; border-left: 10px solid #F7C407; padding: 20px;'>AdaBoost</div>

# In[ ]:


#Trying on AdaBoost
clf = AdaBoostClassifier(n_estimators=50,learning_rate=1,random_state=0)
train_test_func(clf)


# #### <div style=' color: black; background-color: #C7D86F; border-left: 10px solid #F7C407; padding: 20px;'>KNN</div>

# In[ ]:


#Trying on KNN
neigh = KNeighborsClassifier(n_neighbors=5)
train_test_func(neigh)


# #### <div style=' color: black; background-color: #C7D86F; border-left: 10px solid #F7C407; padding: 20px;'>LightGBM</div>

# In[ ]:


#Trying on LightGBM

params = {}
params['learning_rate'] = 0.003
params['boosting_type'] = 'gbdt'
params['objective'] = 'multiclass'
params['metric'] = 'multi_logloss'
params['sub_feature'] = 0.5
params['num_leaves'] = 30
params['min_data'] = 50
params['max_depth'] = 20
params['num_class'] = 5

k = 5
kf = KFold(n_splits=k)
    
avg_cm = np.zeros((5,5))
avg_acc = 0
    
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
        
    #Training
    d_train = lgb.Dataset(X_train, label=y_train)
    y_pred = lgb.train(params, d_train, 100)

    #Prediction
    y_pred_val0 = y_pred.predict(X_test)
    print(y_pred_val0[0])
    
    y_pred_val = []

    for x in y_pred_val0:
        print(np.argmax(x))
        y_pred_val.append(np.argmax(x))
    
    #Metrics
    cm, acc = conf_matrix(y_pred_val,y_test)
    avg_acc += acc
    avg_cm += cm
print_metrics(avg_cm/k,avg_acc/k)


# #### <div style=' color: black; background-color: #C7D86F; border-left: 10px solid #F7C407; padding: 20px;'>XGBoost</div>

# In[ ]:


#Trying on XGBoost
xgbt = xg.XGBClassifier()
train_test_func(xgbt)


# 
