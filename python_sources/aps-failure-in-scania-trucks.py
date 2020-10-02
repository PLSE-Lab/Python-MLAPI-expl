#!/usr/bin/env python
# coding: utf-8

# >     Predicting Air pressure system failures in Scania trucks
# The dataset consists of data collected from heavy Scania trucks in everyday usage. The system in focus is the Air Pressure system (APS) which generates pressurized air that is utilized in various functions in a truck, such as braking and gear changes. The datasets' positive class consists of component failures for a specific component of the APS system. The negative class consists of trucks with failures for components not related to the APS.
# The training set contains 60000 examples in total in which 59000 belong to the negative class and 1000 positive class. The test set contains 16000 examples. There are 171 attributes per record.

# In[ ]:


#Import required libraries
import numpy as np 
import pandas as pd
import seaborn as sns
import lightgbm as lgb
import scikitplot as skplot
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import warnings
warnings.filterwarnings('ignore')


# Load the datasets

# In[ ]:


df_train=pd.read_csv("../input/aps-failure-at-scania-trucks-data-set/aps_failure_training_set_processed_8bit.csv")
df_test=pd.read_csv("../input/aps-failure-at-scania-trucks-data-set/aps_failure_test_set_processed_8bit.csv")
print("Shape of the datasets...")
print("Shape of train dataset:",df_train.shape)
print("Shape of the test dataset:",df_test.shape)


# In[ ]:


df_train.head()


# In[ ]:


df_test.head()


# In[ ]:


#Concanate train & test dataset
dataset=pd.concat(objs=[df_train.drop(columns=["class"]),df_test.drop(columns=["class"])],axis=0)
dataset.shape


# In[ ]:


dataset.head()


# In[ ]:


dataset.info()


# In[ ]:


dataset.describe()


# Detecting missing  values

# In[ ]:


total_miss_values=dataset.isna().sum().sort_values(ascending=False)
total_miss_values


# It seems no missing values are present in the dataset.

# In[ ]:


#Encode labels to 0 & 1
le=LabelEncoder()
df_train["class"]=le.fit_transform(df_train["class"])
df_test["class"]=le.transform(df_test["class"])
print("Target labels are :",le.classes_);


# Target labels, 0- Negative class &  1-positve class
#                    

# Visualzing the distribution of a dataset

# > Data visulization with Correlation matrix

# In[ ]:


#Correlation matrix
df_train.corr()


# In[ ]:


#plot correlation matrix
f=plt.figure(figsize=(15,15))
ax=f.add_subplot(111)
cax=ax.matshow(df_train.corr(),interpolation='nearest')
f.colorbar(cax)
plt.title('Correlation matrix',fontsize=15)
plt.show();


# The above plot clearly shows that, in 171 attributes, the atributes numbered from 70 to 80 are negatively correlated with other attributes.

# In[ ]:


#Train dataset target labels distribution
plt.figure(figsize=(15,8))
sns.distplot(df_train["class"]);


# This clearly shows that the target labels in train dataset are imbalanced i.e., negative samples are more than the positve samples. This dataset needs to be balanced to get accurate prediction results.

# > Split the datasets

# In[ ]:


# Train dataset
X_train=df_train.drop(columns=["class"])
y_train=df_train["class"]

#Test dataset
X_test=df_test.drop(columns=["class"])
y_test=df_test["class"]


# > SMOTE (Synthetic minority over sampling technique) Algorithm        
#   - Used to balance the classes in the datasets

# In[ ]:


sm=SMOTE(random_state=42)
#Resample the train dataset
X_train,y_train=sm.fit_sample(X_train,y_train)
print("Resampled train dataset shape :",X_train.shape,y_train.shape);


# Visualzing the distributon of the resampled dataset

# In[ ]:


# distribution of classes in train dataset
plt.figure(figsize=(15,8))
sns.distplot(y_train);


# In[ ]:


# distribution of classes in test dataset
plt.figure(figsize=(15,8))
sns.distplot(y_test);


# After the resampling, the classes in train dataset are balanced.

# > Visualizing the distribution of attributes/features

# In[ ]:


df_train.hist(figsize=(16,35),bins=10,xlabelsize=8,ylabelsize=8);


# In above plots, there are 24 attributes share a similar distribution to the 'class' distribution.

# > Build Logistic Regresson model

# In[ ]:


lr=LogisticRegression()
lr.fit(X_train,y_train);


# In[ ]:


#predict on test data
y_pred_lr=lr.predict(X_test)


# In[ ]:


#confusion matrix
cm=confusion_matrix(y_test,y_pred_lr,labels=[0,1])
cm


# In[ ]:


#Plot confusion matrix
skplot.metrics.plot_confusion_matrix(y_test,y_pred_lr,figsize=(15,8),title='Confusion matrix for Logistic Regression model')


# Out of 16000 test instances, 97.08% of instances are classified correctly & remaining 2.91% instances are misclassified.

# > Build a LghtGBM model

# In[ ]:


#load datasets in lgb formate
train_data=lgb.Dataset(X_train,label=y_train)


# In[ ]:


#Create the validation dataset
X_train,X_val,y_train,y_val=train_test_split(X_train,y_train,test_size=0.1,random_state=42)
print("The shape of X_train is : {} & the shape of y_train is : {}".format(X_train.shape,y_train.shape))
print("The shape of X_val is : {} & the shape of y_val is : {}".format(X_val.shape,y_val.shape))
validation_data=lgb.Dataset(X_val,label=y_val)


# In[ ]:


#set parameters for training
params={ 'num_leaves':145,
        'object':'binary',
        'metric':['auc','binary_logloss']
       }


# In[ ]:


#Train the model
num_round=20
lgb_model=lgb.train(params,train_data,num_round,valid_sets=validation_data,early_stopping_rounds=5)


# In[ ]:


#Prediction on unseen dataset
y_pred=lgb_model.predict(X_test,num_iteration=lgb_model.best_iteration)>0.5


# In[ ]:


#Confusion matrix
cm=confusion_matrix(y_test,y_pred,labels=[0,1])
cm


# In[ ]:


#Plot confusion matrix
skplot.metrics.plot_confusion_matrix(y_test,y_pred,figsize=(15,8),title='Confusion matrix for LGBM model')
plt.show()


# Out of 16000 test instances, 98.56% of instances are classified correctly & remaining 1.43% instances are misclassified.

# The LGBM model performed well on test dataset than logistic regression model. The performance of LGBM model improved further by optimizing the hyper parameters.
