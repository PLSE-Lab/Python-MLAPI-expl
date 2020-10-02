#!/usr/bin/env python
# coding: utf-8

# ### Working with Imbalanced Datasets ### 
# We can perform undersampling or oversampling over Imbalanced Datasets so that our models can be accurate as well as give a good recall and precision score.
# 
# However, performing cross validation after sampling the dataset may not give the correct model. Hence, sampling needs to be performed during cross validation.
# 
# Here, we have created a Pipeline with 2 steps. First step performs sampling using **NearMiss** or **SMOTE** technique and second step is the **Logistic Regression**. Inorder to get the best parameters of the LR, we have used the **GridSearch** cross validation approach and used **F1** as the scoring priority. For splitting the training dataset during cross validation, we have used **StratifiedShuffleSplit**
# 
# Inside GridSearch, we have passed the Pipeline as the estimator. Later on, once we get the best parameters for LR, we will continue to use the Pipeline to fit with the training data and predict with the test data.
# 
# For oversampling, we have figured out the threshold to be used to get the best F1 score and then produced the predictions based on the selected threshold value.

# In[12]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold,StratifiedShuffleSplit
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from imblearn.under_sampling import NearMiss
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, classification_report
from sklearn.metrics import precision_recall_curve, roc_curve, roc_auc_score,auc,confusion_matrix

#Import learning models
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[13]:


df = pd.read_csv('../input/creditcard.csv')


# In[14]:


df.info()


# #### Observation: There are no missing values and all columns are of float type

# In[15]:


df.head(5)


# In[16]:


df.describe()


# **Exploring the Data**

# In[17]:


sns.distplot(df['Class'],kde=False,bins=30) 


# In[18]:


print("Number of non Fraud samples:",df[df["Class"]==0].shape[0])
print("Number of Fraud samples:",df[df["Class"]==1].shape[0])


# **This is an example of Imbalanced Datasets.**

# **Separating data from label**

# In[19]:


X = df.drop(["Class"],axis=1)
y = df["Class"]


# #### Scaling the Amount and Time columns

# In[20]:


std = StandardScaler()
X_new = pd.DataFrame(std.fit_transform(X))
X_new.columns = X.columns
X_new.head(5)


# In[21]:


#Splitting dataset into training and test
X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.2, random_state=101)


# ### **As the table has Imbalanced Labels, we will perform undersampling and oversampling to build a suitable model for prediction**

# In[22]:


#Define a iterable for Grid Search Cross Validation
skf = StratifiedShuffleSplit(n_splits=3)


# ### **Phase 1: Undersampling**

# We will perform cross validation using GridSearchCV to get the best model. Since this is imbalanced set, we will perform undersampling using Near Miss technique first. We will create a pipeline with NearMiss and Logistic Regression model and provide the pipeline as as estimator in the GridSearchCV model. For cross validation iterable, we will provide the StratifiedShuffleSplit model as parameter.
# 
# After running GridSearchCV on the training data, we will get the pipeline with best parameters of Logistic Regression.

# In[23]:


#Grid Search to determine best parameters for the learning models
log_reg_params = {"logisticregression__penalty": ['l1', 'l2'], 'logisticregression__C': [ 0.01, 0.1, 1, 10, 100]} 
#knears_params = {"n_neighbors": list(range(2,5,1)), 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']}
#svc_params = {'C': [0.5, 0.7, 0.9, 1], 'kernel': ['rbf', 'poly', 'sigmoid', 'linear']}
#tree_params = {"criterion": ["gini", "entropy"],"max_depth": list(range(2,4,1)), "min_samples_leaf": list(range(5,7,1))}

log_reg = LogisticRegression(solver='liblinear')

#Created Pipeline to perform undersampling using Near Miss technique and then fit the Learning model
my_pipeline_undersample = make_pipeline(NearMiss(sampling_strategy='majority'),log_reg)
log_gridcv_model = GridSearchCV(my_pipeline_undersample,log_reg_params,scoring='f1',cv=skf)
log_gridcv_model.fit(X_train,y_train)
log_gridcv_model_best = log_gridcv_model.best_estimator_
log_gridcv_model_best


# Once we get the best model from cross validation, we will run the model with training data and check the Precision and Recall scores. Later, we will plot the precision vs recall graph and the ROC curve to visualize the model's performance.

# In[24]:


#Calculate training recall and precision
log_gridcv_model_best.fit(X_train,y_train)
pred_train = log_gridcv_model_best.predict(X_train)
pred_prob_train = log_gridcv_model_best.predict_proba(X_train)
print("Training Recall:",recall_score(y_train,pred_train))
print("Training Precision:",precision_score(y_train,pred_train))


# In[25]:


#Function to Create Precision Recall Curve
def create_precision_recall_curve(y_train,prob):
    precision, recall, threshold = precision_recall_curve(y_train,prob)
    plt.step(recall, precision, color='b', alpha=0.2,where='post')
    plt.fill_between(recall, precision, alpha=0.2, color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])


# In[26]:


#Create precision recall curve for training result
create_precision_recall_curve(y_train,pred_prob_train[:,1])


# In[27]:


#Function to Create ROC curve using the training result
def create_roc_curve(label,result):
    fpr, tpr, _ = roc_curve(label,result)
    plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.2f)' % auc(fpr, tpr))
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()


# In[28]:


#Create ROC curve for training result
create_roc_curve(y_train,pred_train)


# In[29]:


#Calculate test recall and precision
pred_test = log_gridcv_model_best.predict(X_test)
pred_prob_test = log_gridcv_model_best.predict_proba(X_test)
print("Test Recall:",recall_score(y_test,pred_test))
print("Test Precision:",precision_score(y_test,pred_test))


# In[30]:


#Create precision recall curve for test result
create_precision_recall_curve(y_test,pred_prob_test[:,1])


# In[31]:


#Create ROC curve for test result
create_roc_curve(y_test,pred_test)


#  ## **Phase-2 - Oversampling**

# We will perform cross validation using GridSearchCV to get the best model. Since this is imbalanced set, we will perform oversampling using SMOTE technique first. We will create a pipeline with SMOTE and Logistic Regression model and provide the pipeline as as estimator in the GridSearchCV model. For cross validation iterable, we will provide the StratifiedShuffleSplit model as parameter.
# 
# After running GridSearchCV on the training data, we will get the pipeline with best parameters of Logistic Regression.

# In[32]:


#Created Pipeline to perform oversampling using SMOTE technique and then fit the Learning model
my_pipeline_oversample = make_pipeline(SMOTE(sampling_strategy='minority'),log_reg)
log_gridcv_oversample_model = GridSearchCV(my_pipeline_oversample,log_reg_params,scoring='f1',cv=skf)
log_gridcv_oversample_model.fit(X_train,y_train)
log_gridcv_oversample_model_best = log_gridcv_oversample_model.best_estimator_
log_gridcv_oversample_model_best


# Once we get the best model from cross validation, we will run the model with training data and check the Precision and Recall scores. Later, we will plot the precision vs recall graph and the ROC curve to visualize the model's performance.

# In[33]:


#Calculate training recall and precision
log_gridcv_oversample_model_best.fit(X_train,y_train)
pred_train = log_gridcv_oversample_model_best.predict(X_train)
pred_prob_train = log_gridcv_oversample_model_best.predict_proba(X_train)
print("Training Recall:",recall_score(y_train,pred_train))
print("Training Precision:",precision_score(y_train,pred_train))


# In[34]:


#Create precision recall curve for training result
create_precision_recall_curve(y_train,pred_prob_train[:,1])


# We need to find a threshold which will give the best results for precision, recall tradeoff

# In[35]:


threshold = [0.1,0.2,0.3,0.4,0.6,0.7,0.8,0.9]
f1_best = f1_score(y_train,pred_train)
threshold_best = 0.5
pred_best = pred_train
for threshold_value in threshold:
    pred_new_oversample = [1 if pred>threshold_value else 0 for pred in pred_prob_train[:,1]]
    f1 = f1_score(y_train,pred_new_oversample)
    if f1>f1_best:
        f1_best = f1
        threshold_best = threshold_value
        pred_best = pred_new_oversample
print("Best F1 score:",f1_best)
print("Best threshold:",threshold_best)
print("Best Recall:",recall_score(y_train,pred_best))
print("Best Precision:",precision_score(y_train,pred_best))


# In[36]:


#Create ROC curve for training result
create_roc_curve(y_train,pred_best)


# In[37]:


#Calculate test recall and precision using the threshold_best
pred_prob_test = log_gridcv_oversample_model_best.predict_proba(X_test)
pred_test = [1 if pred>threshold_best else 0 for pred in pred_prob_test[:,1]]
recall_oversample_test = recall_score(y_test,pred_test)
precision_oversample_test = precision_score(y_test,pred_test)
print("Test Recall:",recall_oversample_test)
print("Test Precision:",precision_oversample_test)


# In[38]:


#Create precision recall curve for test result
create_precision_recall_curve(y_test,pred_prob_test[:,1])


# In[39]:


#Create ROC curve for test result
create_roc_curve(y_test,pred_test)


# ##### Following results we got after performing UnderSampling and OverSampling before running the Logistic Regression model on the test set
# ##### Phase 1 - UnderSampling and Logistic Regression
# * Recall - 90%
# * Precision - 1%
# 
# ##### Phase 2 - OverSampling and Logistic Regression
# * Recall - 85%
# * Precision - 20%

# In[ ]:




