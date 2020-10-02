#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import scipy
from sklearn.metrics import classification_report,accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
data=pd.read_csv('/kaggle/input/creditcardfraud/creditcard.csv',sep=',')
data.head()


# In[ ]:


data.info()


# In[ ]:


data.isnull().values.any()


# In[ ]:


data['Class'].value_counts()


# In[ ]:


fraud=data[data['Class']==1]
normal=data[data['Class']==0]


# In[ ]:


fraud.Amount.describe()


# In[ ]:


normal.Amount.describe()


# In[ ]:


sns.distplot(fraud.Amount,bins=100,color='blue')


# In[ ]:


sns.distplot(normal.Amount,bins=100,color='red')


# In[ ]:


fig,ax=plt.subplots()
sns.scatterplot(x='Time',y='Amount',data=fraud,ax=ax,color='blue')
sns.scatterplot(x='Time',y='Amount',data=normal,ax=ax,color='red')


# In[ ]:


data1=data.sample(frac=0.1,random_state=1)
data1.shape


# In[ ]:


fraud=data1[data1['Class']==1]
valid=data1[data1['Class']==0]


# In[ ]:


corrmat=data1.corr()
top_corr_features=corrmat.index
plt.figure(figsize=(20,20))
sns.heatmap(data[top_corr_features].corr(),annot=True)


# In[ ]:


columns=data1.columns.tolist()
columns=[c for c in columns if c not in ['Class']]
target='Class'
state=np.random.RandomState(42)
X=data1[columns]
Y=data1[target]
X_outliers=state.uniform(low=0,high=1,size=(X.shape[0],X.shape[1]))
outlier_fraction = len(fraud)/float(len(valid))


# In[ ]:


##Define the outlier detection methods

classifiers = {
    "Isolation Forest":IsolationForest(n_estimators=100, max_samples=len(X), 
                                       contamination=outlier_fraction,random_state=state, verbose=0),
    "Local Outlier Factor":LocalOutlierFactor(n_neighbors=20, algorithm='auto', 
                                              leaf_size=30, metric='minkowski',
                                              p=2, metric_params=None, contamination=outlier_fraction),
    "Support Vector Machine":OneClassSVM(kernel='rbf', degree=3, gamma=0.1,nu=0.05, 
                                         max_iter=-1)
   
}


# In[ ]:


n_outliers = len(fraud)
for i, (clf_name,clf) in enumerate(classifiers.items()):
    #Fit the data and tag outliers
    if clf_name == "Local Outlier Factor":
        y_pred = clf.fit_predict(X)
        scores_prediction = clf.negative_outlier_factor_
    elif clf_name == "Support Vector Machine":
        clf.fit(X)
        y_pred = clf.predict(X)
    else:    
        clf.fit(X)
        scores_prediction = clf.decision_function(X)
        y_pred = clf.predict(X)
    #Reshape the prediction values to 0 for Valid transactions , 1 for Fraud transactions
    y_pred[y_pred == 1] = 0
    y_pred[y_pred == -1] = 1
    n_errors = (y_pred != Y).sum()
    # Run Classification Metrics
    print("{}: {}".format(clf_name,n_errors))
    print("Accuracy Score :")
    print(accuracy_score(Y,y_pred))
    print("Classification Report :")
    print(classification_report(Y,y_pred))


# In[ ]:


observations
Isolation Forest detected 73 errors versus Local Outlier Factor detecting 97 errors vs. SVM detecting 8516 errors
Isolation Forest has a 99.74% more accurate than LOF of 99.65% and SVM of 70.09

