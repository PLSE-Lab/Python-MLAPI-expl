#!/usr/bin/env python
# coding: utf-8

# **Basic EDA**
# * Logistic Regression
# * GridsearchCV,RandomizedsearchCV
# * GaussianNB
# * Support Vector Machine
# 
# **Performance measurements**
# * Classification report,Confusion Matrix, precision, recall, F1 score, roc_auc_score, accuracy_score

# In[1]:


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
import warnings
warnings.filterwarnings('ignore')


# Import some common libraries used down the line

# In[2]:


import seaborn as sns
import matplotlib.pyplot as plt


# loading the dataset into pandas dataframe

# In[3]:


data = pd.read_csv("../input/mushrooms.csv")
data.head()


# In[4]:


data.shape


# There are many features with non-numeric or categorical data. ML can be applied only on numeric values. so we have to convert it to numerical values for which you can use label encoder.

# In[5]:


#Let's check for missing null values
data.info()


# In[6]:


#let's see how many categories are available for predictions
data['class'].unique()


# In[7]:


#converting categorical data (including Class column) into numeric using LabelEncoder
from sklearn.preprocessing import LabelEncoder
lablencoder = LabelEncoder()

for col in data.columns:
    data[col] = lablencoder.fit_transform(data[col])

data.head()


# In[8]:


#we already saw how many types of classes we need to predict. Just 'p','e' in this case.
#let's see how many observations are related to each class
print(data['class'].value_counts())


# Pretty Good!! we have balanced set of observations for each class

# In[9]:


#visualizing the same using seaborn
sns.countplot(data['class'])


# In[10]:


#seperate the features & response to feed to algorithms
X = data.iloc[:,1:23]
y = data.iloc[:,0]


# In[11]:


#check the correlation of the data
data.corr()


# Many algorithms are sensitive to the scale of Features. It's better to standardize the data before we feed it to algos.

# In[12]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X=scaler.fit_transform(X)
X


# split the data into training & test set

# In[13]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2, random_state=42)


# ![](http://)**1. Training Default Logistic Regression Model**

# In[14]:


from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train,y_train)


# In[15]:


from sklearn.metrics import accuracy_score
y_pred = logreg.predict(X_test)
print(accuracy_score(y_test,y_pred))


# we got 95% accuracy with default Logistic Regression Model. Let's perform PCA on the data & see if that improves the accuracy score

# In[16]:


from sklearn.decomposition import PCA
pca = PCA(n_components=17)
pca.fit_transform(X)


# In[17]:


#split the data into training & test set based on PCA data
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2, random_state=42)


# In[18]:


logreg1 = LogisticRegression()
logreg1.fit(X_train,y_train)
y_pred1 = logreg1.predict(X_test)

print(accuracy_score(y_test,y_pred1))


# so PCA does not have any effect on Logistict Regression model.
# **Let's fine tune the Model by changing its hyperparameters**

# In[19]:


from sklearn.linear_model import LogisticRegression
from sklearn import metrics

LR_model = LogisticRegression()

tuned_parameters = {'C':[0.001, 0.01, 0.1, 1, 10, 100, 1000], 'penalty':['l1','l2']}


# In[20]:


from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(LR_model,tuned_parameters, cv=10)

grid_search.fit(X_train, y_train)


# In[21]:


print(grid_search.best_params_)


# In[22]:


LR_model = LogisticRegression(C=1000, penalty='l1')
LR_model.fit(X_train,y_train)

y_train_pred = LR_model.predict(X_train)
print('accuracy on Train data',accuracy_score(y_train,y_train_pred))

y_pred = LR_model.predict(X_test)
print('accuracy on Test data',accuracy_score(y_test,y_pred))


# **so we have got an accuracy of 96% with Fine Tuned Logistic Regression**
# 
# Let's see some other models now

# **Gaussian Naive Bayes**

# In[23]:


from sklearn.naive_bayes import GaussianNB
GNB_model = GaussianNB()

GNB_model.fit(X_train, y_train)


# In[24]:


y_pred = GNB_model.predict(X_test)

print(accuracy_score(y_test,y_pred))
GNB_model.score(X_test,y_pred)


# Gaussian Naive Bayes model has 92% accuracy which is less than Logistic Regression Model

# In[25]:


print("Number of mislabeled points from %d points : %d"
      % (X_test.shape[0],(y_test!= y_pred).sum()))


# **Support Vector Machines**

# In[26]:


from sklearn.svm import SVC
svc_model = SVC()

params = {
    'C':[1, 10, 100,500, 1000], 'kernel':['linear','rbf'],
    'C': [1, 10, 100,500, 1000], 'gamma': [1,0.1,0.01,0.001, 0.0001], 'kernel': ['rbf'],
}


# In[27]:


from sklearn.model_selection import RandomizedSearchCV
rnd_search = RandomizedSearchCV(svc_model,params,cv=10,scoring='accuracy',n_iter=20)
rnd_search.fit(X_train, y_train)


# In[28]:


print(rnd_search.best_score_)


# In[29]:


y_train_pred = rnd_search.predict(X_train)
print('accuracy on Train data',accuracy_score(y_train,y_train_pred))
y_pred = rnd_search.predict(X_test)
print('accuracy on Test data',accuracy_score(y_test,y_pred))


# **We have got 100% accuracy using this SVM model**
# 
# Let's see its precision, recall,f1score,roc_auc_scores to confirm it classifies perfectly

# In[30]:


from sklearn import metrics


# In[31]:


confusion_matrix = metrics.confusion_matrix(y_test,y_pred)
confusion_matrix


# False Positive Rate, False Negative Rate are 0 which means it predicts/classified perfectly.
# 
# True Positive rate = 843 (which means there are 843 observations with Class = 0, posionous)
# True Negative rate = 782 (which means there are 782 observations with Class = 1, edible)
# 
# check the same below

# In[32]:


y_test.value_counts()


# In[33]:


auc_roc = metrics.classification_report(y_test,y_pred)
auc_roc


# In[34]:


metrics.roc_auc_score(y_test,y_pred)

