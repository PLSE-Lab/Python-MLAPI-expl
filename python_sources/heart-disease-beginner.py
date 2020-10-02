#!/usr/bin/env python
# coding: utf-8

# Import necessary models

# In[ ]:


import numpy as np
import pandas as pd


# Read data file

# In[ ]:


data= pd.read_csv("../input/heart.csv") 


# **DATA ANALYZE**

# data.head() to find the first 5 columns
# data.info() to find the summary of datas
# data.isnull().sum() to find the missing value of data
# data.dtypes to find the type of columns 
# 
# 

# In[ ]:


data.head()


# In[ ]:


data.info()


# In[ ]:


data.isnull().sum()


# In[ ]:


data.dtypes


# 1. We have 3 categorical columns: cp, exang, slope. And we should deal with them

# First, we create dummies value

# In[ ]:


cp=pd.get_dummies(data['cp'],prefix='cp', drop_first= True)
exang=pd.get_dummies(data['exang'],prefix='exang', drop_first= True)
slope=pd.get_dummies(data['slope'],prefix='slope', drop_first=True)


# we then add the dummy values to the data

# In[ ]:


new_data= pd.concat([data,cp,exang,slope], axis=1)
new_data.head()


# we then drop the original values from the data

# In[ ]:


new_data.drop(['cp','exang','slope'], axis= 1, inplace= True)
new_data.head()


# Now we separate target and the rest

# In[ ]:


y=new_data['target']
X=new_data.drop(['target'], axis= 1)


# We then split the data to train our models. Lets put 80% train 20% test

# In[ ]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train, y_test= train_test_split(X,y, test_size=0.2, random_state= 2)


# **MAKE MODELS**

# *Logistic Regression*

# In[ ]:


from sklearn.linear_model import LogisticRegression
lr= LogisticRegression()
lr.fit(X_train,y_train)
lr.score(X_test, y_test)


# *K nearest neighbors*

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
knn= KNeighborsClassifier()
knn.fit(X_train,y_train)
knn.score(X_test, y_test)


# *Decision Tree Classifier*

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(random_state=1)
dt.fit(X_train, y_train)
dt.score(X_test, y_test)


# *Gradient boosting classifier*

# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier()
gbc.fit(X_train, y_train)
gbc.score(X_test, y_test)


# *Gaussian NB*
# 

# In[ ]:


from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(X_train, y_train)
nb.score(X_test, y_test)


# *random forest classifier*

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
for i in range(1, 10):
    rfc = RandomForestClassifier(n_estimators=i)
    rfc.fit(X_train, y_train)
    print('n_estimators : ', i, "score : ", rfc.score(X_test, y_test), end="\n")


# *Support vectors machine*

# In[ ]:


from sklearn.svm import SVC
svc = SVC(kernel='linear')
svc.fit(X_train, y_train)
svc.score(X_test, y_test)


# Of all the models, logistics regression shows the best result
