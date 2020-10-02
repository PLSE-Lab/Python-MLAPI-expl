#!/usr/bin/env python
# coding: utf-8

# In[151]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_predict
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# # Visaulization of Data

# In[115]:


df=pd.read_csv('../input/FIFA 2018 Statistics.csv')
df.head()


# In[116]:


df.info()


# In[117]:


#dummy=pd.get_dummies(df.loc[:,'Man of the Match'])
#df=df.merge(dummy,left_index=True,right_index=True)
df.head()


# In[118]:


df.corr()


# In[119]:


df.describe()


# In[120]:


df.isnull().sum()


# ### since, the columns "own goals, own goal time, and 1st goal have 90% missing values. so rather we will drop these columns"

# In[121]:


for col in df.columns:
    col.rstrip()


# ## One hot encoding is done here after droping and cleaning the data as necessary

# In[122]:


tree = DecisionTreeClassifier()
df.drop(['Own goal Time', 'Own goals', '1st Goal'], axis = 1, inplace= True)
df.drop(['Corners', 'Fouls Committed', 'On-Target'], axis = 1, inplace=True)
df.drop('Date', axis = 1, inplace=True)
one_hot_data = pd.get_dummies(df,drop_first=True)
#one_hot_data.head()
one_hot_data.info()


# In[123]:


#one_hot_data.dropna(inplace=True)
#tree.fit(one_hot_data, one_hot_data.loc[:,'PSO_Yes'])


# In[124]:


one_hot_data.info()


# # Function to calculate Accuracy as well as confusion matrix for the results

# In[125]:


def cal_accuracy(y_test, y_pred): 
      
    print("Confusion Matrix: ", 
        confusion_matrix(y_test, y_pred)) 
      
    print ("Accuracy : ", 
    accuracy_score(y_test,y_pred)*100) 
      
    print("Report : ", 
    classification_report(y_test, y_pred))


# In[126]:


#idx = pd.IndexSlice
#one_hot_data.fillna(one_hot_data['Own goal Time'].mean(),inplace=True)
#one_hot_data.fillna(value=0,inplace=True)


# In[127]:


one_hot_data.isnull().values.any()


# In[128]:


one_hot_data.head()


# In[130]:


df = one_hot_data.copy()
df.describe()


# In[131]:


df = df.apply(LabelEncoder().fit_transform)
df.head()


# # Selecting Label and Features

# In[132]:


label = df['Man of the Match_Yes']

features = df.drop(['Man of the Match_Yes'], axis = 1)


# # Splitting the Data into training and testing by 20%

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(features, label, test_size = 0.20, random_state =0) 


# # Decision Tree Classifier

# In[147]:


clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100,max_depth=3, min_samples_leaf=5) 
clf_gini.fit(X_train, y_train)


# In[148]:


y_pred = clf_gini.predict(X_test) 
print("Predicted values:") 
print(y_pred) 
    
cal_accuracy(y_test, y_pred)


# In[135]:


y_pred = clf_gini.predict(X_train) 
print("Predicted values:") 
print(y_pred) 
    
cal_accuracy(y_train, y_pred)


# # Cross Validation of Decesion Tree

# In[154]:


scores = cross_val_score(clf_gini, features, label, cv=5)
scores  


# In[136]:


one_hot_data.head()


# # Gaussion Naive Bayes

# In[138]:


model = GaussianNB()
model.fit(X_train, y_train)


# In[139]:


y_pred = model.predict(X_test)
cal_accuracy(y_test, y_pred)


# In[140]:


y_pred = model.predict(X_train)
cal_accuracy(y_train, y_pred)


# # cross validation of Naive Bayes

# In[153]:


scores = cross_val_score(model, features, label, cv=5)
scores  


# # Random Forest

# In[142]:


clf = RandomForestClassifier(n_estimators=100, max_depth=1,random_state=0)
clf.fit(X_train, y_train)


# In[143]:


y_pred=clf.predict(X_test)
cal_accuracy(y_test, y_pred)


# In[144]:


y_pred=clf.predict(X_train)
cal_accuracy(y_train, y_pred)


# # Cross Validation of Random Forest

# In[152]:


scores = cross_val_score(clf, features, label, cv=5)
scores  


# # Conclusion

# In[155]:


models = pd.DataFrame({
        'Model'          : ['Naive Bayes',  'Decision Tree', 'Random Forest'],
        'Training_Score' : [model.score(X_train,y_train),  clf_gini.score(X_train,y_train), clf.score(X_train,y_train)],
        'Testing_Score'  : [model.score(X_test,y_test), clf_gini.score(X_test,y_test), clf.score(X_test,y_test)]
    })
models.sort_values(by='Testing_Score', ascending=False)


# In[ ]:




