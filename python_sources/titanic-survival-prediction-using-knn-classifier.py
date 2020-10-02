#!/usr/bin/env python
# coding: utf-8

# # ***About Titanic***

# RMS Titanic was a British passenger liner operated by the White Star Line that sank in the North Atlantic Ocean in the early morning hours of 15 April 1912, after striking an iceberg during her maiden voyage from Southampton to New York City. Of the estimated 2,224 passengers and crew aboard, more than 1,500 died, making the sinking one of modern history's deadliest peacetime commercial marine disasters.

# # **Steps:**
# 1. Getting familier with the data
# 2. Cleaning the data
# 3. Finding optimal K using cross validation (simple cross validation and KFold cross validation)
# 4. Finding the accuracy of the knn classifier when we apply on test data 
# 

# # **1. Getting familier with the data**

# In[ ]:


# Importing related Python libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import csv
import warnings
warnings.filterwarnings("ignore")


# In[ ]:


# Importing the training dataset
df = pd.read_csv('../input/titanic/train.csv')


# In[ ]:


df.head()


# In[ ]:


df.shape


# In[ ]:


df.dtypes


# Dropping the unwanted columns. Name, Ticket, Fare, Cabin seems unwanted columns, so let's remove them.

# In[ ]:


df = df.drop('Name', axis=1,)
df = df.drop('Ticket', axis=1,)
df = df.drop('Fare', axis=1,)
df = df.drop('Cabin', axis=1,)


# Combining # siblings and # parents

# In[ ]:


df['Family'] = df['SibSp'] + df['Parch'] + 1


# In[ ]:


df = df.drop('SibSp', axis=1,)
df = df.drop('Parch', axis=1,)


# In[ ]:


df.describe()


# # 2. Cleaning the data

# In[ ]:


# By describing data, we found out there few NAN's in Age
# so replacing Age with median of the column
df["Age"] = df["Age"].fillna(df["Age"].median())


# In[ ]:


df.describe()


# In[ ]:


df['Embarked'].value_counts()


# In[ ]:


#finding NAN's in Embarked column
df['Embarked'].isna().sum()


# In[ ]:


#Replacing the NAN's with most frequently used one i.e mode(metric that gives us most frequently used value)
print(df["Embarked"].mode())
df["Embarked"] = df["Embarked"].fillna("S")


# In[ ]:


df['Embarked'].describe()


# In[ ]:


# Replacing the categorical value Embarked into numerical value
df["Embarked"].unique()


# In[ ]:


df.Embarked.replace(['S', 'C', 'Q'], [1, 2, 3], inplace=True)


# In[ ]:


df.Sex.replace(['male', 'female'], [1,0], inplace=True)


# In[ ]:


df.head()


# # 3. Finding optimal K using cross validation (simple cross validation and KFold cross validation)

# In[ ]:


# importing libraries
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from collections import Counter
from sklearn.model_selection import cross_validate


# In[ ]:


X = np.array(df.filter(['Pclass','Sex','Embarked','Family','Age'], axis=1))


# In[ ]:


y = np.array(df.filter(['Survived'], axis=1))


# * Splitting the data into train and test data.          
# * Splitting the train data into train and validation data.

# In[ ]:


# simple cross validation
X_1, X_test, y_1, y_test = train_test_split(X,y, test_size=0.3)
X_tr, X_cv, y_tr, y_cv = train_test_split(X_1, y_1, test_size=0.3)


# Getting the best k using simple cross validation.

# In[ ]:


final_scores = []
for i in range(1,30,2):
    knn = KNeighborsClassifier(n_neighbors = i)
    knn.fit(X_tr, y_tr)
    pred = knn.predict(X_cv)
    acc = accuracy_score(y_cv, pred, normalize=True) * float(100)
    final_scores.append(acc)
    print('\n CV accuracy for k=%d is %d'%(i,acc))


# In[ ]:


optimal_k = final_scores.index(max(final_scores))
print(optimal_k)


# So, we got to know that K=optimal_k is the best optimal k by using simple cross validation (varies bocz we are splitting the data in randomly)

# In[ ]:


# getting accuracy if K=5 on the test data
df_test = pd.read_csv('../input/titanic/test.csv')
df_test = df_test.drop('Name', axis=1,)
df_test = df_test.drop('Ticket', axis=1,)
df_test = df_test.drop('Fare', axis=1,)
df_test = df_test.drop('Cabin', axis=1,)
df_test['Family'] = df_test['SibSp'] + df_test['Parch'] + 1
df_test = df_test.drop('SibSp', axis=1,)
df_test = df_test.drop('Parch', axis=1,)
df_test["Age"] = df_test["Age"].fillna(df["Age"].median())


# In[ ]:


df_test1 = pd.read_csv('../input/titanic/test.csv')
df_test1


# In[ ]:


print(df_test["Embarked"].mode())
df_test["Embarked"] = df_test["Embarked"].fillna("S")


# In[ ]:


df_test.Embarked.replace(['S', 'C', 'Q'], [1, 2, 3], inplace=True)
df_test.Sex.replace(['male', 'female'], [1,0], inplace=True)


# In[ ]:


X_test = np.array(df_test.filter(['Pclass','Sex','Embarked','Family','Age'], axis=1))
knn = KNeighborsClassifier(n_neighbors = optimal_k)
knn.fit(X_tr, y_tr)
pred = knn.predict(X_test)
print(pred)


# In[ ]:


#creating file for submission
df_test['Survived'] = pd.Series(pred, index=df_test.index)


# In[ ]:


df_test


# In[ ]:


final_df = df_test.filter(['PassengerId','Survived'], axis=1)


# In[ ]:


final_df.shape


# In[ ]:


final_df.to_csv("pred_survival.csv", encoding='utf-8')


# In[ ]:




