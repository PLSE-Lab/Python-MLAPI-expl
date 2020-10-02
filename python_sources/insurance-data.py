#!/usr/bin/env python
# coding: utf-8

# ### Insurance Data Exploration
# ***
# .

# In[ ]:


# IMPORT 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


#! head -n 3 ../input/exercise_02_train.csv


# In[ ]:


insu = pd.read_csv('../input/exercise_02_train.csv')


# In[ ]:


insu.head(2).T


# In[ ]:


insu.y.value_counts().plot(kind='bar')


# In[ ]:


insu.info(verbose=True, max_cols=101)


# In[ ]:


insu.head(1).T


# ### Handling Columns of type object
# ***
# - Cleaning
# - Standardizing Names
# - Value count bar-plots

# In[ ]:


# CATEGORICAL COLUMNS
insu.loc[:,insu.dtypes==object].head()


# #### Object-type Columns are
# - x34: Brand
# - x35: Day
# - x41: Money amount
# - x45: Some %-age
# - x68: Month
# - x93: Country
# ***
# We need to clean some columns.

# In[ ]:


# Remove the $ symbol
insu['x41'] = insu['x41'].str.replace('$','').astype(float)


# In[ ]:


#Remove the % symbol
insu['x45'] = insu['x45'].str.replace('%','').astype(float)


# In[ ]:


insu.loc[:,insu.dtypes==object].head()


# In[ ]:


insu['x34'].value_counts().plot(kind='barh')


# In[ ]:


# Make all brand names lowercase
insu['x34'] = insu['x34'].str.lower()


# In[ ]:


insu['x35'].value_counts().plot(kind='barh')


# In[ ]:


s1 = insu['x35']


# In[ ]:


# Standardize the day names
insu['x35'] = s1.replace({'monday':'mon', 'tuesday':'tue', 'wednesday':'wed',
        'thurday':'thu', 'thur':'thu','friday':'fri'})


# In[ ]:


insu['x35'].value_counts().plot(kind='barh')


# In[ ]:


insu.loc[:,insu.dtypes==object].head()


# In[ ]:


insu['x68'].value_counts().sort_values().plot(kind='barh')


# In[ ]:


insu['x68'] = insu['x68'].str.lower()


# In[ ]:


#Standardize the month names
insu['x68'] = insu['x68'].replace({'january':'jan', 'dev':'dec', 'sept.':'sep',
        'july':'jul'})


# In[ ]:


insu['x93'].value_counts().sort_values().plot(kind='barh')


# In[ ]:


insu.loc[:,insu.dtypes==object].head()


# ### Missing data

# In[ ]:


# Look at missing rows
insu[insu.isnull().any(axis=1)].shape


# In[ ]:


#Drop rows with missing data
insu.dropna(how='any', inplace=True)


# In[ ]:


# Look at missing rows AGAIN
insu[insu.isnull().any(axis=1)].shape


# In[ ]:


insu.x0.plot(kind='hist')


# In[ ]:



cols = insu.columns
insu.boxplot(column=['x0', 'x1', 'x2'])


# ### Handle Outliers
# *** 
# We will return on this in next class

# ### Encode categorical columns using get_dummies

# In[ ]:


target = insu.y
insu.drop('y', axis=1, inplace=True)


# In[ ]:


insu2 = pd.get_dummies(insu, columns=['x34', 'x35', 'x68', 'x93'])


# In[ ]:


insu2.head()


# #### Prepare data for modeling

# In[ ]:


X = insu2.values
y = target.values


# In[ ]:


corr = np.corrcoef(X.T,y)


# In[ ]:


sns.heatmap(data = corr,vmin=0, vmax=1)


# In[ ]:


# Split the data for train and dev/test purpose 
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, y,test_size=0.20, 
                                                    random_state=10)


# In[ ]:


#Normalizer or Standardized
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X_train)
Xn_train = scaler.transform(X_train)
Xn_test = scaler.transform(X_test)


# In[ ]:


from sklearn.linear_model import LogisticRegressionCV


# #### Logistic Regression

# In[ ]:


lrm = LogisticRegressionCV(tol=0.0001)
lrm.fit(Xn_train, Y_train)


# In[ ]:


print("Test Accuracy: ", 100*lrm.score(Xn_test, Y_test))


# In[ ]:


from sklearn.metrics import classification_report
print ("CLASSIFICATION REPORT:\n")
print (classification_report(Y_test, lrm.predict(Xn_test)))


# In[ ]:





# #### Adaboost an ensemble of Logistic Regression Classifiers

# In[ ]:


from sklearn.ensemble import AdaBoostClassifier
abc = AdaBoostClassifier(base_estimator=LogisticRegressionCV())
abc.fit(Xn_train, Y_train)
print("Test Accuracy: \n", 100*abc.score(Xn_test, Y_test))
print( "CLASSIFICATION REPORT:\n")
print (classification_report(Y_test,abc.predict(Xn_test)))


# **Work in progress....**

# In[ ]:





# In[ ]:




