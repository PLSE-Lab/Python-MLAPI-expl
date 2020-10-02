#!/usr/bin/env python
# coding: utf-8

# # Machine Learning Tutorial for Beginners
# 
# Hello !
# 
# I am going to try to tell about machine learning algorithm and mathematics behind machine learning models. Firstly I'll make data analysis, data visualization and data cleaning.
# 
# Let's start! 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# We'll examine the data.

# In[ ]:


adult_data = pd.read_csv('../input/adult_train.csv')


# In[ ]:


adult_data.head()


# We'll try to predict to salary that it's last column that called Target. Firstly we should think about the data. Which parameters are useful for your model that you'll create. We'll make visualization and cleaning before create machine learning model.

# In[ ]:


adult_data.info()


# # Exploratory Data Analysis

# In[ ]:


sns.pairplot(adult_data, hue = 'Target', markers= ['o','s'])


# In[ ]:


plt.figure(figsize=(9,6))
sns.countplot(x='Target', hue='Sex', data=adult_data)


# In[ ]:


adult_data['Workclass'].value_counts()


# In[ ]:


plt.figure(figsize=(9,6))
sns.countplot(x='Target', hue='Education', data=adult_data ,palette='rainbow')


# In[ ]:


adult_data.groupby(['Country','Target'])[['Target']].count().head(20)


# In[ ]:


adult_data['Target']=[0 if i==' <=50K' else 1 for i in adult_data['Target']]


# In[ ]:


#assing X as a Dataframe of features and y as a Series of outcome variable

X = adult_data.drop('Target', axis = 1)
y = adult_data.Target


# In[ ]:


X.info()


# In[ ]:


# I will decide which categorical data variables I want to use in model 
for col_name in X.columns:
    if X[col_name].dtypes == 'object':
        unique_categorical = len(X[col_name].unique())
        print("Feature '{col_name}' has {unique_categorical} unique categories".format(col_name=col_name, unique_categorical = unique_categorical))


# In[ ]:


X['Country'] = ['United-States' if i == ' United-States' else 'Other' for i in X['Country']]
X['Country'].value_counts().sort_values(ascending = False)


# In[ ]:


X.columns


# In[ ]:


X.isnull().sum().sort_values(ascending=False)


# In[ ]:


#create a list of features to dummy
todummy_list = ['Workclass', 'Education','Martial_Status', 'Occupation', 'Relationship', 'Race', 'Sex', 'Country']


# In[ ]:


# Function to dummy all the categorical variables used for modeling

def dummy_df(df,todummy_list):
    for x in todummy_list:
        dummies = pd.get_dummies(df[x],prefix=x, dummy_na=False)
        df = df.drop(x,1)
        df = pd.concat([df,dummies],axis = 1)
    return df


# In[ ]:


X = dummy_df(X,todummy_list)


# In[ ]:


X.isnull().sum().sort_values(ascending = False)


# In[ ]:


# Handling missing data
from sklearn.preprocessing import Imputer
imp = Imputer(missing_values = 'NaN', strategy = 'median',axis = 0)
imp.fit(X)
X = pd.DataFrame(data = imp.transform(X),columns = X.columns)

X.isnull().sum().sort_values(ascending = False)


# ## Outlier Detection

# There are so much way to find outlier values. Today we are going to one.
# * **Tukey IQR**
# 
# 
# ## Tukey IQR

# In[ ]:


def find_outliers_tukey(x):
    q1 = np.percentile(x,25)
    q3 = np.percentile(x,75)
    iqr = q3 - q1
    floor = q1 - 1.5*iqr
    ceiling = q3 + 1.5*iqr
    outlier_indices = list(x.index[(x<floor) | (x>ceiling)])
    outlier_values = list(x[outlier_indices])
    
    return outlier_indices,outlier_values


# In[ ]:


tukey_indices,tukey_values = find_outliers_tukey(X['Age'])
np.sort(tukey_values)


# # Feature Engineering

# In[ ]:


from itertools import combinations
from sklearn.preprocessing import PolynomialFeatures

def add_interactions(df):
    combos = list(combinations(list(df.columns),2))
    colnames = list(df.columns) + ['_'.join(x) for  x in combos]
    
    poly = PolynomialFeatures(interaction_only=True, include_bias=False)
    df = poly.fit_transform(df)
    df = pd.DataFrame(df)
    df.columns = colnames
    
    noint_indicies = [i for i, x in enumerate(list((df == 0).all())) if x]
    df = df.drop(df.columns[noint_indicies], axis= 1)
    
    return df


# In[ ]:


X = add_interactions(X)


# In[ ]:


X.head() # as you can se there are may many features now.


# ## Dimesionality reduction using PCA
# 
# PCA is a tecnique that transforms a dataset of many features into pricipal components that summarize the variance that underlies the data

# In[ ]:


from sklearn.decomposition import PCA

pca = PCA(n_components=10)
X_pca = pd.DataFrame(pca.fit_transform(X))


# In[ ]:


X_pca.head()


# We are ready to create machine learning models.

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


Xtrain,Xtest,ytrain,ytest = train_test_split(X,y, test_size=0.1,random_state=101)


# In[ ]:


X.shape


# In[ ]:


import sklearn.feature_selection

select = sklearn.feature_selection.SelectKBest(k=20)
selected_features = select.fit(Xtrain,ytrain)
indices_selected = selected_features.get_support(indices=True)
colnames_selected = [X.columns[i] for i in indices_selected] 


Xtrain_selected = Xtrain[colnames_selected]
Xtest_selected = Xtest[colnames_selected]


# In[ ]:


colnames_selected


# In[ ]:


Xtrain_selected


# # Random Forest Classifier

# In[ ]:


from sklearn.ensemble import RandomForestClassifier 


# In[ ]:


rf= RandomForestClassifier(n_estimators=100)
rf.fit(Xtrain_selected,ytrain)


# In[ ]:


rf_prediction = rf.predict(Xtest_selected)


# In[ ]:


from sklearn.metrics import roc_auc_score
print(roc_auc_score(ytest, rf_prediction))


# In[ ]:


from sklearn.metrics import confusion_matrix, classification_report


# In[ ]:


print(confusion_matrix(ytest, rf_prediction))


# In[ ]:


print(classification_report(ytest, rf_prediction))


# This model is not good !! We have to improve this algorithm.

# In[ ]:




