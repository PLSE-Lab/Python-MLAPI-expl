#!/usr/bin/env python
# coding: utf-8

# ## Aim-> To predict the Survied and unsurvied passangers.

# ### Importing Packages
# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt #Plotting the Data

import seaborn as sns

import math
import missingno
import pylab 
import scipy.stats as stats

import re #Regular Expression

# Display for notebooks
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams["figure.figsize"] = (20,10)
plt.style.use('seaborn-whitegrid')


# > ### Initial Handshake with data

# In[ ]:


titanic_train=pd.read_csv("../input/train.csv")
titanic_test=pd.read_csv("../input/test.csv")


# In[ ]:


titanic_train.head()


# In[ ]:


titanic_train.info()
titanic_test.info()


# In[ ]:


for x in titanic_train.columns:
        print("No of unique values of  "+x+" "+str(titanic_train[x].nunique()))
        
print("----------Test data set-----------")
        
for y in titanic_test.columns:
       print("No of unique values of  "+y+" "+str(titanic_test[y].nunique()))


# ##### We can remove the PassengerId and Ticket column from data set as we have distinct values for each record. We are keeping Name as it has potential to get some insights.

# In[ ]:


#Ticket unique value %
print(681/891)

print(363/418)


# In[ ]:


titanic_train=titanic_train.drop(['PassengerId','Ticket'],axis=1)
titanic_test=titanic_test.drop(['PassengerId','Ticket'],axis=1)


# In[ ]:


print(titanic_train.columns)
print(titanic_train.columns)


# In[ ]:


#Describing Numeirc Data

titanic_train.describe()


# In[ ]:


#Describing Categorical Data

titanic_train.describe(include=['O'])


# ## Plot the Distrubution of Each future

# In[ ]:


def plot_distribution(dataset, cols=5, width=20, height=15, hspace=0.2, wspace=0.5):
    plt.style.use('seaborn-whitegrid')
    fig = plt.figure(figsize=(width,height))
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=wspace, hspace=hspace)
    rows = math.ceil(float(dataset.shape[1]) / cols)
    for i, column in enumerate(dataset.columns):
        ax = fig.add_subplot(rows, cols, i + 1)
        ax.set_title(column)
        if dataset.dtypes[column] == np.object:
            g = sns.countplot(y=column, data=dataset)
            substrings = [s.get_text()[:18] for s in g.get_yticklabels()]
            g.set(yticklabels=substrings)
            plt.xticks(rotation=25)
        else:
            g = sns.distplot(dataset[column])
            plt.xticks(rotation=25)
    
plot_distribution(titanic_train.dropna(), cols=3, width=20, height=20, hspace=0.45, wspace=0.5)


# In[ ]:


#How many missing values are there in dataset
missingno.matrix(titanic_train,figsize=(10,5))
missingno.bar(titanic_train,sort='ascending')


# In[ ]:


#Checking % of missing values
vars_with_na=[]
def findVariablesWithMissingValues(df):
    # make a list of the variables that contain missing values
    global vars_with_na
    vars_with_na = [var for var in df.columns if df[var].isnull().sum()>1]

    # print the variable name and the percentage of missing values
    for var in vars_with_na:
        print(var, np.round(df[var].isnull().mean()*100, 3),  ' % missing values')

findVariablesWithMissingValues(titanic_train)


# In[ ]:


l=['Pclass', 'Sex', 'Age', 'SibSp','Parch', 'Embarked']
def relBetVarSur(dataset, cols=5, width=20, height=15, hspace=0.2, wspace=0.5):
    plt.style.use('seaborn-whitegrid')
    fig = plt.figure(figsize=(width,height))
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=wspace, hspace=hspace)
    rows = math.ceil(float(len(dataset)) / cols)
    for i, column in enumerate(dataset):
        ax = fig.add_subplot(rows, cols, i + 1)
        ax.set_title(column)
        titanic_train.groupby([column,'Survived'])["Sex"].count().plot.bar()
        plt.xticks(rotation=25)
    
relBetVarSur(l, cols=2, width=20, height=20, hspace=0.45, wspace=0.5)


# In[ ]:


def analyse_na_value(dataset, cols=5, width=20, height=15, hspace=0.2, wspace=0.5):
    plt.style.use('seaborn-whitegrid')
    fig = plt.figure(figsize=(width,height))
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=wspace, hspace=hspace)
    rows = math.ceil(float(len(dataset)) / cols)
    for i, column in enumerate(dataset):
        ax = fig.add_subplot(rows, cols, i + 1)
        ax.set_title(column)
        df = titanic_train.copy()
        df[column] = np.where(df[column].isnull(), 1, 0)
        df.groupby([column,'Survived'])["Sex"].count().plot.bar()
#         substrings = [s.get_text()[:18] for s in g.get_yticklabels()]
#         g.set(yticklabels=substrings)
        plt.xticks(rotation=25)
    
analyse_na_value(vars_with_na, cols=2, width=20, height=20, hspace=0.45, wspace=0.5)


# In[ ]:


# list of numerical variables
num_vars = [var for var in titanic_train.columns if titanic_train[var].dtypes != 'O']

print('Number of numerical variables: ', len(num_vars))

# visualise the numerical variables
titanic_train[num_vars].head()


# In[ ]:


#  list of discrete variables
discrete_vars = [var for var in num_vars if titanic_train[var].nunique()<20 ]

print('Number of discrete variables: ', len(discrete_vars))
titanic_train[discrete_vars].head().drop('Survived',axis=1)


# In[ ]:


def analyse_discrete(dataset, cols=5, width=20, height=15, hspace=0.2, wspace=0.5):
    plt.style.use('seaborn-whitegrid')
    fig = plt.figure(figsize=(width,height))
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=wspace, hspace=hspace)
    rows = math.ceil(float(len(dataset)) / cols)
    for i, column in enumerate(dataset):
        ax = fig.add_subplot(rows, cols, i + 1)
        ax.set_title(column)
        df = titanic_train.copy()
        df.groupby([column]).Survived.count().plot.bar()
#         substrings = [s.get_text()[:18] for s in g.get_yticklabels()]
#         g.set(yticklabels="Survived")
        ax.set_ylabel('No of Passengers')
        plt.xticks(rotation=25)
    
analyse_discrete(discrete_vars, cols=2, width=20, height=20, hspace=0.45, wspace=0.5)


# In[ ]:


# list of continuous variables
cont_vars = [var for var in num_vars if var not in discrete_vars]

print('Number of continuous variables: ', len(cont_vars))
titanic_train[cont_vars].head()


# In[ ]:


def analyse_continous(df, var):
    df = df.copy()
    plt.figure(figsize=(20,6))
    plt.subplot(1, 2, 1)
    df[var].hist(bins=20)
    plt.ylabel('Survived')
    plt.xlabel(var)
    plt.title(var)
    plt.subplot(1, 2, 2)
    stats.probplot(df[var], dist="norm", plot=pylab)
    plt.show()
    
    
for var in cont_vars:
    analyse_continous(titanic_train, var)


# In[ ]:


#Outliers

def find_outliers(dataset, cols=5, width=20, height=15, hspace=0.2, wspace=0.5):
    plt.style.use('seaborn-whitegrid')
    fig = plt.figure(figsize=(width,height))
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=wspace, hspace=hspace)
    rows = math.ceil(float(len(dataset)) / cols)
    for i, column in enumerate(dataset):
        ax = fig.add_subplot(rows, cols, i + 1)
        ax.set_title(column)
        df = titanic_train.copy()
        df.boxplot(column=column)
#         substrings = [s.get_text()[:18] for s in g.get_yticklabels()]
#         ax.set(yticklabels="Survived")
        ax.set_ylabel(column)
        plt.xticks(rotation=25)
    
find_outliers(cont_vars, cols=2, width=20, height=20, hspace=0.45, wspace=0.5)


# ## Titanic Train Data Insights****

# 1. ##### Exploring the Name column with Survival

# In[ ]:


titanic_train['Name'].head()


# In[ ]:


titanic_train['Title']=titanic_train['Name'].apply(lambda x:re.search("([A-Z][a-z]+)\.",x).group(1))


# In[ ]:


plt.Figure(figsize=(12,5))

sns.countplot(x='Title',data=titanic_train,palette='hls')
plt.xlabel('Titles',fontsize=12)
plt.ylabel('Count',fontsize=16)
plt.title('Title distiribution',fontsize=20)
plt.xticks(rotation=45)
plt.show()


# In[ ]:


titanic_train['Title'] = titanic_train['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
titanic_train['Title'] = titanic_train['Title'].replace('Mlle', 'Miss')
titanic_train['Title'] = titanic_train['Title'].replace('Ms', 'Miss')
titanic_train['Title'] = titanic_train['Title'].replace('Mme', 'Mrs')

titanic_train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()


# In[ ]:


#Checking
titanic_train['Title'].isnull().sum()


# In[ ]:


#We can convert the categorical titles to ordinal.
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

titanic_train['Title'] = titanic_train['Title'].map(title_mapping)

titanic_train.head()


# In[ ]:


titanic_train=titanic_train.drop('Name',axis=1)

titanic_train.head()


# In[ ]:




