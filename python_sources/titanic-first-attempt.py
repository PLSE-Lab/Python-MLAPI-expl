#!/usr/bin/env python
# coding: utf-8

# Titanic
# 
# This kernel is mainly to keep track of the step i followed for this problem.

# In[ ]:


# import Libraries 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # plotting
import seaborn as sns # combined plotting 


# **Importing Data**
# 
# Going to import both datasets here so similar transformations can be done later on.
# PassengerId can be used as the index column.

# In[ ]:


df = pd.read_csv('../input/train.csv', index_col=0)
df_test = pd.read_csv('../input/test.csv', index_col=0)


# Take a quick look to the data

# In[ ]:


df.head(5)


# Checking the structure

# In[ ]:


df.dtypes


# **Visualize  Data**
# 
# Check the Survived distribution

# In[ ]:


df.Survived.hist()


# Seems there is enough data for both values. 
# 
# Lets go on and see how the other features are related. 
# Lets plot some histograms. 

# In[ ]:


## Function to plot conditioned histograms
def cond_hists(df, plot_cols, grid_col):
    
    ## Loop over the list of columns
    for col in plot_cols:
        grid1 = sns.FacetGrid(df, col=grid_col)
        grid1.map(plt.hist, col, alpha=.7, bins=6)
    return grid_col

## we need to conver the Sex and Embarked colums to numbers for the function to work 
df.Sex = df.Sex.map({'male':0, 'female':1})
df_test.Sex = df_test.Sex.map({'male':0, 'female':1})

## Define columns for making a conditioned histogram
plot_cols = ["Pclass",
               "Age",
               "Sex",
               "Parch",
               "SibSp"]

cond_hists(df, plot_cols, 'Survived')


# It seems that Pclass and Sex have a bigger impact on the 'Survived' Label. Age also makes some difference.
# Correlation matrix below also seems to agree. However that is not sure it means too much

# In[ ]:


df.corr()


# Lets check some other colums.

# In[ ]:


df.hist(column='Survived', by='Embarked', bins =3)


# Port C is making a difference compared to the 'Survived' histogram. However its a small part of the sample. 

# In[ ]:


df.hist(column='Pclass', by='Embarked', bins=5)


# Seems that people coming from port C have a big percentage of Pclass 1.  Pclass 1 is favored to surviving. 

# In[ ]:


df.boxplot(column='Fare', by='Survived')


# These two overlap but seems that people who did survived payed higher prices in average. Not sure what to assume yet, since the Fare can be correlated to the Pclass and the Embarked features. 

# Check Some more correlations

# In[ ]:


df.boxplot(column='Fare', by='Pclass')


# In[ ]:


df.boxplot(column='Fare', by='Parch')


# **Checking Nulls**

# In[ ]:


df.isnull().sum()


# There are just 2 nulls in the Embarked column.  We could drop those or just set as 'S'

# In[ ]:


df.Embarked = df.Embarked.fillna('S')


# There is a lot of information missing related cabins.  Probably just means most of the people do not have cabins. We can try change the column to 0 = no cabin 1 = cabin 

# In[ ]:


# fill NAN first with something, and replace all
df.Cabin = df.Cabin.fillna('U')
df.Cabin = df.Cabin.map(lambda x: 0 if x == 'U' else 1)

# repeat for the test dataset 
df_test.Cabin = df.Cabin.fillna('U')
df_test.Cabin = df.Cabin.map(lambda x: 0 if x == 'U' else 1)


# Question remains what to do with age. 177 missing values is not small.  We need to check some correlations.  Might be needing the title from the name column.
# 

# In[ ]:


df['Title'] = df.Name.map(lambda x: x.split(',')[1].split('.')[0])
df_test['Title'] = df_test.Name.map(lambda x: x.split(',')[1].split('.')[0])
df.Title.unique()


# In[ ]:


df.Title = df.Title.map({' Mr':0, ' Mrs':1, ' Miss':2, ' Master':3, ' Don':4, ' Rev':5, ' Dr':6, ' Mme':7, ' Ms':8, ' Major':9, ' Lady':10, ' Sir':11, ' Mlle':12, ' Col':13, ' Capt':14, ' the Countess':15, ' Jonkheer':16})


# In[ ]:


df.boxplot(column='Age', by='Title')


# In[ ]:


df.corr()


# Foe now just gonna take the mean based on the Pclass column wich is more correlated

# In[ ]:


age1 = df[(df.Pclass == 1) & (df.Age.isnull() == False)]['Age'].mean()
age2 = df[(df.Pclass == 2) & (df.Age.isnull() == False)]['Age'].mean()
age3 = df[(df.Pclass == 3) & (df.Age.isnull() == False)]['Age'].mean()

df.loc[(df.Pclass == 1) & (df.Age.isnull() == True),'Age'] = age1
df.loc[(df.Pclass == 2) & (df.Age.isnull() == True),'Age'] = age2
df.loc[(df.Pclass == 3) & (df.Age.isnull() == True),'Age'] = age3

df_test.loc[(df_test.Pclass == 1) & (df_test.Age.isnull() == True),'Age'] = age1
df_test.loc[(df_test.Pclass == 2) & (df_test.Age.isnull() == True),'Age'] = age2
df_test.loc[(df_test.Pclass == 3) & (df_test.Age.isnull() == True),'Age'] = age3


# **Feature Selection and Engineering**

# In[ ]:


df.hist(column='Survived', by='SibSp', bins = 3)


# In[ ]:


def cond_SibSp (x):
    if x == 0:
        return 0 
    if x in [1,2]:
        return 1
    if x > 2:
        return 2
    
df.SibSp = df.SibSp.map(cond_SibSp)
df_test.SibSp = df_test.SibSp.map(cond_SibSp)


# In[ ]:


df.hist(column='SibSp', by='Survived', bins = 3)


# In[ ]:


df.hist(column='Survived', by='Parch', bins = 3)


# In[ ]:


df.Parch = df.Parch.map(lambda x: 0 if x == 0 else 1)
df_test.Parch = df_test.Parch.map(lambda x: 0 if x == 0 else 1)


# In[ ]:


df.hist(column='Survived', by='Parch', bins = 3)


# In[ ]:


df.corr()

