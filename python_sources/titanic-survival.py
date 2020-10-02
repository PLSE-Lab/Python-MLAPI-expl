#!/usr/bin/env python
# coding: utf-8

# ## 1. Introductions
# * Load data
# *  Data Analysis
# 
# ## 2. Data Preparation
# * Check for null and missing values
# * Label encoding
# 
# ## 3. Define model 
# * check different models
# * Evaluate different model
# 
# ## 4. Prediction and Submission

# ## Introduction
# ### 1.1 Load data

# In[ ]:


import pandas as pd
import numpy as np

train_path = '../input/train.csv'
test_path = '../input/test.csv'

train = pd.read_csv(train_path, index_col = 'PassengerId' )
test= pd.read_csv(test_path, index_col = 'PassengerId')
train.head()


# ### 1.2 Data Analysis

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns

def bar_plot(cat):
    live = train[train['Survived'] == 1][cat].value_counts()
    dead = train[train['Survived'] == 0][cat].value_counts()
    df = pd.DataFrame([live, dead, live/dead],['live', 'dead', 'live/dead'])
    print(df)
    df.plot.bar(stacked = True)


# In[ ]:


# number of male are more dead than number of female
bar_plot('Sex')


# In[ ]:


# Class 1 people having more survival chance than class 3

bar_plot('Pclass')


# In[ ]:


# Siblings and Spouse
bar_plot('SibSp')


# In[ ]:


# Parent and Child
bar_plot('Parch')


# In[ ]:


# Embarked
bar_plot("Embarked")


# ## 2 check for missing values and label encoding

# In[ ]:


train.info()


# In[ ]:


# Age 177  , Cabin 687
train.isnull().sum()


# In[ ]:


# Age 87, Cabin 327
test.isnull().sum()


# In[ ]:


# fill the "Age" missing values in train and test dataset 
combined = [train, test]
for df in combined:
    df['Name']=df['Name'].str.extract(pat = '([\w]+)\.', expand=True)   


# In[ ]:


train['Name'].head()


# In[ ]:


map_title = {"Mr": 0, "Miss": 1, "Mrs": 2, 
                 "Master": 3, "Dr": 3, "Rev": 3, "Col": 3, "Major": 3, "Mlle": 3,"Countess": 3,
                 "Ms": 3, "Lady": 3, "Jonkheer": 3, "Don": 3, "Dona" : 3, "Mme": 3,"Capt": 3,"Sir": 3 }
for df in combined:
    df['Name']=df['Name'].map(map_title)


# In[ ]:


train.head()


# In[ ]:


# fill missing 'Age' values 
train['Age'].fillna(train.groupby('Name')['Age'].transform('median'),inplace=True)
test['Age'].fillna(test.groupby('Name')['Age'].transform('median'),inplace=True)


# In[ ]:


sex_map = {'male': 0, 'female': 1}
for df in combined:
    df['Sex'] = df['Sex'].map(sex_map)
    


# In[ ]:


for df in combined:
    df['Age'] = np.where(df['Age'] <= 14, 0, df['Age'])
    df['Age'] = np.where((df['Age'] >14) & (df['Age'] <= 24), 1 ,df['Age'])
    df['Age'] = np.where((df['Age'] > 24) & (df['Age'] <=64 ), 2, df['Age'])
    df['Age'] = np.where(df['Age'] >64, 3, df['Age'])


# In[ ]:


train.head()


# In[ ]:


train['Cabin'].value_counts()


# In[ ]:


train.sample(40).plot.scatter(x='Pclass', y= 'Fare')

