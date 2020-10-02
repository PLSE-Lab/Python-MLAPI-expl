#!/usr/bin/env python
# coding: utf-8

#  **Goal:** The goal here is to make a classifier that takes in-game attributes of an item as an input and outputs it's item quality (restricted to Uncommon, Rare and Epic). In other words it's just a simple classification problem.
# 
# The metric used during testing will simply be the accuracy score.
# 
# 
# **Dependencies: numpy, pandas, matplotlib, seaborn, scikit-learn, xgboost.**

# In[ ]:


import os

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from random import seed
seed(42)


# Extracting the data for different body parts and joining them

# In[ ]:


chest_df = pd.read_csv('../input/world-of-warcraft-items-dataset/chest.csv')
hands_df = pd.read_csv('../input/world-of-warcraft-items-dataset/hands.csv')
feet_df = pd.read_csv('../input/world-of-warcraft-items-dataset/feet.csv')
head_df = pd.read_csv('../input/world-of-warcraft-items-dataset/head.csv')
legs_df = pd.read_csv('../input/world-of-warcraft-items-dataset/legs.csv')

# One hot encoding of body part type, as this is a very important feature
chest_df['chest_type']=1
hands_df['hands_type']=1
feet_df['feet_type']=1
head_df['head_type']=1
legs_df['legs_type']=1


# ### Way to automate this:
# i = 0
# df = pd.DataFrame()
# for file in os.listdir('./world-of-warcraft-items-dataset'):
#     app = pd.read_csv(f'./world-of-warcraft-items-dataset/{file}')
#     app['item_type_'+str(i)]=1
#     df = pd.concat([df,app])
#     i+=1
    


# In[ ]:


df = pd.concat([chest_df,feet_df,hands_df,head_df,legs_df], sort=True)
df.head()


# # Data Exploration and Cleaning

# Checking for NaN values

# In[ ]:


df.isnull().sum()/df.shape[0]


# Dropping Features with too many NaN values

# In[ ]:


tooMuchNa = df.columns[df.isnull().sum()/df.shape[0] > 0.98]


# In[ ]:


df = df.drop(tooMuchNa, axis =1)


# Dropping name because its irrevelant and classes + socket information

# In[ ]:


df = df.drop(['name_enus','classes','socket1','socket2','socket3'], axis =1)


# Renaming quality to target as this is going to be the target of the Classification

# In[ ]:


df = df.rename({'quality':'target'},axis =1)


# Dropping rows with NaN target

# In[ ]:


df = df.dropna(subset=['target'])


# Filling NaN values to 0 (which is fair as many of those NaN represent a 0 in the stat)

# In[ ]:


df = df.fillna(0)
df


# Checking data distribution

# In[ ]:


df.max()


# In[ ]:


df.min()


# Itemset is an ID just like name so I decided to drop it

# In[ ]:


df = df.drop('itemset', axis=1) 


# Data distribution : linear vs log ?

# In[ ]:


sns.distplot(df['agi'])


# In[ ]:


sns.distplot(df['agi'].apply(lambda x: np.log(x+1)))


# I think stats with a high max will be better distributed and applying log as shown above

# In[ ]:


logNeeded = df.drop('target',axis=1).max()[df.drop('target',axis=1).max() > 500].index
for column in logNeeded:
    df[column]=df[column].apply(lambda x: np.log(x+1))


# In[ ]:


df.max()


# agiint and strint are almost duplicates of agi and str so I'm dropping them

# In[ ]:


df = df.drop(['agiint','strint'], axis=1)


# Now that the data is cleaned, let's see the distributions

# In[ ]:


df.hist(figsize=(20,20))


# Target distribution

# In[ ]:


sns.countplot(df['target'])


# The dataset is very unbalanced, I could have for example oversampled the very rare classes but I decided to drop them and just do classification on the 3 most important ones.

# In[ ]:


df = df[df['target'].isin(['Uncommon', 'Rare', 'Epic'])]
df


# Label encoding for Classification

# In[ ]:


#Label Encoding
LE_df = df.replace( {'target': {'Uncommon':0,'Rare':1, 'Epic':2}})


# Train test splitting

# In[ ]:


from sklearn.model_selection import train_test_split
X = LE_df.drop('target',axis =1)
y = LE_df['target']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)


# # Classification
# Tree based methods seem great for this kind of problem as they divide the input space in 'boxes'

# Classification using XGBoost

# In[ ]:


import xgboost as xgb
from sklearn.metrics import accuracy_score


XGBC = xgb.XGBClassifier(max_depth=10,n_estimators=50)
XGBC.fit(X_train,y_train)
y_pre_xgb= XGBC.predict(X_test)
print('Accuracy : ',accuracy_score(y_test,y_pre_xgb))


# Great accuracy!

# In[ ]:


fig, ax = plt.subplots(figsize=(10,10))
xgb.plot_importance(XGBC, height=0.5, ax=ax, importance_type='gain')
plt.show()


# Knowing what these features mean I'm not surprised to see this, and quite happy that this time the classifier "thinks like a human being"!

# Thank you for reading !
