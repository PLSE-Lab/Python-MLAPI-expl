#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
import matplotlib.pyplot as plt
import lightgbm as lgb
import numpy as np
import seaborn as sns
import os


# In[ ]:


import warnings
warnings.filterwarnings("ignore")


# # 1. Data

# In[ ]:


df = pd.read_csv("../input/chess/games.csv", delimiter=',')


# In[ ]:


df.head()


# In[ ]:


df.info()


# In[ ]:


df.columns


# In[ ]:


print(df.head())


# In[ ]:


df.isnull().sum()


# In[ ]:


df.describe()


# In[ ]:


from sklearn.preprocessing import LabelEncoder

categorical_column = ['rated', 'winner', 'victory_status']
                      

for i in categorical_column:
    le = LabelEncoder()
    df[i] = le.fit_transform(df[i])
print(df.head())


# # 2.Distribution Plots

# In[ ]:


df[['black_rating', 'white_rating', 'turns', 'opening_ply', 'created_at', 'last_move_at']].hist(figsize=(12, 6), bins=50, grid=False)


# In[ ]:


sns.distplot(df['white_rating'])


# In[ ]:


sns.distplot(df['white_rating'],kde=False,bins=30)


# In[ ]:


sns.jointplot(x='white_rating',y='black_rating',data=df,kind='scatter')


# In[ ]:


sns.jointplot(x='white_rating',y='black_rating',data=df,kind='hex')


# In[ ]:


sns.jointplot(x='white_rating',y='black_rating',data=df,kind='reg')


# In[ ]:


sns.pairplot(df)


# In[ ]:


sns.pairplot(df,hue='winner',palette='coolwarm')


# # 3.Categorical Plots

# In[ ]:


sns.barplot(x='winner',y='turns',data=df)


# In[ ]:


sns.barplot(x='winner',y='turns',data=df,estimator=np.std)


# In[ ]:


sns.countplot(x='winner',data=df)


# In[ ]:


sns.boxplot(x="winner", y="turns", data=df,palette='rainbow')


# In[ ]:


sns.violinplot(x="winner", y="turns", data=df,palette='rainbow')


# # 4.Matrix plot

# In[ ]:


df.corr()


# In[ ]:


sns.heatmap(df.corr())


# In[ ]:


sns.heatmap(df.corr(),cmap='coolwarm',annot=True)


# # 5. Grids

# In[ ]:


g = sns.PairGrid(df)
g.map(plt.scatter)


# In[ ]:


sns.pairplot(df)


# # 6. Pandas with Data Visualization
# 

# In[ ]:


df['turns'].hist();


# In[ ]:


df['black_rating'].hist()


# In[ ]:


plt.style.use('ggplot')
df['white_rating'].hist()


# In[ ]:


plt.style.use('bmh')
df['white_rating'].hist()


# In[ ]:


plt.style.use('dark_background')
df['turns'].hist()


# In[ ]:


plt.style.use('ggplot')
df.plot.area(alpha=0.4)


# In[ ]:


plt.style.use('fivethirtyeight')
df.plot.area(alpha=0.4)


# In[ ]:




