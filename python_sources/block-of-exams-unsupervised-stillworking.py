#!/usr/bin/env python
# coding: utf-8

# ## Description
# 
# Adaptation of [previous notebook](https://www.kaggle.com/ossamum/exploratory-data-analysis-and-feature-importance)
# 
# This notebook has three sections: Importings; Processing; Exploiting and Classifying.

# ### Importings

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import missingno as msno
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


# In[ ]:


df = pd.read_excel('../input/covid19/dataset.xlsx')


# ### Processing

# In[ ]:


df['SARS-Cov-2 exam result'].value_counts().plot.barh()

df.rename(columns = {'SARS-Cov-2 exam result': 'target'}, inplace = True)

df['target'] = df.target.map({'negative': 0, 'positive': 1})

df.target.isnull().any()


# In[ ]:


# checking null values
msno.bar(df, figsize=(16, 4))


# It seems that there are some columns with all values equal null.

# Lets see the columns with the smallest numbers of null values.

# ### Exploiting

# In[ ]:


df_columns = df.isnull().mean()


# In[ ]:


df_columns.sort_values()[:30]


# - We are not gonna use the addmission columns. It should be a function of whether a preliminar diagnosis points to coronavirus or not.
# 
# - There is a pattern indicating that there are blocks of exams. Lets check it.

# In[ ]:


interest_columns = df_columns[df_columns == df_columns.sort_values()[8]].index.values
df1 = df.loc[:, interest_columns]


# In[ ]:


np.random.seed(24)


# In[ ]:


df1.sample(4)


# In[ ]:


(len(df1) - len(df1.dropna()))/len(df1)


# In[ ]:


interest_columns = df_columns[df_columns == df_columns.sort_values()[10]].index.values
df2 = df.loc[:, interest_columns]


# In[ ]:


(len(df2) - len(df2.dropna()))/len(df2)


# In[ ]:


interest_columns = df_columns[df_columns == df_columns.sort_values()[29]].index.values
df3 = df.loc[:, interest_columns]


# In[ ]:


(len(df3) - len(df3.dropna()))/len(df3)


# In[ ]:


df2.sample(6)


# In[ ]:


df3.sample(6)


# #### Baskets of Exames Exploiting

# In[ ]:


# Block now represents the set of exams that the patient did
df['block'] = df.notnull().apply(lambda x: x.astype(int).astype(str).sum()  ,axis = 1)


# In[ ]:


df.iloc[:4, :]


# In[ ]:


df.groupby('block').target.count().sort_values(ascending = False).reset_index()[:20]


# We are gonna study the effectiveness of each block of exams using hierarchical clustering as learning method and information gain as evaluation of effectivess of each block of exams.
# 
# The number of clusters can be choosen using some distance threshold.

# ## Using a test classifier

# In[ ]:


not_null = df.notnull().values


# First I am gonna work on the third block of exams, because of its numerical format and the relevance showed on previous notebooks of the leukocytes.

# In[ ]:


not_made = df3.Platelets.isnull()


# In[ ]:


df.target[not_made].mean()


# In[ ]:


df.target[~not_made].mean()


# Here we can see that people who made the third block of exams are realy more leaning to have the desease.

# ### Classifying

# In[ ]:


df3['age'] = df['Patient age quantile']
df3['target'] = df['target']


# In[ ]:


df3.dropna(inplace = True)
df3.reset_index(inplace = True, drop = True)
df3.head()


# In[ ]:


from sklearn.preprocessing import StandardScaler


# In[ ]:


scaler = StandardScaler()
df3.iloc[:, :11] = scaler.fit_transform(df3.iloc[:, :11])


# In[ ]:


df3.iloc[:, :11].head()


# In[ ]:


import scipy.cluster.hierarchy as shc

plt.figure(figsize = (10, 7))

dend = shc.dendrogram(shc.linkage(df3.iloc[:, :11], method = 'ward'))


# We can see 5 major classes, let split it.

# In[ ]:


from sklearn.cluster import AgglomerativeClustering

cluster = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')


# In[ ]:


df3['cluster'] = cluster.fit_predict(df3.iloc[:, :11])


# In[ ]:


df3.groupby('cluster').agg({'target': {'media': np.mean, 'n': lambda x: x.count()}})


# In[ ]:


df3.iloc[:, :11] = scaler.inverse_transform(df3.iloc[:, :11])


# In[ ]:


df3.groupby('cluster').mean().sort_values('target')


# In[ ]:


Still necessary study the other blocks of exams.

