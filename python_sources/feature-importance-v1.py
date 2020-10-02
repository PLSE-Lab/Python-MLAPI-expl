#!/usr/bin/env python
# coding: utf-8

# # Feature importance
# 
# I will explore the importance of the features using Univariate. Plan of attck is as follows:
# 1. Reduce training data set down to last month only (i.e. May 2016)
# 2. Clean data
# 3. Split up the 24 products and compute feature importance for each product
# 4. Summarise findings
# 
# This is my first attempt to compute feature importance, any comments and suggestions are welcome.
# 
# **_If you find this notebook useful, I shall be grateful for any upvote :)_**

# ## 1) Reduce training data set to last month

# In[ ]:


f = open('../input/train_ver2.csv','r')
g = open('last_month.csv','w')

for line in f:
    date = line[:10]
    if date == '2016-05-28':
        g.write(line)


# In[ ]:


with open('../input/train_ver2.csv', 'r') as f:
    cols = f.readline().split(',')


# In[ ]:


cols = [s.replace('"', '') for s in cols]


# ## 2) Data Cleaning

# In[ ]:


import numpy as np
import pandas as pd


# In[ ]:


df = pd.read_csv('last_month.csv',dtype={'indrel_1mes': str, 'conyuemp':str},names=cols)


# In[ ]:


df_features = df.iloc[:,:24]


# In[ ]:


df_features.head()


# Remove irrelevant features

# In[ ]:


del df_features['fecha_dato']
del df_features['ncodpers']


# In[ ]:


df_features['tipodom'].value_counts()


# 'tipodom' has only one value -> remove

# In[ ]:


del df_features['tipodom']


# Proportion of null values within each feature:

# In[ ]:


(df_features.isnull().sum()/len(df_features)).sort_values()


# Dumping 'ult_fec_cli_1t' and 'conyuemp'

# In[ ]:


del df_features['ult_fec_cli_1t']
del df_features['conyuemp']


# Label encode all categorical features (this will replace null values in categorical features with a value)

# In[ ]:


from sklearn import preprocessing

for f in df_features.columns:
    if df_features[f].dtype == 'object':
        print(f)
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(df_features[f].values))
        df_features[f] = lbl.transform(list(df_features[f].values))


# The idea is to replace the null values of 'renta' with appropriate median values. To do so, get most correlated features for 'renta' ...

# In[ ]:


df_features.corr()['renta'].sort_values()


# 'segmento' and 'ind_actividad_cliente' seem reasonable and relevant fetaures to group by to get the median for 'renta':

# In[ ]:


df_features['segmento'].value_counts()


# In[ ]:


df_features['ind_actividad_cliente'].value_counts()


# Calculate median values for 'renta' grouped by 'segmento' and 'ind_actividad_cliente'

# In[ ]:


median_renta = np.zeros((2,4))
for i in range(2):
    for j in range (4):
        median_renta[i][j] = df_features[(df_features['ind_actividad_cliente'] == i) &                                          (df_features['segmento'] == j)]['renta'].dropna().median()


# In[ ]:


median_renta


# Replace null values with median values

# In[ ]:


for i in range(0, 2):
    for j in range(0, 4):
        df_features.loc[(df_features['renta'].isnull()) &                         (df_features['ind_actividad_cliente'] == i) &                         (df_features['segmento'] == j), 'renta'] = median_renta[i][j]


# In[ ]:


(df_features.isnull().sum()/len(df_features)).sort_values()


# Fill null values for 'cod_prov' with median (with such few observations I don't bother with segmentation)

# In[ ]:


df_features['cod_prov'] = df_features['cod_prov'].fillna(df_features['cod_prov'].median())


# ## 3) Compute feature importance

# In[ ]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
import operator


# In[ ]:


X = df_features.values
test = SelectKBest(score_func=f_regression)
prod_cols = list(df.columns[24:48])


# Loop over product columns, assign every column to the target vector y and compute the feature scores each run

# In[ ]:


d = {}
for p in prod_cols:
    y = np.array(df[p])
    fit = test.fit(X, y)
    l = zip(df_features.columns, np.around(fit.scores_))
    d[p] = sorted(l, key=lambda x: x[1], reverse=True)


# ## 4) Summarise findings
# 
# Here I create a dataframe with the rankings for each product and feature and sum them up to find the most important features overall

# In[ ]:


df_ranking = pd.DataFrame(index=df_features.columns,columns=prod_cols)


# In[ ]:


for p in prod_cols:
    i = 0
    for r in d[p]:
        df_ranking[p][r[0]] = i
        i += 1


# In[ ]:


df_ranking['total'] = df_ranking.sum(axis=1)


# In[ ]:


df_ranking.sort_values('total')


# In[ ]:


df_ranking.sort_values('total')['total'].plot(kind='bar')


# ## Conclusion
# 
# This last step of summing up the rankings might not make much sense, but I was curious to see the consistency of importance each variable exhibits. It seems to me that there is a high degree of consistency, e.g. 'ind_actividad'cliente' is almost always in the top 3 for all products. This might give you a sense of how to consider the features when you use a classification or clustering algorithm.
