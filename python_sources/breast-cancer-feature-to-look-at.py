#!/usr/bin/env python
# coding: utf-8

# **feature importance and feature reduction using ExtraTreeClassifer**

# In[2]:


import numpy as np
import pandas as pd 
from sklearn.decomposition import PCA


# In[3]:


df = pd.read_csv('../input/data.csv')


# In[4]:


df.loc[df['diagnosis'] != 'M', 'diagnosis'] = 1
df.loc[df['diagnosis'] == 'M', 'diagnosis'] = 0


# In[5]:


df = df.drop('id', axis = 1)
df = df.fillna(0)


# In[6]:


df.head()


# In[7]:


df.columns


# In[8]:


df = df.drop('Unnamed: 32', axis = 1)


# In[9]:


df.head()


# In[10]:


df_x = df.copy()
X = df_x.drop('diagnosis', axis = 1)
y = df['diagnosis']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y , test_size=0.1, random_state=42)


# In[11]:


import matplotlib.pyplot as plt
# lets try to relate different features
# how to find the importance ? 
# ExtraTreeClassifier can help in finding the feature importance

from sklearn.ensemble import ExtraTreesClassifier
# Build a forest and compute the feature importances
forest = ExtraTreesClassifier(n_estimators=250,
                              random_state=0)
forest.fit(X_train, y_train)
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]
print("Feature ranking:")
for f in range(X.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))


# In[12]:


plt.figure(figsize=(12,10))
plt.title("Feature importances")
plt.bar(range(X_train.shape[1]), importances[indices],
       color="g", yerr=std[indices], align="center")
plt.xticks(range(X_train.shape[1]), indices)
plt.xlim([-1, X_train.shape[1]])
plt.show()


# In[13]:


# as we know the important feature , we now add a feature reduction technique !
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_r = pca.fit(X).transform(X)
print('explained variance ratio (first two components): %s'
      % str(pca.explained_variance_ratio_))


# In[15]:


import seaborn as sns
sns.set(style="whitegrid")


# In[19]:


sns.residplot(X_r[:50,0], X_r[:50,1], lowess=True, color="g")


# In[33]:


sns.kdeplot(X_r[:20,0],X_r[:20,1],cmap="Reds", shade=True, shade_lowest=False)


# In[40]:


f, ax = plt.subplots(figsize=(18, 7))
v1 = ax.violinplot(X_r[:,1], points=50, positions=np.arange(0, len(X_r[:,1])), widths=0.85,
               showmeans=False, showextrema=False, showmedians=False)
for b in v1['bodies']:
    m = np.mean(b.get_paths()[0].vertices[:, 0])
    b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], -np.inf, m)
    b.set_color('r')
v2 = ax.violinplot(X_r[:,0], points=50, positions=np.arange(0, len(X_r[:,1])), widths=0.85,
               showmeans=False, showextrema=False, showmedians=False)
for b in v2['bodies']:
    m = np.mean(b.get_paths()[0].vertices[:, 0])
    b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], m, np.inf)
    b.set_color('b')


# In[ ]:




