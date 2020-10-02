#!/usr/bin/env python
# coding: utf-8

# # PCA
# 
# **Dimensionality reduction:** If we want to visualize 2 variables(2 dimensional) we can do it using any type of plot, 1 variable at x-axis and another in y-axis. But, what if want to visualize 200 variables (200 dimension)? we can use dimensionality reduction techniques like PCA,t-SNE, UMAP for the same. This techniques intelligently summarizes/group information related to multi dimension to the required low dimension. Unfortunately this techniques didn't help much in this competetion.

# # Please upvote, if you find this kernel interesting

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
sns.set_style('whitegrid')
import os
import time
import lightgbm as lgb
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.model_selection import GridSearchCV
import sklearn


# In[ ]:


print(os.listdir("../input"))


# In[ ]:


train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
print('Rows: ',train_df.shape[0],'Columns: ',train_df.shape[1])
train_df.info()


# In[ ]:


train_df.head()


# In[ ]:


train_df['target'].value_counts()


# In[ ]:


sns.countplot(train_df['target'])
sns.set_style('whitegrid')


# In[ ]:


X_test = test_df.drop('ID_code',axis=1)


# In[ ]:


X = train_df.drop(['ID_code','target'],axis=1)
y = train_df['target']


# n_components=2 (Final output will be converted to 2 Dimension)

# In[ ]:


from sklearn.decomposition import PCA
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(X)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])


# "explained_variance_ratio_" gives you percentage of variance covered by the first 2 principal component

# In[ ]:


pca.explained_variance_ratio_


# In[ ]:


finalDf = pd.concat([principalDf, train_df[['target']]], axis = 1)


# In[ ]:


import matplotlib.pyplot as plt
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
targets = [1.0, 0.0]
colors = ['r', 'g']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['target'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()


# In[ ]:


fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
targets = [1.0]
colors = ['r']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['target'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 2']
               , finalDf.loc[indicesToKeep, 'principal component 1']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()


# Another way to use PCA. 
# By specifying PCA(.95) we mean that we need 95% of variance to be covered. So out might be multiple principal, instead just 2.

# In[ ]:


#pca = PCA(n_components=2)
pca = PCA(.95)
P_X = pca.fit_transform(X)
P_X_TEST = pca.transform(X_test)


# Shows number principal components required to cover 95% of variance

# In[ ]:


pca.n_components_


# In[ ]:


pca.explained_variance_ratio_


# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# P_X_T=scaler.fit_transform(P_X)
# P_X_TEST_T=scaler.transform(P_X_TEST)

# Running a model by taking the output of PCA(111 principal components).

# In[ ]:


from catboost import CatBoostClassifier,Pool
train_pool = Pool(P_X,y)
m = CatBoostClassifier(iterations=3000,eval_metric="AUC", objective="Logloss",learning_rate=0.003)
m.fit(P_X,y,silent=True)
y_pred1 = m.predict_proba(P_X_TEST)[:,0]


# In[ ]:


sub1 = pd.DataFrame({"ID_code": test_df.ID_code.values})
sub1["target"] = y_pred1
sub1.to_csv("submission1.csv", index=False)


# # Please upvote, if you find this kernel interesting
