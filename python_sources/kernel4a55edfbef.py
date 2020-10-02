#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[12]:


brain_df=pd.read_csv('../input/emotions.csv')
brain_df.head()


# 

# Firstly need to understand the distribution of the data

# In[13]:


import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(12,5))
sns.countplot(x=brain_df.label,color='mediumseagreen')
plt.title('Emotional sentiment class,fontsize=20')
plt.ylabel('class counts',fontsize=18)
plt.xlabel('class counts',fontsize=18)
plt.xticks(rotation='vertical')


# In[ ]:


brain_df.count


# In[14]:


label_df=brain_df['label']
brain_df.drop('label',axis=1,inplace=True)
brain_df.head()


# We are going to use random forest approach

# In[16]:


get_ipython().run_cell_magic('time', '', "from sklearn.pipeline import Pipeline\nfrom sklearn.ensemble import RandomForestClassifier\nfrom sklearn.model_selection import cross_val_score,train_test_split\npl_random_forest=Pipeline(steps=[('random_forest',RandomForestClassifier())])\nscores=cross_val_score(pl_random_forest,brain_df,label_df,cv=10,scoring='accuracy')\nprint('Accuracy for RandomForest:',scores.mean())")


# Now we will try Logistic Regression 

# In[17]:


get_ipython().run_cell_magic('time', '', "from sklearn.pipeline import Pipeline\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.linear_model import LogisticRegression\nfrom sklearn.model_selection import cross_val_score,train_test_split\npl_log_reg=Pipeline(steps=[('scaler',StandardScaler()),('log_reg',LogisticRegression(multi_class='multinomial',solver='saga',max_iter=200))])\nscores=cross_val_score(pl_log_reg,brain_df,label_df,cv=10,scoring='accuracy')\nprint('Accuracy for Logistic Regression:',scores.mean())")


# curse of dimentionality affect the accuracy and time ,why dont we try out pca to reduce the dimentionality of the data

# In[19]:


get_ipython().run_cell_magic('time', '', 'from sklearn.decomposition import PCA\nfrom sklearn.preprocessing import StandardScaler\nscaler=StandardScaler()\nscale=scaler.fit_transform(brain_df)\npca=PCA(n_components=20)\npca_vectors=pca.fit_transform(scale)\nfor index,var in enumerate(pca.explained_variance_ratio_):\n    print(\'variances by PCA\',(index+1),":",var)')


# In[21]:


import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(25,8))
sns.scatterplot(x=pca_vectors[:,0],y=pca_vectors[:,1],hue=label_df)
plt.title('PCA distribution ,fontsize=20')
plt.ylabel('PCA 1',fontsize=18)
plt.xlabel('PCA 2',fontsize=18)
plt.xticks(rotation='vertical')


# In[22]:


get_ipython().run_cell_magic('time', '', "from sklearn.pipeline import Pipeline\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.linear_model import LogisticRegression\nfrom sklearn.model_selection import cross_val_score,train_test_split\npl_log_reg_pca=Pipeline(steps=[('scaler',StandardScaler()),('pca',PCA(n_components=2)),('log_reg',LogisticRegression(multi_class='multinomial',solver='saga',max_iter=200))])\nscores=cross_val_score(pl_log_reg_pca,brain_df,label_df,cv=10,scoring='accuracy')\nprint('Accuracy for Logistic Regression 2 PCA component:',scores.mean())")


# Try to increase the pca components

# In[23]:


get_ipython().run_cell_magic('time', '', "from sklearn.pipeline import Pipeline\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.linear_model import LogisticRegression\nfrom sklearn.model_selection import cross_val_score,train_test_split\npl_log_reg_pca=Pipeline(steps=[('scaler',StandardScaler()),('pca',PCA(n_components=10)),('log_reg',LogisticRegression(multi_class='multinomial',solver='saga',max_iter=200))])\nscores=cross_val_score(pl_log_reg_pca,brain_df,label_df,cv=10,scoring='accuracy')\nprint('Accuracy for Logistic Regression 2 PCA component:',scores.mean())")


# In[25]:


get_ipython().run_cell_magic('time', '', "from sklearn.pipeline import Pipeline\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.linear_model import LogisticRegression\nfrom sklearn.model_selection import cross_val_score,train_test_split\npl_log_reg_pca=Pipeline(steps=[('scaler',StandardScaler()),('pca',PCA(n_components=20)),('log_reg',LogisticRegression(multi_class='multinomial',solver='saga',max_iter=200))])\nscores=cross_val_score(pl_log_reg_pca,brain_df,label_df,cv=10,scoring='accuracy')\nprint('Accuracy for Logistic Regression 2 PCA component:',scores.mean())")


# here am going to use MLClassifier like ANN

# In[26]:


get_ipython().run_cell_magic('time', '', "from sklearn.pipeline import Pipeline\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.neural_network import MLPClassifier\npl_mlp=Pipeline(steps=[('scaler',StandardScaler()),('mil_ann',MLPClassifier(hidden_layer_sizes=(1275,637)))])\nscores=cross_val_score(pl_mlp,brain_df,label_df,cv=10,scoring='accuracy')\nprint('ANN:',scores.mean())")


# In[27]:


get_ipython().run_cell_magic('time', '', "from sklearn.pipeline import Pipeline\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.svm import LinearSVC\npl_svm=Pipeline(steps=[('scaler',StandardScaler()),('svm',LinearSVC())])\nscores=cross_val_score(pl_svm,brain_df,label_df,cv=10,scoring='accuracy')\nprint('SVM:',scores.mean())")


# We will try with xgboost algorithm for the better performance

# In[28]:


get_ipython().run_cell_magic('time', '', "from sklearn.pipeline import Pipeline\nimport xgboost as xgb\npl_xgb=Pipeline(steps=[('svm',xgb.XGBClassifier(objective='multi:softmax'))])\nscores=cross_val_score(pl_xgb,brain_df,label_df,cv=10,scoring='accuracy')\nprint('XGBoost:',scores.mean())")

