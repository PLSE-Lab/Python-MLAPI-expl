#!/usr/bin/env python
# coding: utf-8

# # Kaggle-heart disease dataset
# 
# ## 2019-02-20

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier as GBC
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


# In[ ]:


original=pd.read_csv('../input/heart.csv')
print(original.head())
print(original.shape)


# In[ ]:


np.isnan(original.any())


# ## Principal Component Analysis

# In[ ]:


print(original.columns.shape[0])
print(original.columns)


# In[ ]:


data=original.iloc[:,0:13]
target=original.loc[:,'target']
print(data.head())


# In[ ]:


pca=PCA(n_components=13).fit(data)


# In[ ]:


pvr=pd.DataFrame(pca.explained_variance_ratio_)
x=np.array(data.columns)
pvr['columns']=x
print(pvr)


# ## PCA
# 
# With pvr,if I want to reduce data's dimension
# 
# I'll set n_components between two~three
# 
# but data size is small,so try not to reduce data's dimension
# 
# Then try to analysis correlation

# ## Correlation Analysis-Pearson coefficient

# In[ ]:


pearsonMatrix=pd.DataFrame(np.round(original.corr(method='pearson'),2))
pearsonMatrix.sort_values(by='target',ascending=False)


# ## pearson coefficient
# 
# With this matrix,features:cp.thalach.slope and restecg have positive correlation
# 
# other features have negative correlation
# 
# So,Use four features to predict

# ## predict-using SVC

# In[ ]:


features=original.loc[:,['cp','thalach','slope','restecg']]
print(features.head())


# In[ ]:


dataTrain,dataTest, targetTrain,targetTest = train_test_split(features,target,train_size=0.8)


# In[ ]:


heartSVC=SVC().fit(dataTrain,targetTrain)


# In[ ]:


heartSVC


# In[ ]:


pre=heartSVC.predict(dataTrain)


# In[ ]:


print(classification_report(targetTrain,pre))


# In[ ]:


predict=heartSVC.predict(dataTest)
print(classification_report(targetTest,predict))


# ## predict-using GBC

# In[ ]:


heartGBC=GBC(max_depth=2)


# In[ ]:


heartGBC.fit(dataTrain,targetTrain)


# In[ ]:


Gpre=heartGBC.predict(dataTrain)
print(classification_report(targetTrain,Gpre))


# In[ ]:


Gpredict=heartGBC.predict(dataTest)
print(classification_report(targetTest,Gpredict))


# ## predict
# 
# With using GradientBoostingClassifier and SVC,
# 
# GradientBoostingClassifier predict test data's f1-score is better than SVC
# 
# So use GradientBoostingClassifier may be better than SVC
# 
# 

# In[ ]:




