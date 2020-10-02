#!/usr/bin/env python
# coding: utf-8

# # Introduction

# The analysis of "Admission_Predict.csv" to alayze the student performance during exam by using K-means Algorithm  

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import warnings 
warnings.filterwarnings('ignore')

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# ## Load dataset

# In[ ]:


base_dataset1=pd.read_csv("../input/admissionpredictioncsv/Admission_Predict.csv")


# In[ ]:


base_dataset=base_dataset1


# In[ ]:


base_dataset.head()


# ### Null Values Treatmenet

# In[ ]:


base_dataset.isna().sum()


# ## Analysis

# In[ ]:


base_dataset.columns


# In[ ]:


base_dataset.shape


# In[ ]:


base_dataset.var()


# In[ ]:


base_dataset=base_dataset[['GRE Score','TOEFL Score']]


# ## visualization

# In[ ]:


sns.boxplot(base_dataset['GRE Score'])


# In[ ]:


sns.boxplot(base_dataset['TOEFL Score'])


# ### univariate and Bivariate 

# In[ ]:


base_dataset.describe()


# In[ ]:


import pandas_profiling


# In[ ]:


base_dataset.profile_report(style={'full_width':True})


# In[ ]:


base_dataset.head()


# In[ ]:


from sklearn.preprocessing import MinMaxScaler


# In[ ]:


mn=MinMaxScaler()
mn.fit(base_dataset)
test=mn.transform(base_dataset)
test=pd.DataFrame(test,columns=['GRE Score','TOEFL Score'])


# In[ ]:


test.describe()


# In[ ]:


from sklearn.cluster import KMeans
km=KMeans(n_clusters=3)
km.fit(test)
km.labels_


# In[ ]:


base_dataset1['cluster']=km.labels_


# In[ ]:


base_dataset1[base_dataset1['cluster']==0].shape


# In[ ]:


base_dataset1[base_dataset1['cluster']==1].shape


# In[ ]:


base_dataset1[base_dataset1['cluster']==2].shape


# In[ ]:


base_dataset1[base_dataset1['cluster']==0]['GRE Score'].min(),base_dataset1[base_dataset1['cluster']==0]['GRE Score'].max()


# ## Row Index of dataset students having less than the average value of GRE Score

# In[ ]:


avg_c1=base_dataset1[base_dataset1['cluster']==0]['GRE Score'].mean()
cluster1=base_dataset1[base_dataset1['cluster']==0]['GRE Score']
cluster1[cluster1<avg_c1].index


# In[ ]:


x=[]
for i in cluster1.values:
    if (i-cluster1.values.mean())<0:
        x.append(abs(i-cluster1.values.mean()))


# In[ ]:


x=np.array(x)


# In[ ]:


x


# ## Total Student having less than average value 

# In[ ]:


len(x)


# ### Total 78 students need to pay attention on study as to improve GRE Score
