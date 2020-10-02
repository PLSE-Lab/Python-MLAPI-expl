#!/usr/bin/env python
# coding: utf-8

# In[1]:


import xgboost
import numpy as np
import pandas as pd
from math import sqrt

import seaborn as sns
sns.set(style="white", color_codes=True)


import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn import cross_validation, metrics
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score


# In[2]:


train = pd.read_excel("../input/research_student (1).xlsx")


# In[3]:


train.head()


# In[14]:


train['CGPA_Gr_7'] = np.where(train['CGPA']>=7, train['CGPA'], 'NaN')


# In[16]:


train


# In[10]:



train['CGPA_Gr_7']=train.loc[train['CGPA'] > 7] 
#df.loc[df['A'] > df['B'], 'C'] = 1
#df.loc[df['A'] < df['B'], 'C'] = -1


# In[ ]:





# In[ ]:


train.info


# In[ ]:


train.describe()


# In[ ]:


sns.FacetGrid(train, hue="Branch", size=5)    .map(plt.scatter, "Marks[10th]", "Marks[12th]")    .add_legend()


# In[ ]:


train.info()


# In[ ]:


train.Branch.value_counts()


# In[ ]:


train.shape


# In[ ]:


print(train["Branch"])


# In[ ]:


train.Branch.value_counts()


# In[ ]:


train.Gender.value_counts()


# In[ ]:


train.Category.value_counts()


# In[ ]:


train.Board[12th].value_counts()


# In[ ]:





# In[ ]:


print(train.loc[5])


# **performing scaling**

# In[ ]:


train.columns


# In[ ]:


train[['Gender','Branch']]


# In[ ]:


scale_list =[ 'Marks[10th]', 'Marks[12th]',
      'GPA 1', 'Rank', 'Normalized Rank', 'CGPA',
       'Current Back', 'Ever Back', 'GPA 2', 'GPA 3', 'GPA 4', 'GPA 5',
       'GPA 6', 'Olympiads Qualified', 'Technical Projects', 'Tech Quiz',
       'Engg. Coaching', 'NTSE Scholarships', 'Miscellany Tech Events']
sc = train[scale_list]


# In[ ]:


sc.head()


# In[ ]:



sc.drop([0,221,222])


# In[ ]:


sc.head()


# In[ ]:


train  = train.fillna(0)


# In[ ]:


sc = sc.fillna(0)


# In[ ]:


scaler = StandardScaler()
sc = scaler.fit_transform(sc)


# In[ ]:


train[scale_list] = sc


# In[ ]:


train[scale_list].head()


# In[ ]:


sc[0]


# **DATA VISUALIZATIO**

# In[ ]:


train[scale_list].plot(kind="scatter", x="Marks[10th]", y="Marks[12th]")


# In[ ]:


train[scale_list].plot(kind="scatter", x="Marks[10th]", y="Marks[12th]")


# **This show  that student whose 10th and 12th marks are high are more participated in tech activities than others**

# In[ ]:



sns.FacetGrid(train[scale_list], hue="Miscellany Tech Events", size=5)    .map(plt.scatter, "Marks[10th]", "Marks[12th]")    .add_legend()


# In[ ]:


train.columns


# In[ ]:


train= train.dropna()


# In[ ]:


train.head()


# In[ ]:




