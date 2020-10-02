#!/usr/bin/env python
# coding: utf-8

# ## Example (Simple Linear Regression for Red Wine)

# In[ ]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from seaborn import regplot 
import matplotlib.pyplot as plt


# In[ ]:


df = pd.read_csv('../input/wine-whitered/Wine_red.csv', sep=';')
df.head()


# In[ ]:


regplot(x='density',y='quality',data=df)


# ## Determine the importance of each feature for predicting the quality of red wine. Compute train adn test R2 for each feature

# In[ ]:


(df_train,df_test) = train_test_split(df,
                                     train_size=0.8,
                                     test_size=0.2,
                                     random_state =0)


# In[ ]:


feature_train = df_train.iloc[:,0]


# In[ ]:


feature_train.values.reshape(-1,1).shape


# In[ ]:


df.columns[0]


# In[ ]:


lr = LinearRegression()

feature_name = []
R2_train = []
R2_test = []

target_train = df_train.iloc[:,-1]
target_test = df_test.iloc[:,-1]

for k in range(df.shape[1]-1):
    feature_train = df_train.iloc[:,k]
    feature_test = df_test.iloc[:,k]
    lr.fit(feature_train.values.reshape(-1,1),target_train)
    feature_name.append(df.columns[k])
    R2_train.append(lr.score(feature_train.values.reshape(-1,1),target_train))
    R2_test.append(lr.score(feature_test.values.reshape(-1,1),target_test))
    
results =pd.DataFrame()
results['feature'] = feature_name
results['train R2'] = R2_train
results['test R2'] = R2_test
results.sort_values('test R2',ascending = False).head(5)


# In[ ]:




