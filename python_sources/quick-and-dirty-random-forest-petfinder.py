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


# In[ ]:


df=pd.read_csv('../input/train/train.csv')


# In[ ]:


df.head()


# In[ ]:


df.info()


# Using numerical values 

# In[ ]:


df_obj=df.dtypes[df.dtypes=='object'].index
df_num=df.drop(columns=df_obj)


# Number of N/A values in each feature

# In[ ]:


pd.isnull(df_num).mean()


# Check the unique values of in training set.  

# In[ ]:


for i in df_num.columns:
    print('Feature: "{}", # unique: {}. '.format(i,df_num.loc[:,i].nunique()),df_num.loc[:,i].unique())


# In[ ]:


from sklearn.model_selection import train_test_split
RANDOM_SEED=142
np.random.seed(RANDOM_SEED)
def xy_split(df,y_cols):
    y=df.loc[:,y_cols]
    x=df.drop(columns=y_cols)
    return x,y
x,y=xy_split(df_num,'AdoptionSpeed')
x_train,x_valid,y_train,y_valid=train_test_split(x,y,test_size=0.1,random_state=RANDOM_SEED)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:


m=RandomForestClassifier(n_estimators=100,max_features='log2')
get_ipython().run_line_magic('time', 'm.fit(x_train,y_train)')


# In[ ]:


m.score(x_valid,y_valid)


# In[ ]:


df_test=pd.read_csv('../input/test/test.csv')


# In[ ]:


df_test.head()


# In[ ]:


df_test.info()


# In[ ]:


x_test=df_test.loc[:,x_train.columns]
x_test.info()


# In[ ]:


y_pred=m.predict(x_test)


# In[ ]:


sub=df_test.loc[:,['PetID']].copy()
sub.loc[:,'AdoptionSpeed']=y_pred


# In[ ]:


sub.to_csv('submission.csv',index=False)


# In[ ]:


sub.loc[:,'AdoptionSpeed']=2
sub.to_csv('submission_g2.csv',index=False)

