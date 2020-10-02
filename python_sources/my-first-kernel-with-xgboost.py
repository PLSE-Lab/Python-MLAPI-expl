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

import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
# Any results you write to the current directory are saved as output.


# In[ ]:


train = pd.read_csv("../input/train/train.csv")
test = pd.read_csv("../input/test/test.csv")
breeds = pd.read_csv("../input/breed_labels.csv")
colors = pd.read_csv("../input/color_labels.csv")
states = pd.read_csv("../input/state_labels.csv")


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


breeds.head()


# In[ ]:


colors.head()


# In[ ]:


states.head()


# In[ ]:


train.isnull().sum()


# In[ ]:


train.dtypes


# In[ ]:


target=train['AdoptionSpeed']
target.head()


# In[ ]:


df_train=train.drop(['Name','RescuerID','Description','PetID','AdoptionSpeed'],axis=1)
df_test=test.drop(['Name','RescuerID','Description','PetID'],axis=1)


# In[ ]:


xg=xgb.XGBClassifier()
xg.fit(df_train,target)


# In[ ]:


pred=xg.predict(df_test)


# In[ ]:


submit=pd.DataFrame()
submit['PetID']=test['PetID']
submit['AdoptionSpeed']=pred
submit.to_csv('submission.csv',index=False)


# In[ ]:


xgb.plot_importance(xg)


# In[ ]:


sns.distplot(target)


# In[ ]:




