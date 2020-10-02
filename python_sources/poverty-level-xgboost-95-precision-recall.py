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


train=pd.read_csv("../input/train.csv")
test=pd.read_csv("../input/test.csv")
sample_submission=pd.read_csv("../input/sample_submission.csv")


# In[ ]:


train.head()


# In[ ]:


train.shape


# In[ ]:


test.head()


# In[ ]:


test.shape


# In[ ]:


train.Target.value_counts(dropna=False)


# In[ ]:


train['train_or_test']='train'
test['train_or_test']='test'


# In[ ]:


df=pd.concat([train,test])


# In[ ]:


df.shape


# In[ ]:


df=df.drop(columns=['Id','idhogar'])


# In[ ]:


df.select_dtypes(object)


# In[ ]:


df=pd.get_dummies(df)


# In[ ]:


df.shape


# In[ ]:


df.head()


# In[ ]:


modified_train=df[df['train_or_test_train']==1]
modified_test=df[df['train_or_test_train']==0]


# In[ ]:


modified_train.head()


# In[ ]:


modified_train.Target.value_counts(dropna=False)


# In[ ]:


x=modified_train.drop(['Target','train_or_test_test','train_or_test_train'],axis=1)
y=modified_train['Target']
#modified_test=modified_test.drop()


# In[ ]:


x.head()


# In[ ]:


#Train test split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1)


# In[ ]:


from xgboost import XGBClassifier


# In[ ]:


xgb=XGBClassifier(max_depth=10)


# In[ ]:


xgb.fit(x_train,y_train)


# In[ ]:


y_pred=xgb.predict(x_test)


# In[ ]:


from sklearn.metrics import confusion_matrix,classification_report


# In[ ]:


print(classification_report(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))


# In[ ]:


modified_test.drop(columns=['Target','train_or_test_train','train_or_test_test'],axis=1,inplace=True)


# In[ ]:


#xgb.fit(x,y)


# In[ ]:


y_pred_final=xgb.predict(modified_test)


# In[ ]:


y_pred_final


# In[ ]:


sample_submission['final_target']=y


# In[ ]:


sample_submission.head()


# In[ ]:


sample_submission.final_target.value_counts(dropna=False)


# In[ ]:




