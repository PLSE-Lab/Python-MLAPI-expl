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


train=pd.read_csv('../input/train.csv')
test=pd.read_csv('../input/test.csv')
sample_submission=pd.read_csv('../input/sample_submission.csv')


# In[ ]:


print(train.shape,test.shape)


# In[ ]:


train.target.value_counts()*100/train.shape[0]


# ## Imbalanced dataset

# In[ ]:


train.isnull().sum().sort_values(ascending=False).head()


# In[ ]:


train.isnull().sum().sort_values(ascending=False).head()


# ## Good news! No null in train as well as test

# ### Modelling

# In[ ]:


weights={0:1,1:9}
target = train["target"]
train = train.drop(columns=["ID_code","target"],axis=1)
test_df = test.drop(columns=["ID_code"],axis=1)


# In[ ]:


from sklearn.model_selection import train_test_split,GridSearchCV
x_train,x_test,y_train,y_test=train_test_split(train,target,test_size=0.2)


# In[ ]:


from lightgbm import LGBMClassifier


# In[ ]:


lgbm=LGBMClassifier(class_weight=weights)


# In[ ]:


param={'max_depth':[5,7,12],'n_estimators':[300,500],'learning_rate':[0.01,0.1,1,3],'num_leaves':[50,60]}


# In[ ]:


gr=GridSearchCV(cv=5,estimator=lgbm,error_score='auc',n_jobs=16,verbose=True,param_grid=param)


# In[ ]:


#gr.fit(x_train,y_train)


# In[ ]:


gr.best_score_


# In[ ]:


gr.best_params_


# In[ ]:


pred_gr=gr.predict(x_test)


# In[ ]:


from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
print(accuracy_score(y_test,pred_gr))
print(confusion_matrix(y_test,pred_gr))
print(classification_report(y_test,pred_gr))


# In[ ]:


pred_gr=gr.predict(test)


# In[ ]:


sub_df_gr = pd.DataFrame({"ID_code":test["ID_code"].values})
sub_df_gr["target"] = pred_gr
sub_df_gr.to_csv("gr_pred.csv", index=False)


# In[ ]:





# In[ ]:




