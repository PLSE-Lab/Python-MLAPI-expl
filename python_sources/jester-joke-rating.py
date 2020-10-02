#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
print(os.listdir("../input"))


# In[ ]:


train=pd.read_csv("../input/train.csv")
test=pd.read_csv("../input/test.csv")


# In[ ]:


sample=pd.read_csv("../input/sample_submission_5ms57N3.csv")
sample.head(10)


# In[ ]:


joke_df=pd.read_csv('../input/jokes.csv')
joke_df.shape


# In[ ]:


group=train.groupby('joke_id').size()
group.plot()


# In[ ]:


group_test=test.groupby('joke_id').size()
group_test.values


# In[ ]:


import plotly
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)

trace0=go.Scatter(
    x=group.index,
    y=group.values,
    name='no of users for particular joke in train data'
    
)

trace1=go.Scatter(
    x=group_test.index,
    y=group_test.values,
    name='no of users for particular joke in test data'
    
)

plotly.offline.iplot({
"data":[trace0,trace1],
"layout": go.Layout(title="No of users for particular joke in data")
})


# In[ ]:


joke_df.columns


# In[ ]:


merge_train=pd.merge(train,joke_df,on='joke_id',how='left')
joke_text=merge_train["joke_text"]
y=train["Rating"]
train.drop("Rating",axis=1,inplace=True)


# In[ ]:


def agg_fuction(df,prefix):
    agg_fuc={
    'joke_id':['mean','sum','min','max']
    }
    agg_df=df.groupby('user_id').agg(agg_fuc)
    agg_df.columns = [ '_'.join(col).strip() 
                           for col in agg_df.columns.values]
    agg_df.reset_index(inplace=True)
    
    df1 = (df.groupby('user_id')
          .size()
          .reset_index(name='{}users_count'.format(prefix)))
    
    agg_df = pd.merge(df1, agg_df, on='user_id', how='left')
    return agg_df


# In[ ]:


agg_train=agg_fuction(train,'hist')
agg_test=agg_fuction(test,'hist')
agg_test.head()


# In[ ]:


merge_train=pd.merge(train,agg_test,on='user_id',how='left')


# In[ ]:


m=pd.merge(merge_train,joke_df,on='joke_id',how='left')


# In[ ]:


#users for joke 2 to 9 is high in both the dataset
from sklearn.feature_extraction.text import TfidfVectorizer

vect=TfidfVectorizer()
X=vect.fit_transform(m['joke_text'])


# In[ ]:


X[0:1,:]


# In[ ]:


num_feats = merge_train.values

from scipy import sparse

training_data = sparse.hstack((X, num_feats.astype(np.int64)))


# In[ ]:


merge_test=pd.merge(test,joke_df,on='joke_id',how='left')


# In[ ]:


merge_test=pd.merge(merge_test,agg_test,on='user_id',how='left')


# In[ ]:




vect1=TfidfVectorizer()
X1=vect.fit_transform(merge_test["joke_text"])


# In[ ]:


merge_test.drop("joke_text",axis=1,inplace=True)
merge_test.head()


# In[ ]:


num_feats1=merge_test.values

test_data=sparse.hstack((X1,num_feats1.astype(np.int64)))


# In[ ]:



feature_names=list(merge_train.columns)
feature_names_tf=list(vect.get_feature_names())
#feature_names.append([vect.get_feature_names()])
# vect.get_feature_names()
# feature_names=feature_names+vect.get_feature_names()
feature_names=feature_names+feature_names_tf
print(len(feature_names))


# In[ ]:


import lightgbm as lgb

d_train=lgb.Dataset(training_data,label=y,feature_name=feature_names)
params = {'num_leaves': 45,
         'min_data_in_leaf': 30, 
         'objective':'regression',
         'max_depth': -1,
         'learning_rate': 0.015,
         "boosting": "gbdt",
         "feature_fraction": 0.9,
         "bagging_freq": 1,
         "bagging_fraction": 0.9,
         "bagging_seed": 11,
         "metric": 'rmse',
         "lambda_l1": 0.1,
         "verbosity": -1,
         "nthread": 4,
         "random_state": 4950}


clf=lgb.train(params,d_train,110)


# In[ ]:


y_pred=clf.predict(test_data)


# In[ ]:


# from sklearn.metrics import mean_squared_error

# err=mean_squared_error(y,y_pred)**0.5
# err


# In[ ]:


sample=pd.read_csv('../input/sample_submission_5ms57N3.csv')
sample["Rating"]=y_pred
sample.to_csv("jester_submission.csv",index=False)


# In[ ]:





# In[ ]:




