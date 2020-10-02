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

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import re
from glob import glob
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import os
import sys
import gc
from sklearn.model_selection import cross_val_score

files = glob('../input/*.csv')
files


# In[ ]:


for each in files:
    if 'depart' in each:
        deps = pd.read_csv(each)
    if 'products.csv' in each:
        prods = pd.read_csv(each)
    if 'orders.csv' in each:
        ords = pd.read_csv(each)
    if 'aisles.csv' in each:
        aisl = pd.read_csv(each)
    if 'train.csv' in each:
        ordt = pd.read_csv(each)


# In[ ]:


joined_df = pd.merge(left=ords[ords['eval_set']=='train'], right=ordt, on='order_id')
joined_df = pd.merge(left=joined_df, right=prods, on='product_id')
joined_df = pd.merge(left=joined_df, right=deps, on='department_id')
joined_df = pd.merge(left=joined_df, right=aisl, on='aisle_id')


# In[ ]:


joined_df.columns.tolist()


# In[ ]:


topn = joined_df['product_id'].value_counts()[:100]
topn = topn.keys()
for item in topn:
    print(item)
    joined_df[item]=0
    joined_df.loc[joined_df['product_id']==item, item]=1


# In[ ]:


X=joined_df[['user_id', 'order_dow', 'order_hour_of_day', 'days_since_prior_order']].values.tolist()
Y=joined_df[topn].values.tolist()


# In[ ]:


from keras.models import Sequential
from keras.layers import Dense

# fix random seed for reproducibility
np.random.seed(7)


# In[ ]:





# In[ ]:


topn = joined_df['product_id'].value_counts()[:100]
topn = topn.keys()
for item in topn:
    print(item)
    joined_df[item]=0
    joined_df.loc[joined_df['product_id']==item, item]=1

X=joined_df[['user_id', 'order_dow', 'order_hour_of_day', 'days_since_prior_order']].values.tolist()
Y=joined_df[topn].values.tolist()

from keras.models import Sequential
from keras.layers import Dense

# fix random seed for reproducibility
np.random.seed(7)

model = Sequential()
model.add(Dense(12, input_dim=4, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(100, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, Y, epochs=20, batch_size=10)


# In[ ]:


model.fit(X, Y, epochs=20, batch_size=10)


# In[ ]:


joined_df.groupby(['department','product_name'])['order_id'].count().plot()


# In[ ]:


aisles = aisl['aisle_id'].values.tolist()
departs = deps['department_id'].values.tolist()

top_aisles = joined_df['aisle_id'].value_counts()[:20]
top_aisles = top_aisles.keys()

out_df = pd.DataFrame()
n_feats= ['user_id','order_dow','days_since_prior_order']
x_test = ords[ords['eval_set']=='test'][n_feats]


# In[ ]:


for ea in top_aisles:
    for ed in departs:
        train_on = joined_df[(joined_df['aisle_id']==ea)&(joined_df['department_id']==ed)]
         
        if train_on.shape[0]>0:

            train_off = joined_df[(joined_df['aisle_id']!=ea)&(joined_df['department_id']!=ed)]
            train_off['product_id']=-1
            train_off = train_off.drop_duplicates(subset=n_feats+['aisle_id','department_id'])
            train_off = train_off.sample(int(train_on.shape[0]/2))
            
            all_train = pd.concat([train_on, train_off])
            
            all_train = all_train.sample(n=min(20000, int(0.5*all_train.shape[0])))
            vc = all_train['product_id'].value_counts()
            to_remove=[]
            for val, count in vc.iteritems():
                if count<3:
                    to_remove.append(val)

            all_train = all_train[~all_train['product_id'].isin(to_remove)]
            
            print(all_train[all_train['product_id']==-1].shape[0])
            print(all_train[all_train['product_id']!=-1].shape[0])
            X = all_train[n_feats+['aisle_id','department_id']]
            Y = all_train['product_id']
            
            
            x_test.loc[:,'aisle_id'] = ea
            x_test.loc[:,'deparment_id'] = ed
            
            clf = RandomForestClassifier(n_jobs=-1, random_state=1)
            scores = cross_val_score(clf, X, Y, cv=3)
            print(ea,ed,len(X))
            print(scores)
            y_test = clf.fit(X,Y).predict(x_test)

            #out_df[str(ea)+'_'+str(ed)] = y_test
            gc.collect()
                  


# In[ ]:


out_df.head(20)


# In[ ]:


pro_cols = out_df.columns.tolist()
pro_cols = pro_cols[:-2]
pro_cols


# In[ ]:


def process(row):
    out_str=[]
    for index, value in row.iteritems():
        if value!=-1:
            out_str.append(str(value))
    if len(out_str)>0:
        return ' '.join(out_str)
    else:
        return 'None'

out_df['products'] = out_df[pro_cols].apply(lambda x: process(x), axis=1)


# In[ ]:


out_df['products'].head(10)


# In[ ]:


out_df['order_id']=ords[ords['eval_set']=='test']['order_id'].values.tolist()
out_df['order_id'].head(10)


# In[ ]:


out_df[out_df['products']=='None'].head(10)


# In[ ]:


out_df[['order_id','products']].to_csv('new_out_2.csv')


# In[ ]:


out_df[out_df['products'].str.len()<=105].shape


# In[ ]:




