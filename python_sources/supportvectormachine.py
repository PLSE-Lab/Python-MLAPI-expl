#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from datetime import datetime, date
import os
print(os.listdir("../input"))


# In[ ]:


bought_df = pd.read_csv('../input/final-bought-dataset.csv')
bought_df.head()


# In[ ]:


len(bought_df), len(bought_df[bought_df['purchased']==1]), len(bought_df[bought_df['purchased']==0])


# In[ ]:


df = pd.DataFrame(bought_df)

# df['dow_first'] = df['dow_first'].cat.codes


# In[ ]:


from sklearn.model_selection import train_test_split

X = df.drop(['purchased'], axis=1)
y = df[['purchased']]
X_train, X_validate, y_train, y_validate = train_test_split(X, y, test_size=0.25)
X_train.shape, X_validate.shape, y_train.shape, y_validate.shape


# In[ ]:


## SVM
from sklearn import svm

clf = svm.SVC(kernel='rbf')
clf.fit(X_train, y_train.values.ravel())

y_pred = clf.predict(X_validate)


# In[ ]:


from sklearn.metrics import roc_curve, auc, accuracy_score
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_true=y_validate, y_score=y_pred)
roc_auc = auc(false_positive_rate, true_positive_rate)
print("roc_auc", roc_auc)
print("accuracy_score", accuracy_score(y_pred, y_validate))


# In[ ]:


y_pred = pd.DataFrame(y_pred, index=y_validate.index, columns=['purchased'])
what_to_buy_df = pd.merge(X_validate, y_pred, left_index=True, right_index=True)
what_to_buy_df.head(3)


# In[ ]:


unique_sid = what_to_buy_df['session'].unique()

fp = open("solution.dat","w")

for sid in unique_sid:
#     all_items = what_to_buy_df['session'==sid , 'item']
    bought_items = what_to_buy_df.loc[(what_to_buy_df['session']==sid) & (what_to_buy_df['purchased']==1), 'item'].values
    if len(bought_items)>0:
        items = ','.join(map(str, bought_items))
        print(f"{sid};{items}") 
        
        print(f"{sid};{','.join(map(str, bought_items))}", file=fp) 


fp.close()


# In[ ]:


# A sample session id with what to buy and not buy
what_to_buy_df[what_to_buy_df['session']==11]

