#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
from sklearn import preprocessing 
import matplotlib.pyplot as plt
from scipy.stats import skew
from scipy.stats.stats import pearsonr


get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina' #set 'png' here when working on notebook")
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


train = pd.read_csv('../input/dataset/Train_full.csv')
test = pd.read_csv('../input/dataset-v2/Test_small_features.csv')


# In[3]:


test.shape


# In[4]:


train.shape


# In[5]:


train.head()


# In[6]:


test.head()


# In[ ]:


# test['up_down'] = test['hour']
# for i in range(0,test.shape[0]):
#     if i == 0: continue
#     else:
#         if (test.at[i,'body'] > 0):
#             test.at[i-1,'up_down'] = 1
#         else:
#             test.at[i-1,'up_down'] = 0


# In[ ]:


# test.at[test.shape[0]-1, 'up_down'] = 1


# In[ ]:


# multiplier = 2/27
# # print(train.at[0,'EMA_20'])
# train['EMA_26'] = train['Close']
# prev_value = 0
# for i in train['EMA_26'].index:
#     if i == 0: 
#         prev_value = train.at[i,'EMA_26']
#         continue
#     else:
# #         print(all_data.at[i,'EMA_50'])
#         train.at[i,'EMA_26'] = multiplier*train.at[i,'EMA_26'] + prev_value*(1-multiplier)
#         prev_value = train.at[i,'EMA_26']
    
# # print(all_data['EMA_50'])


# In[ ]:


# multiplier = 2/13
# # print(train.at[0,'EMA_20'])
# train['EMA_12'] = train['Close']
# prev_value = 0
# for i in train['EMA_12'].index:
#     if i == 0: 
#         prev_value = train.at[i,'EMA_12']
#         continue
#     else:
# #         print(all_data.at[i,'EMA_50'])
#         train.at[i,'EMA_12'] = multiplier*train.at[i,'EMA_12'] + prev_value*(1-multiplier)
#         prev_value = train.at[i,'EMA_12']
    
# # print(all_data['EMA_50'])


# In[ ]:


# train.head()


# In[ ]:


# multiplier = 2/27
# # print(train.at[0,'EMA_20'])
# test['EMA_26'] = test['Close']
# prev_value = 0
# for i in test['EMA_26'].index:
#     if i == 0: 
#         prev_value = test.at[i,'EMA_26']
#         continue
#     else:
# #         print(all_data.at[i,'EMA_50'])
#         test.at[i,'EMA_26'] = multiplier*train.at[i,'EMA_26'] + prev_value*(1-multiplier)
#         prev_value = test.at[i,'EMA_26']
    
# # print(all_data['EMA_50'])


# In[ ]:


# multiplier = 2/13
# # print(train.at[0,'EMA_20'])
# test['EMA_12'] = test['Close']
# prev_value = 0
# for i in test['EMA_12'].index:
#     if i == 0: 
#         prev_value = test.at[i,'EMA_12']
#         continue
#     else:
# #         print(all_data.at[i,'EMA_50'])
#         test.at[i,'EMA_12'] = multiplier*train.at[i,'EMA_12'] + prev_value*(1-multiplier)
#         prev_value = test.at[i,'EMA_12']
    
# # print(all_data['EMA_50'])


# In[ ]:


# test.head()


# In[7]:


#merge all data
all_data = pd.concat((train.loc[:,'Open':'lag_return_96'],
                      test.loc[:,'Open':'lag_return_96']))
all_data.head()


# In[ ]:


# all_data['closedivopen'] = all_data['Close']/all_data['Open']
# all_data['delta_SMA'] = all_data['SMA_20'] - all_data['SMA_50']
# all_data['delta_EMA'] = all_data['EMA_12'] - all_data['EMA_26']
# all_data['amount'] = all_data['Volume']*all_data['return_2']


# In[8]:


all_data = all_data.drop(['Volume', 'upper_tail','lower_tail'], axis = 1)
# all_data = all_data.drop(['upper_tail','lower_tail'], axis = 1)


# In[11]:


X_train = all_data[:train.shape[0]]
X_test = all_data[train.shape[0]:]
y = train.up_down


# In[ ]:


# X_train = train.drop(['Unnamed: 0', 'Volume', 'upper_tail','lower_tail'], axis = 1)
# X_test = test.drop(['Unnamed: 0', 'Volume', 'upper_tail','lower_tail'], axis = 1)


# In[ ]:


# X_train.head()


# In[10]:


from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
import xgboost


# In[16]:


from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
valid_score = []
num_model = 0
pred_test = np.zeros(len(X_test))
for i in range(1, 9, 1):
    num_model += 1
    model = XGBClassifier(max_depth=5, n_estimators=2000, n_jobs=16, 
                          random_state=4, subsample=0.9, gpu_id=0, 
                          colsample_bytree=0.9, max_bin=512, tree_method='gpu_hist')
    x_train, x_valid, y_train, y_valid = train_test_split(X_train, y, test_size = i*0.1, random_state = 8, shuffle = False)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_valid)
    valid_score.append(accuracy_score(y_pred, y_valid))
    pred_test += model.predict(X_test)/num_model
means = np.mean(valid_score)
var = np.std(valid_score)


# In[17]:


print(means)
print(var)


# In[ ]:


# y = all_data['up_down']
# x_train, x_valid, y_train, y_valid = train_test_split(all_data, y, test_size = 0.2, stratify = dayofweek, random_state = 8, shuffle = True)
# x_train, x_valid, y_train, y_valid = train_test_split(X_train, y, test_size = 0.2, random_state = 8, shuffle = True)
# # x_train = X_train.loc[:,'Open':'lag_return_96']
# # y_train = X_train['up_down']
# # x_valid = X_test.loc[:,'Open':'lag_return_96']
# # y_valid = X_test['up_down']


# In[ ]:


# from xgboost import XGBClassifier
# model = XGBClassifier(max_depth = 5, n_estimators = 2000, n_jobs = 16, random_state = 4, subsample = 0.9, gpu_id = 0, colsample_bytree = 0.9, max_bin = 16, tree_method = 'gpu_hist')
# model.fit(X=x_train,y=y_train,eval_set = [(x_train,y_train),(x_valid, y_valid)], eval_metric = ['logloss'], early_stopping_rounds = 70)
# model.fit(X=x_train,y=y_train,eval_set = [(x_train,y_train),(x_valid, y_valid)], eval_metric = ['error'])


# In[ ]:


# from xgboost import plot_importance
# plot_importance(model, max_num_features = 20)


# In[ ]:


# prediction = model.predict(test.drop(['Unnamed: 0', 'Volume', 'upper_tail','lower_tail','up_down'], axis = 1), ntree_limit = model.best_ntree_limit)


# In[ ]:


# prediction


# In[ ]:


# mysubmit = pd.DataFrame({'up_down': prediction})


# In[ ]:


# mysubmit.shape


# In[ ]:


# mysubmit.to_csv('submission.csv', index=True)

