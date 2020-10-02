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


# #Data Overview and Understanding

# In[ ]:


import os

for file in os.listdir("../input/"):
    print('-'*20,file,'-'*20,'\n')
    df = pd.read_csv("../input/" + file)
    print('Data shape : '.ljust(30), df.shape)
    print('Data Info : ', df.info())
    print('Data View : \n', df.head())
    print('\n')


# **Read Data**

# In[ ]:


df_train = pd.read_csv('../input/train.csv')
df_test  = pd.read_csv('../input/test.csv')


# **Target Variable:**
# 
# "y" is the variable we need to predict. So let us do some analysis on this variable first.

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8,6))
plt.scatter(range(df_train.shape[0]), np.sort(df_train.y.values))
plt.xlabel('index', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.show()


# Note - There a single point lying way above rest of the values. Y is distributed between 70 to 170
# 
# **Distribution of Y**

# In[ ]:


ulimit = 180
df_train['y'].ix[df_train['y']>ulimit] = ulimit

plt.figure(figsize=(12,8))
sns.distplot(df_train.y.values, bins=50, kde=False)
plt.xlabel('y value', fontsize=12)
plt.show()


# **Data Types**

# In[ ]:


dtype_df = df_train.dtypes.reset_index()
dtype_df.columns = ["Count", "Column Type"]
dtype_df.groupby("Column Type").aggregate('count').reset_index()


# **Missing values:**
# 
# Let us now check for the missing values.

# In[ ]:


missing_df = df_train.isnull().sum(axis=0).reset_index()
missing_df.columns = ['column_name', 'missing_count']
missing_df = missing_df.ix[missing_df['missing_count']>0]
missing_df = missing_df.sort_values(by='missing_count')
missing_df


# **Integer Columns**

# In[ ]:


unique_values_dict = {}
for col in df_train.columns:
    if col not in ["ID", "y", "X0", "X1", "X2", "X3", "X4", "X5", "X6", "X8"]:
        unique_value = str(np.sort(df_train[col].unique()).tolist())
        tlist = unique_values_dict.get(unique_value, [])
        tlist.append(col)
        unique_values_dict[unique_value] = tlist[:]
for unique_val, columns in unique_values_dict.items():
    print("Columns containing the unique values : ",unique_val)
    print(columns)
    print("--------------------------------------------------")


# **Categorical Columns**

# In[ ]:


var_name = "X0"
col_order = np.sort(df_train[var_name].unique()).tolist()
plt.figure(figsize=(12,6))
sns.stripplot(x=var_name, y='y', data=df_train, order=col_order)
plt.xlabel(var_name, fontsize=12)
plt.ylabel('y', fontsize=12)
plt.title("Distribution of y variable with "+var_name, fontsize=15)
plt.show()


# In[ ]:


var_name = "X2"
col_order = np.sort(df_train[var_name].unique()).tolist()
plt.figure(figsize=(12,6))
sns.boxplot(x=var_name, y='y', data=df_train, order=col_order)
plt.xlabel(var_name, fontsize=12)
plt.ylabel('y', fontsize=12)
plt.title("Distribution of y variable with "+var_name, fontsize=15)
plt.show()


# In[ ]:


var_name = "X3"
col_order = np.sort(df_train[var_name].unique()).tolist()
plt.figure(figsize=(12,6))
sns.violinplot(x=var_name, y='y', data=df_train, order=col_order)
plt.xlabel(var_name, fontsize=12)
plt.ylabel('y', fontsize=12)
plt.title("Distribution of y variable with "+var_name, fontsize=15)
plt.show()


# In[ ]:


var_name = "X4"
col_order = np.sort(df_train[var_name].unique()).tolist()
plt.figure(figsize=(12,6))
sns.violinplot(x=var_name, y='y', data=df_train, order=col_order)
plt.xlabel(var_name, fontsize=12)
plt.ylabel('y', fontsize=12)
plt.title("Distribution of y variable with "+var_name, fontsize=15)
plt.show()


# In[ ]:


var_name = "X5"
col_order = np.sort(df_train[var_name].unique()).tolist()
plt.figure(figsize=(12,6))
sns.boxplot(x=var_name, y='y', data=df_train, order=col_order)
plt.xlabel(var_name, fontsize=12)
plt.ylabel('y', fontsize=12)
plt.title("Distribution of y variable with "+var_name, fontsize=15)
plt.show()


# In[ ]:


var_name = "X6"
col_order = np.sort(df_train[var_name].unique()).tolist()
plt.figure(figsize=(12,6))
sns.boxplot(x=var_name, y='y', data=df_train, order=col_order)
plt.xlabel(var_name, fontsize=12)
plt.ylabel('y', fontsize=12)
plt.title("Distribution of y variable with "+var_name, fontsize=15)
plt.show()


# In[ ]:


var_name = "X8"
col_order = np.sort(df_train[var_name].unique()).tolist()
plt.figure(figsize=(12,6))
sns.boxplot(x=var_name, y='y', data=df_train, order=col_order)
plt.xlabel(var_name, fontsize=12)
plt.ylabel('y', fontsize=12)
plt.title("Distribution of y variable with "+var_name, fontsize=15)
plt.show()


# print(df_train.shape)
# print(df_test.shape)

# **Convert Categorical features to numerical features - Risk Table Approach**

# In[ ]:


str_columns = []
df_test_r = df_train.copy()

for x in df_train.columns:
    if(df_train[x].dtypes==object):
        str_columns.append(x)

def risk_score(df_r, df, var):
    tmp = df_r.groupby(var)['y'].mean().reset_index()
    tmp.columns = [var, 'r_' + var]
    out_df = pd.merge(df, tmp, on = var, how = 'left')
    del out_df[var]
    return out_df

for x in str_columns:
    df_train = risk_score(df_train, df_train, x)

for x in str_columns:
    df_test = risk_score(df_test_r, df_test, x)


# #View Corr of features with y

# In[ ]:


import numpy

for x in df_train.columns:
    print(numpy.corrcoef(df_train[x],df_train['y'])[0,1])


# #Run basic XGBoost with all features

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as snsimport xgboost as xgb

y_train = df_train['y'].values
X_train = df_train.drop(["ID", "y"], axis=1)
X_test  = df_test.drop(["ID"], axis=1)

# Thanks to anokas for this #
def xgb_r2_score(preds, dtrain):
    labels = dtrain.get_label()
    return 'r2', r2_score(labels, preds)

xgb_params = {
    'eta': 0.05,
    'max_depth': 6,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'objective': 'reg:linear',
    'silent': 1
}
dtrain = xgb.DMatrix(X_train, y_train, feature_names=X_train.columns.values)
model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=100, feval=xgb_r2_score, maximize=True)

# plot the important features #
fig, ax = plt.subplots(figsize=(12,18))
xgb.plot_importance(model, max_num_features=50, height=0.8, ax=ax)
plt.show()

#write output
pred_time = pd.DataFrame()
dtest = xgb.DMatrix(X_test, feature_names=X_test.columns.values)
pred_time['y'] = model.predict(dtest)
pred_time['ID'] = df_test['ID']
print(pred_time.shape)
pred_time.to_csv('xgb_out.csv', index = False)


# In[ ]:




