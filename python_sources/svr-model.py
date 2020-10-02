#!/usr/bin/env python
# coding: utf-8

# ### Loading libraries

# In[ ]:


get_ipython().run_cell_magic('time', '', 'import pandas as pd')


# In[ ]:


import  lightgbm as lgb


# #### Reading test and train data

# In[ ]:


get_ipython().run_cell_magic('time', '', 'import os\nprint(os.listdir("../input"))\ndf_train = pd.read_csv(\'../input/train.csv\')\ndf_test = pd.read_csv(\'../input/test.csv\')\ncombine = [df_train, df_test]')


# #### Displaying head of training and testing

# In[ ]:


print(df_train.head(3))
print(df_test.head(3))


# #### Define column date as datatype date and define new date features

# In[ ]:


# Define column date as datatype date and define new date features
for dataset in combine:
    dataset['date'] = pd.to_datetime(dataset['date'])
    dataset['year'] = dataset.date.dt.year
    dataset['month'] = dataset.date.dt.month
    dataset['day'] = dataset.date.dt.day
    dataset['dayofyear'] = dataset.date.dt.dayofyear
    dataset['dayofweek'] = dataset.date.dt.dayofweek
    dataset['weekofyear'] = dataset.date.dt.weekofyear


# #### Dropping date column

# In[ ]:


dataset.drop('date', axis=1, inplace=True)


# In[ ]:


df_train.head()


# ### Add new features daily aveage sales and monthly average sales

# In[ ]:


df_train['daily_avg']=df_train.groupby(['item','store','dayofweek'])['sales'].transform('mean')
df_train['monthly_avg']=df_train.groupby(['item','store','month'])['sales'].transform('mean')


# In[ ]:


daily_avg=df_train.groupby(['item','store','dayofweek'])['sales'].mean().reset_index()
monthly_avg=df_train.groupby(['item','store','month'])['sales'].mean().reset_index()


# In[ ]:


monthly_avg


# ### Merging new features

# In[ ]:


def merge(x,y,col,col_name):
    x =pd.merge(x, y, how='left', on=None, left_on=col, right_on=col,
            left_index=False, right_index=False, sort=True,
             copy=True, indicator=False,validate=None)
    
    x=x.rename(columns={'sales':col_name})
    return x

df_test=merge(df_test, daily_avg,['item','store','dayofweek'],'daily_avg')
df_test=merge(df_test, monthly_avg,['item','store','month'],'monthly_avg')


# #### Displaying columns in train and testing 

# In[ ]:


print(df_test.columns)
print(df_train.columns)


# #### Dropping columns

# In[ ]:



df_test=df_test.drop(['id'],axis=1)
df_train=df_train.drop(['date'],axis=1)


# In[ ]:


df_test.columns


# #### INPUT size

# In[ ]:


df_train.shape


# In[ ]:


df_test.shape


# In[ ]:


df_train.head(2)


# In[ ]:


df_test.head(2)


# ### Checking for missing  values

# In[ ]:


df_train.isnull().sum()


# In[ ]:


df_test.isnull().sum()


# ### Checking datatypes

# In[ ]:


df_train.dtypes


# In[ ]:


df_test.dtypes


# ### Modelling  

# In[ ]:


#setting parameters for lightgbm
param = {'num_leaves':150, 'max_depth':7,'learning_rate':.05,'max_bin':200}
param['metric'] = ['auc', 'binary_logloss']


# In[ ]:





# In[ ]:


y=pd.DataFrame()
y=df_train['sales']


# In[ ]:


df_train=df_train.drop(['sales'],axis=1)


# In[ ]:


x=df_train


# In[ ]:


get_ipython().run_cell_magic('time', '', 'train_data = lgb.Dataset(x,y)\nmodel =lgb.train(param,train_data,)')


# In[ ]:


get_ipython().run_cell_magic('time', '', 'output=model.predict(df_test)\nresult=pd.DataFrame(output)\nresult')


# ### Submission

# In[ ]:


get_ipython().run_cell_magic('time', '', "test=pd.read_csv('../input/test.csv',usecols=['id'])\nfin=pd.DataFrame(test)\nfin['sales']=result\nfin.to_csv('Sales_Lgm.csv',index=False)\n ")

