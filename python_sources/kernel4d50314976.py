#!/usr/bin/env python
# coding: utf-8

# In[21]:


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


# In[22]:


train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')


# In[23]:


train_df.head()

#arrange data merge
#from fastai tabular Starter fork kernel
# In[24]:


structures = pd.read_csv('../input/structures.csv')

def map_atom_info(df, atom_idx):
    df = pd.merge(df, structures, how = 'left',
                  left_on  = ['molecule_name', f'atom_index_{atom_idx}'],
                  right_on = ['molecule_name',  'atom_index'])
    
    df = df.drop('atom_index', axis=1)
    df = df.rename(columns={'atom': f'atom_{atom_idx}',
                            'x': f'x_{atom_idx}',
                            'y': f'y_{atom_idx}',
                            'z': f'z_{atom_idx}'})
    return df

train_df = map_atom_info(train_df, 0)
train_df = map_atom_info(train_df, 1)

test_df = map_atom_info(test_df, 0)
test_df = map_atom_info(test_df, 1)


# In[25]:


train_df.tail()


# In[26]:


scalar_df = pd.read_csv('../input/scalar_coupling_contributions.csv')
scalar_df.tail()


# 

# 
# #arranging and merging the scalar coupling coefficient using same method
# 
# def map_scalar_atom_info(df):
#     df = pd.merge(df, scalar_df, how = 'left',
#                   left_on  = ['molecule_name', 'atom_index_0','atom_index_1','type'],
#                   right_on = ['molecule_name',  'atom_index_0','atom_index_1','type'])
#     
# 
#     return df
# 
# train_df = map_scalar_atom_info(train_df)
# 
# test_df = map_scalar_atom_info(test_df)

# 

# In[29]:


test_df.tail()


# In[30]:


train_df.head()


# In[34]:


train_df['atom_1'].unique()


# In[35]:


train_df['atom_0'].unique()


# In[36]:


test_df['atom_0'].unique()


# In[37]:


test_df['atom_1'].unique()


# In[38]:


train_df['type'].unique()


# In[39]:


test_df['type'].unique()


# In[41]:


#replace atom_0 and atom_1 
#C:0,H:1,N:2

train_df=train_df.replace({'atom_0': {'H': 0, 'C': 1,'N':2}})
train_df=train_df.replace({'atom_1': {'H': 0, 'C': 1,'N':2}})

#test set

test_df=test_df.replace({'atom_0': {'H': 0, 'C': 1,'N':2}})
test_df=test_df.replace({'atom_1': {'H': 0, 'C': 1,'N':2}})



#replace type with 
# '2JHC':0, '1JHC':1, '3JHH':2, '3JHC':3, '2JHH':4, '1JHN':5, '3JHN:6', '2JHN':7

train_df=train_df.replace({'type': {'2JHC':0, '1JHC':1, '3JHH':2, '3JHC':3, '2JHH':4, '1JHN':5, '3JHN':6, '2JHN':7}})
test_df=test_df.replace({'type': {'2JHC':0, '1JHC':1, '3JHH':2, '3JHC':3, '2JHH':4, '1JHN':5, '3JHN':6, '2JHN':7}})


# In[48]:


#molecules to be test names are store in separate file
molecules_list = test_df['molecule_name']
molecules_list.head(2)


# In[49]:


train_df = train_df.drop('molecule_name',axis=1)
train_df = train_df.drop('id',axis=1)

test_df =  test_df.drop('molecule_name',axis=1)
test_df = test_df.drop('id',axis=1)

train_df.head()


# In[51]:


test_df.head()


# In[53]:


train_df.shape


# In[54]:


test_df.shape


# In[56]:


#taking y value out from train dataframes
y_df = train_df['scalar_coupling_constant']

#removing scalar_coupling_constant from train dataframes
train_df= train_df.drop('scalar_coupling_constant',axis=1)


# In[58]:


train_df.shape,test_df.shape,y_df.shape


# In[59]:


#convert into numpy array
x_train = np.array(train_df,dtype='float32')
y_train = np.array(y_df,dtype='float32')
x_test = np.array(test_df,dtype='float32')

x_train.shape,y_train.shape,x_test.shape


# In[60]:


#spilit data for validation and cross check only for 20%

from sklearn.model_selection import train_test_split
(x_train,x_validate,y_train,y_validate) = train_test_split(
    x_train,y_train,test_size=0.2,random_state=12345
)


# In[61]:


x_train.shape,x_validate.shape


# In[64]:


#normalise the data

from sklearn import preprocessing

x_train = preprocessing.scale(x_train)
x_validate = preprocessing.scale(x_validate)
x_test = preprocessing.scale(x_test)




# In[65]:


x_train.shape,x_validate.shape,x_test.shape,y_train.shape,y_validate.shape


# In[66]:


#models

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier


# In[70]:


#error occured Unknown label type: 'continous'
#solution found from https://www.kaggle.com/pratsiuk/valueerror-unknown-label-type-continuous
from sklearn import utils

lab_enc = preprocessing.LabelEncoder()
y_train_encoded = lab_enc.fit_transform(y_train)
y_validate_encoded = lab_enc.fit_transform(y_validate)
print(y_train_encoded)
print(utils.multiclass.type_of_target(y_train))
print(utils.multiclass.type_of_target(y_train.astype('int')))
print(utils.multiclass.type_of_target(y_train_encoded))


# In[ ]:


#1 model logistic regression
logreg = LogisticRegression()
logreg.fit(x_train, y_train_encoded)
y_pred_lr = logreg.predict(x_test)
print('Training Accuracy for logistic regression is : { } '.format(logreg.score(x_train, y_train_encoded)))
print('Testing Accuracy for logistic regression is : { } '.format(logreg.score(x_validate, y_validate_encoded)))


# In[ ]:




