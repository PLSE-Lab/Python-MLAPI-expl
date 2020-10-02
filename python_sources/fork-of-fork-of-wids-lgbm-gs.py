#!/usr/bin/env python
# coding: utf-8

# # Forked from the Fork of my own kernel - Double Recurrent?

# ![WiDS.PNG](attachment:WiDS.PNG)

# # Acknowledgements
# 
# kernels: 
# * https://www.kaggle.com/jayjay75/wids2020-lgb-starter-adversarial-validation
# * https://www.kaggle.com/danofer/wids-2020-starter-catboost-0-9045-lb
# * https://www.kaggle.com/mlisovyi/lightgbm-hyperparameter-optimisation-lb-0-761
# 
# Discussion & Comments:
# * https://www.kaggle.com/c/widsdatathon2020/discussion/130532 (Thanks @brunotavares for the suggestions)
# * @arashnic - Thanks for the suggestions on reducing overfitting 
# * Thank you everyone for the support. Code on....

# <a class="anchor" id="0.1"></a>
# ## Table of Contents
# 
# 1. [Import libraries, Memory reduction](#1)
# 1. [Reading & Checking categorical variables](#2)
# 1. [Visualization of categorical variables](#3)
# 1. [EDA - Let's form Teams](#4)
# 1. [Dropping few column/s with single value and all unique values](#5)
# 1. [Checking NAs for initial column clipping](#6)
# 1. [MICE Imputation](#7)
# 1. [Using formula to impute BMI](#8)
# 1. [Extracting columns to change to Categorical](#9)
# 1. [Resampling: Minority to Majority](#10)
# 1. [Splitting and preparing to Model](#11)
# 1. [Model - LGBM](#12)  
# 1. [Grid Search for 'learning_rate' & 'num_iterations'](#13)
# 1. [Grid Search for 'scale_pos_weight'](#14)
# 1. [Grid Search for 'colsample_bytree', 'num_leaves', 'min_child_samples', 'min_child_weight'](#15)
# 1. [Grid Search for 'max_bin', 'max_depth', 'min_data_in_leaf'](#16)
# 1. [Grid Search for 'reg_lambda', 'boosting'](#17)
# 1. [Final Build](#18)

# ## 1. Import libraries, Memory reduction <a class="anchor" id="1"></a>
# [Back to Table of Contents](#0.1)

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

import warnings
warnings.filterwarnings("ignore")

import seaborn as sns
import math
import matplotlib as p
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import scipy.stats as sps
import re
import copy

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import Imputer
# from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[ ]:


def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df


def import_data(file):
    """create a dataframe and optimize its memory usage"""
    df = pd.read_csv(file, parse_dates=True, keep_date_col=True)
    df = reduce_mem_usage(df)
    return df


# ## 2. Reading & Checking categorical variables <a class="anchor" id="2"></a>
# [Back to Table of Contents](#0.1)
# 

# In[ ]:


train = import_data('../input/widsdatathon2020/training_v2.csv')
test = import_data('../input/widsdatathon2020/unlabeled.csv')
st = pd.read_csv('../input/widsdatathon2020/solution_template.csv')
ss = pd.read_csv('../input/widsdatathon2020/samplesubmission.csv')
dictionary = pd.read_csv('../input/widsdatathon2020/WiDS Datathon 2020 Dictionary.csv')


# In[ ]:


# Checking shapes
print('train    ', train.shape)
print('test     ', test.shape)

# Combining train and test to explore the categorical attributes
train_len = len(train)
combined_dataset = pd.concat(objs = [train, test], axis = 0)
print('combined', combined_dataset.shape)


# In[ ]:


pd.set_option('display.max_rows', 200)
# Dictionary
dictionary.style.set_properties(subset=['Description'], **{'width': '500px'})


# In[ ]:


# Extracing categorical columns
df_cat = combined_dataset.select_dtypes(include=['object', 'category'])
df_cat.columns


# In[ ]:


# Checking unique values for each categorical columns
col = df_cat.columns
for i in col:
    n = list(df_cat[i].unique())
    print("Unique values for ", i)
    print(n)


# ## 3. Visualization of categorical variables <a class="anchor" id="3"></a>
# #### ['hospital_admit_source', 'icu_admit_source', 'icu_stay_type', 'icu_type', 'apache_3j_bodysystem', 'apache_2_bodysystem']
# [Back to Table of Contents](#0.1)

# In[ ]:


sns.catplot('hospital_admit_source', data= train, kind='count', alpha=0.7, height=6, aspect= 3.5)

# Get current axis on current figure
ax = plt.gca()

# Max value to be set
y_max = train['hospital_admit_source'].value_counts().max() 

# Iterate through the list of axes' patches
for p in ax.patches:
    ax.text(p.get_x() + p.get_width()/5., p.get_height(),'%d' % int(p.get_height()),
            fontsize=13, color='blue', ha='center', va='bottom')
plt.title('Frequency plot of Hospital Admit Source for train', fontsize = 20, color = 'black')
plt.show()


# In[ ]:


sns.catplot('hospital_admit_source', data= test, kind='count', alpha=0.7, height=6, aspect= 3.5)

# Get current axis on current figure
ax = plt.gca()

# Max value to be set
y_max = test['hospital_admit_source'].value_counts().max() 

# Iterate through the list of axes' patches
for p in ax.patches:
    ax.text(p.get_x() + p.get_width()/5., p.get_height(),'%d' % int(p.get_height()),
            fontsize=13, color='blue', ha='center', va='bottom')
plt.title('Frequency plot of Hospital Admit Source for test', fontsize = 20, color = 'black')
plt.show()


# In[ ]:


sns.catplot('icu_admit_source', data= train, kind='count', alpha=0.7, height=4, aspect= 6)

# Get current axis on current figure
ax = plt.gca()

# Max value to be set
y_max = train['icu_admit_source'].value_counts().max() 

# Iterate through the list of axes' patches
for p in ax.patches:
    ax.text(p.get_x() + p.get_width()/5., p.get_height(),'%d' % int(p.get_height()),
            fontsize=13, color='blue', ha='center', va='bottom')
plt.title('Frequency plot of ICU Admit Source for train', fontsize = 20, color = 'black')
plt.show()


# In[ ]:


sns.catplot('icu_admit_source', data= test, kind='count', alpha=0.7, height=4, aspect= 6)

# Get current axis on current figure
ax = plt.gca()

# Max value to be set
y_max = test['icu_admit_source'].value_counts().max() 

# Iterate through the list of axes' patches
for p in ax.patches:
    ax.text(p.get_x() + p.get_width()/5., p.get_height(),'%d' % int(p.get_height()),
            fontsize=13, color='blue', ha='center', va='bottom')
plt.title('Frequency plot of ICU Admit Source for test', fontsize = 20, color = 'black')
plt.show()


# In[ ]:


sns.catplot('icu_stay_type', data= train, kind='count', alpha=0.7, height=6, aspect= 3.5)

# Get current axis on current figure
ax = plt.gca()

# Max value to be set
y_max = train['icu_stay_type'].value_counts().max() 

# Iterate through the list of axes' patches
for p in ax.patches:
    ax.text(p.get_x() + p.get_width()/5., p.get_height(),'%d' % int(p.get_height()),
            fontsize=13, color='blue', ha='center', va='bottom')
plt.title('Frequency plot of ICU Stay Type for train', fontsize = 20, color = 'black')
plt.show()


# In[ ]:


sns.catplot('icu_stay_type', data= test, kind='count', alpha=0.7, height=4, aspect= 6)

# Get current axis on current figure
ax = plt.gca()

# Max value to be set
y_max = test['icu_stay_type'].value_counts().max() 

# Iterate through the list of axes' patches
for p in ax.patches:
    ax.text(p.get_x() + p.get_width()/5., p.get_height(),'%d' % int(p.get_height()),
            fontsize=13, color='blue', ha='center', va='bottom')
plt.title('Frequency plot of ICU Stay Type for test', fontsize = 20, color = 'black')
plt.show()


# In[ ]:


sns.catplot('icu_type', data= train, kind='count', alpha=0.7, height=4, aspect= 6)

# Get current axis on current figure
ax = plt.gca()

# Max value to be set
y_max = train['icu_type'].value_counts().max() 

# Iterate through the list of axes' patches
for p in ax.patches:
    ax.text(p.get_x() + p.get_width()/5., p.get_height(),'%d' % int(p.get_height()),
            fontsize=13, color='blue', ha='center', va='bottom')
plt.title('Frequency plot of ICU Type for train', fontsize = 20, color = 'black')
plt.show()


# In[ ]:


sns.catplot('icu_type', data= test, kind='count', alpha=0.7, height=4, aspect= 6)

# Get current axis on current figure
ax = plt.gca()

# Max value to be set
y_max = test['icu_type'].value_counts().max() 

# Iterate through the list of axes' patches
for p in ax.patches:
    ax.text(p.get_x() + p.get_width()/5., p.get_height(),'%d' % int(p.get_height()),
            fontsize=13, color='blue', ha='center', va='bottom')
plt.title('Frequency plot of ICU Type for test', fontsize = 20, color = 'black')
plt.show()


# In[ ]:


sns.catplot('apache_3j_bodysystem', data= train, kind='count', alpha=0.7, height=4, aspect= 5)

# Get current axis on current figure
ax = plt.gca()

# Max value to be set
y_max = train['icu_type'].value_counts().max() 

# Iterate through the list of axes' patches
for p in ax.patches:
    ax.text(p.get_x() + p.get_width()/5., p.get_height(),'%d' % int(p.get_height()),
            fontsize=13, color='blue', ha='center', va='bottom')
plt.title('Frequency plot of apache_3j_bodysystem for train', fontsize = 20, color = 'black')
plt.show()


# In[ ]:


sns.catplot('apache_3j_bodysystem', data= test, kind='count', alpha=0.7, height=4, aspect= 5)

# Get current axis on current figure
ax = plt.gca()

# Max value to be set
y_max = test['icu_type'].value_counts().max() 

# Iterate through the list of axes' patches
for p in ax.patches:
    ax.text(p.get_x() + p.get_width()/5., p.get_height(),'%d' % int(p.get_height()),
            fontsize=13, color='blue', ha='center', va='bottom')
plt.title('Frequency plot of apache_3j_bodysystem for test', fontsize = 20, color = 'black')
plt.show()


# In[ ]:


sns.catplot('apache_2_bodysystem', data= train, kind='count', alpha=0.7, height=4, aspect= 5)

# Get current axis on current figure
ax = plt.gca()

# Max value to be set
y_max = train['icu_type'].value_counts().max() 

# Iterate through the list of axes' patches
for p in ax.patches:
    ax.text(p.get_x() + p.get_width()/5., p.get_height(),'%d' % int(p.get_height()),
            fontsize=13, color='blue', ha='center', va='bottom')
plt.title('Frequency plot of apache_2_bodysystem for train', fontsize = 20, color = 'black')
plt.show()


# In[ ]:


sns.catplot('apache_2_bodysystem', data= test, kind='count', alpha=0.7, height=4, aspect= 5)

# Get current axis on current figure
ax = plt.gca()

# Max value to be set
y_max = test['icu_type'].value_counts().max() 

# Iterate through the list of axes' patches
for p in ax.patches:
    ax.text(p.get_x() + p.get_width()/5., p.get_height(),'%d' % int(p.get_height()),
            fontsize=13, color='blue', ha='center', va='bottom')
plt.title('Frequency plot of apache_2_bodysystem for test', fontsize = 20, color = 'black')
plt.show()


# ## 4. Let's form Teams [EDA] <a class="anchor" id="4"></a>
# * **'hospital_admit_source':** 
#  *     Grouping: ['Other ICU', 'ICU']; ['ICU to SDU', 'Step-Down Unit (SDU)']; ['Other Hospital', 'Other']; ['Recovery Room','Observatoin']
#  *     Renaming: Acute Care/Floor to Acute Care
# * **'icu_type':** 
#  *     Grouping of the following can be explored: ['CCU-CTICU', 'CTICU', 'Cardiac ICU']
# * **'apache_2_bodysystem':** 
#  *     Grouping of the following can be explored: ['Undefined Diagnoses', 'Undefined diagnoses']
#  
#  
# 
# [Back to Table of Contents](#0.1)

# In[ ]:


combined_dataset['hospital_admit_source'] = combined_dataset['hospital_admit_source'].replace({'Other ICU': 'ICU','ICU to SDU':'SDU', 'Step-Down Unit (SDU)': 'SDU',
                                                                                               'Other Hospital':'Other','Observation': 'Recovery Room','Acute Care/Floor': 'Acute Care'})

# combined_dataset['icu_type'] = combined_dataset['icu_type'].replace({'CCU-CTICU': 'Grpd_CICU', 'CTICU':'Grpd_CICU', 'Cardiac ICU':'Grpd_CICU'}) # Can be explored

combined_dataset['apache_2_bodysystem'] = combined_dataset['apache_2_bodysystem'].replace({'Undefined diagnoses': 'Undefined Diagnoses'})


# In[ ]:


sns.catplot('icu_type', data= combined_dataset, kind='count', alpha=0.7, height=4, aspect= 6)

# Get current axis on current figure
ax = plt.gca()

# Max value to be set
y_max = combined_dataset['icu_type'].value_counts().max() 

# Iterate through the list of axes' patches
for p in ax.patches:
    ax.text(p.get_x() + p.get_width()/5., p.get_height(),'%d' % int(p.get_height()),
            fontsize=13, color='blue', ha='center', va='bottom')
plt.title('Frequency plot of ICU Type for combined_dataset', fontsize = 20, color = 'black')
plt.show()


# ## 5. Dropping few column/s with single value and all unique values <a class="anchor" id="5"></a>
# [Back to Table of Contents](#0.1)

# In[ ]:


for i in combined_dataset.columns:
    if combined_dataset[i].nunique() == 1:
        print('With only 1 unique value: ', i)
    if combined_dataset[i].nunique() == combined_dataset.shape[0]:
        print('With all unique value: ', i)


# In[ ]:


# Dropping 'readmission_status', 'patient_id', along with 'gender'
combined_dataset = combined_dataset.drop(['readmission_status', 'patient_id', 'gender'], axis=1)


# In[ ]:


import copy
train = copy.copy(combined_dataset[:train_len])
test = copy.copy(combined_dataset[train_len:])
print('combined dataset ', combined_dataset.shape)
print('train             ', train.shape)
print('test              ', test.shape)


# ## 6. Checking NAs for initial column clipping <a class="anchor" id="6"></a>
# [Back to Table of Contents](#0.1)

# In[ ]:


# On train data
pd.set_option('display.max_rows', 500)
NA_col_train = pd.DataFrame(train.isna().sum(), columns = ['NA_Count'])
NA_col_train['% of NA'] = (NA_col_train.NA_Count/len(train))*100
NA_col_train.sort_values(by = ['% of NA'], ascending = False, na_position = 'first')


# In[ ]:


# On test data
pd.set_option('display.max_rows', 500)
NA_col_test = pd.DataFrame(test.isna().sum(), columns = ['NA_Count'])
NA_col_test['% of NA'] = (NA_col_test.NA_Count/len(test))*100
NA_col_test.sort_values(by = ['% of NA'], ascending = False, na_position = 'first')


# In[ ]:


# Setting threshold of 80%
NA_col_train = NA_col_train[NA_col_train['% of NA'] >= 80]
cols_to_drop = NA_col_train.index.tolist()
# cols_to_drop.remove('hospital_death')
cols_to_drop


# In[ ]:


# Dropping columns with >= 80% of NAs
combined_dataset = combined_dataset.drop(cols_to_drop, axis=1)


# In[ ]:


train = copy.copy(combined_dataset[:train_len])
test = copy.copy(combined_dataset[train_len:])
print('combined dataset ', combined_dataset.shape)
print('train             ', train.shape)
print('test              ', test.shape)


# ## 7. MICE Imputation <a class="anchor" id="7"></a>
# #### ['hospital_admit_source', 'icu_admit_source', 'icu_stay_type', 'icu_type', 'apache_3j_bodysystem', 'apache_2_bodysystem']
# [Back to Table of Contents](#0.1)

# In[ ]:


# Suggestion Courtesy: Bruno Taveres - https://www.kaggle.com/c/widsdatathon2020/discussion/130532
# Adding 2 apache columns as well

# Import IterativeImputer from fancyimpute
from fancyimpute import IterativeImputer

# Initialize IterativeImputer
mice_imputer = IterativeImputer()

# Impute using fit_tranform on diabetes
train[['age', 'height', 'weight', 'apache_4a_hospital_death_prob', 'apache_4a_icu_death_prob']] = mice_imputer.fit_transform(train[['age', 'height', 'weight', 'apache_4a_hospital_death_prob', 'apache_4a_icu_death_prob']])
test[['age', 'height', 'weight', 'apache_4a_hospital_death_prob', 'apache_4a_icu_death_prob']] = mice_imputer.fit_transform(test[['age', 'height', 'weight', 'apache_4a_hospital_death_prob', 'apache_4a_icu_death_prob']])


# In[ ]:


print('Train check')
print(train[['age', 'height', 'weight', 'apache_4a_hospital_death_prob', 'apache_4a_icu_death_prob']].isna().sum())
print('Test check')
print(test[['age', 'height', 'weight', 'apache_4a_hospital_death_prob', 'apache_4a_icu_death_prob']].isna().sum())


# ## 8. Using formula to impute BMI <a class="anchor" id="8"></a>
# #### [Formula: BMI = Weight(kg)/(Height(m)* Height(m))]
# [Back to Table of Contents](#0.1)

# In[ ]:


train['new_bmi'] = (train['weight']*10000)/(train['height']*train['height'])
train[['bmi', 'new_bmi', 'weight', 'height']].head(10)


# In[ ]:


train['bmi'] = train['bmi'].fillna(train['new_bmi'])
train[['bmi', 'new_bmi', 'weight', 'height']].head(10)


# In[ ]:


train = train.drop(['new_bmi'], axis = 1)


# In[ ]:


test['new_bmi'] = (test['weight']*10000)/(test['height']*test['height'])
test[['bmi', 'new_bmi', 'weight', 'height']].head(10)


# In[ ]:


test['bmi'] = test['bmi'].fillna(test['new_bmi'])
test[['bmi', 'new_bmi', 'weight', 'height']].head(10)


# In[ ]:


test = test.drop(['new_bmi'], axis = 1)


# ## 9. Extracting columns to change to Categorical <a class="anchor" id="9"></a>
# [Back to Table of Contents](#0.1)

# In[ ]:


print('For Train')
d1 = train.nunique()
print(sorted(d1))
print("==============================")
print('For Test')
d2 = test.nunique()
print(sorted(d2))


# In[ ]:


col_train = train.columns
col_test = test.columns


# In[ ]:


l1 = []
for i in col_train:
    if train[i].nunique() <= 11:
        l1.append(i)
               
l1.remove('hospital_death')


# In[ ]:


l2 = []
for i in col_test:
    if test[i].nunique() <= 11:
        l2.append(i)
        
l2.remove('hospital_death')


# In[ ]:


# Checking the columns in train and test are same or not
df = pd.DataFrame(l1, columns = ['train'])
df['test'] = pd.DataFrame(l2)
df


# In[ ]:


train[l1] = train[l1].apply(lambda x: x.astype('category'), axis=0)
test[l2] = test[l2].apply(lambda x: x.astype('category'), axis=0)
print('train dtypes:')
print(train[l1].dtypes)
print('======================================')
print('test dtypes:')
print(test[l1].dtypes)


# In[ ]:


cols = train.columns
num_cols = train._get_numeric_data().columns
cat_cols = list(set(cols) - set(num_cols))
cat_cols


# In[ ]:


from sklearn.preprocessing import LabelEncoder
for usecol in cat_cols:
    train[usecol] = train[usecol].astype('str')
    test[usecol] = test[usecol].astype('str')
    
    #Fit LabelEncoder
    le = LabelEncoder().fit(
            np.unique(train[usecol].unique().tolist()+ test[usecol].unique().tolist()))

    #At the end 0 will be used for dropped values
    train[usecol] = le.transform(train[usecol])+1
    test[usecol]  = le.transform(test[usecol])+1
    
    train[usecol] = train[usecol].replace(np.nan, '').astype('int').astype('category')
    test[usecol]  = test[usecol].replace(np.nan, '').astype('int').astype('category')


# ## 10. Resampling: Minority to Majority <a class="anchor" id="10"></a>
# [Back to Table of Contents](#0.1)

# In[ ]:


train.hospital_death.value_counts()


# In[ ]:


# Separate majority and minority classes
df_majority = train[train.hospital_death==0]
df_minority = train[train.hospital_death==1]


# In[ ]:


# Resampling the minority levels to match the majority level
# Upsample minority class
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
df_minority_upsampled = resample(df_minority, 
                                 replace=True,     # sample with replacement
                                 n_samples=83798,    # to match majority class
                                 random_state= 303) # reproducible results
 
# Combine majority class with upsampled minority class
df_upsampled = pd.concat([df_majority, df_minority_upsampled])
 
# Display new class counts
df_upsampled.hospital_death.value_counts()


# ## 11. Splitting and preparing to Model <a class="anchor" id="11"></a>
# [Back to Table of Contents](#0.1)

# In[ ]:


from sklearn.model_selection import train_test_split
Train, Validation = train_test_split(df_upsampled, test_size = 0.3, random_state = 333)


# In[ ]:


X_train = Train.copy().drop('hospital_death', axis = 1)
y_train = Train[['encounter_id','hospital_death']]
X_val = Validation.copy().drop('hospital_death', axis = 1)
y_val = Validation[['encounter_id','hospital_death']]
X_test = test.copy().drop('hospital_death', axis = 1)
y_test = test[['encounter_id', 'hospital_death']]


# In[ ]:


X_train.set_index('encounter_id', inplace = True)
y_train.set_index('encounter_id', inplace = True)
X_val.set_index('encounter_id', inplace = True)
y_val.set_index('encounter_id', inplace = True)
X_test.set_index('encounter_id', inplace = True)
y_test.set_index('encounter_id', inplace = True)


# In[ ]:


sns.catplot('hospital_death', data= train, kind='count', alpha=0.7, height=4, aspect= 3)

# Get current axis on current figure
ax = plt.gca()

# Max value to be set
y_max = train['hospital_death'].value_counts().max() 

# Iterate through the list of axes' patches
for p in ax.patches:
    ax.text(p.get_x() + p.get_width()/5., p.get_height(),'%d' % int(p.get_height()),
            fontsize=13, color='blue', ha='center', va='bottom')
plt.title('Frequency plot of hospital_death', fontsize = 20, color = 'black')
plt.show()


# ## 12. Model - LGBM <a class="anchor" id="12"></a>
# #### Grid Search for 'cat_smooth', 'min_data_per_group', and 'max_cat_threshold'
# [Back to Table of Contents](#0.1)

# In[ ]:


import lightgbm as lgb
from sklearn.model_selection import RandomizedSearchCV
clf = lgb.LGBMClassifier(silent=True, random_state = 333, metric='roc_auc', n_jobs=4)


# In[ ]:


from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform
params ={'cat_smooth' : sp_randint(1, 100), 'min_data_per_group': sp_randint(1,1000), 'max_cat_threshold': sp_randint(1,100)}


# In[ ]:


fit_params={"early_stopping_rounds":2, 
            "eval_metric" : 'auc', 
            "eval_set" : [(X_train, y_train),(X_val,y_val)],
            'eval_names': ['train','valid'],
            'verbose': 300,
            'categorical_feature': 'auto'}


# In[ ]:


gs = RandomizedSearchCV( estimator=clf, param_distributions=params, scoring='roc_auc',cv=3, refit=True,random_state=333,verbose=True)


# In[ ]:


gs.fit(X_train, y_train, **fit_params)
print('Best score reached: {} with params: {} '.format(gs.best_score_, gs.best_params_))


# ## 13. Grid Search for 'learning_rate' & 'num_iterations' <a class="anchor" id="13"></a>
# [Back to Table of Contents](#0.1)

# In[ ]:


gs.best_params_, gs.best_score_


# In[ ]:


clf2 = lgb.LGBMClassifier(random_state=304, metric = 'roc_auc', cat_smooth = 38, max_cat_threshold = 73, min_data_per_group = 73, n_jobs=4)


# In[ ]:


params_2 = {'learning_rate': [0.08, 0.85, 0.09],   
            'num_iterations': sp_randint(1000,3000)}


# In[ ]:


gs2 = RandomizedSearchCV(estimator=clf2, param_distributions=params_2, scoring='roc_auc',cv=3,refit=True,random_state=333,verbose=True)


# In[ ]:


gs2.fit(X_train, y_train, **fit_params)
print('Best score reached: {} with params: {} '.format(gs2.best_score_, gs2.best_params_))


# ## 14. Grid Search for 'scale_pos_weight' <a class="anchor" id="14"></a>
# [Back to Table of Contents](#0.1)

# In[ ]:


gs2.best_params_, gs2.best_score_


# In[ ]:


clf3 = lgb.LGBMClassifier(**clf2.get_params())
clf3.set_params(**gs2.best_params_)


# In[ ]:


params_3 = {'scale_pos_weight': sp_randint(1,15)}


# In[ ]:


gs3 = RandomizedSearchCV(estimator=clf3, param_distributions=params_3, scoring='roc_auc',cv=3,refit=True,random_state=333,verbose=True)


# In[ ]:


gs3.fit(X_train, y_train, **fit_params)
print('Best score reached: {} with params: {} '.format(gs3.best_score_, gs3.best_params_))


# ## 15. Grid Search for 'colsample_bytree', 'num_leaves', 'min_child_samples', 'min_child_weight' <a class="anchor" id="15"></a>
# [Back to Table of Contents](#0.1)

# In[ ]:


gs3.best_params_, gs3.best_score_


# In[ ]:


clf4 = lgb.LGBMClassifier(**clf3.get_params())
clf4.set_params(**gs3.best_params_)


# In[ ]:


params_4 = {'colsample_bytree': sp_uniform(loc=0.4, scale=0.6), 'num_leaves': sp_randint(500, 5000), 
            'min_child_samples': sp_randint(100,500), 'min_child_weight': [1e-2, 1e-1, 1, 1e1]}


# In[ ]:


gs4 = RandomizedSearchCV(estimator=clf4, param_distributions=params_4, scoring='roc_auc',cv=2,refit=True,random_state=333,verbose=True)


# In[ ]:


gs4.fit(X_train, y_train, **fit_params)
print('Best score reached: {} with params: {} '.format(gs4.best_score_, gs4.best_params_))


# ## 16. Grid Search for 'max_bin', 'max_depth', 'min_data_in_leaf' <a class="anchor" id="16"></a>
# [Back to Table of Contents](#0.1)

# In[ ]:


gs4.best_params_, gs4.best_score_


# In[ ]:


clf5 = lgb.LGBMClassifier(**clf4.get_params())
clf5.set_params(**gs4.best_params_)


# In[ ]:


params_5 = {'max_bin': sp_randint(100, 1500), 'max_depth': sp_randint(1, 15), 
            'min_data_in_leaf': sp_randint(500,3500)}


# In[ ]:


gs5 = RandomizedSearchCV(estimator=clf5, param_distributions=params_5, scoring='roc_auc',cv=2,refit=True,random_state=333,verbose=True)


# In[ ]:


gs5.fit(X_train, y_train, **fit_params)
print('Best score reached: {} with params: {} '.format(gs5.best_score_, gs5.best_params_))


# ## 17. Grid Search for 'reg_lambda', 'boosting' <a class="anchor" id="17"></a>
# [Back to Table of Contents](#0.1)

# In[ ]:


gs5.best_params_, gs5.best_score_


# In[ ]:


clf6 = lgb.LGBMClassifier(**clf5.get_params())
clf6.set_params(**gs5.best_params_)


# In[ ]:


params_6 = {'reg_lambda': sp_randint(1, 30), 'boosting': ['goss', 'dart']}


# In[ ]:


gs6 = RandomizedSearchCV(estimator=clf6, param_distributions=params_6, scoring='roc_auc',cv=2,refit=True,random_state=333,verbose=True)


# In[ ]:


gs6.fit(X_train, y_train, **fit_params)
print('Best score reached: {} with params: {} '.format(gs6.best_score_, gs6.best_params_))


# ## 18. Final Build <a class="anchor" id="18"></a>
# [Back to Table of Contents](#0.1)

# In[ ]:


gs6.best_params_


# In[ ]:


final_params = {**gs.best_params_, **gs2.best_params_, **gs3.best_params_, **gs4.best_params_, **gs5.best_params_, **gs6.best_params_,
               'bagging_fraction': 0.6, 'feature_fraction': 0.8, 'scoring':'roc_auc', 'metric':'auc', 'objective': 'binary'}
final_params


# In[ ]:


lgbm_train = lgb.Dataset(X_train, y_train, categorical_feature=cat_cols)
lgbm_val = lgb.Dataset(X_val, y_val, reference = lgbm_train)


# In[ ]:


evals_result = {}  # to record eval results for plotting
model_lgbm = lgb.train(final_params,
                lgbm_train,
                num_boost_round=250,
                valid_sets=[lgbm_train, lgbm_val],
                feature_name=['f' + str(i + 1) for i in range(X_train.shape[-1])],
                categorical_feature= [150],
                evals_result=evals_result,
                verbose_eval=100)


# In[ ]:


ax = lgb.plot_metric(evals_result, metric='auc', figsize=(15, 8))
plt.show()


# In[ ]:


test["hospital_death"] = model_lgbm.predict(X_test, pred_contrib=False)


# In[ ]:


test[["encounter_id","hospital_death"]].to_csv("submission_lgbm.csv",index=False)
test[["encounter_id","hospital_death"]].head()


# In[ ]:


test[["encounter_id","hospital_death"]].describe()

