#!/usr/bin/env python
# coding: utf-8

# Lately I received so stressted learning deep learning, NLP and Image Classificaiton. This kernel relieve my strees!  
# Restaurant Cleanliness is the most important part for customer. Through NYC Restuarnt EDA, we can see the inside of NYC's cleanliness. I hope you enjoy this kernel.

# **0. Setting A Goal  **  
#  What we are intereseted in is where is the cleaniness restuarant. This data-set has three groups variables, Location, Menu, and Score-Measurment variables. I will deep diving into the clean restaraunt in New York.
#  
# **1. Data Auditing, Null Inspection**  
#    Where is the missing check place by NYC DOHMH(Department of Health and Mental Hygiene)?  
# **2. Variable Transformation, Making a New one and Grouping**  
#  There needs some basic transformation to variables, such as datetime and level-code variables.  
# **3. Score Variable Relationship of Restuarant**  
#  The 4 Core Summary for the important variables.  
#  **4. Score Variable & Location Variable**  (I will do it more)

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

df_train = pd.read_csv('../input/DOHMH_New_York_City_Restaurant_Inspection_Results.csv')
df_train.head()


# ### 1. Data Auditing
# - Null Check

# In[2]:


print('Null Variables: ', df_train.columns[df_train.isnull().any()].tolist())


# In[3]:


null_var = ['DBA', 'BUILDING', 'STREET', 'ZIPCODE', 'PHONE', 'ACTION',
       'VIOLATION CODE', 'VIOLATION DESCRIPTION', 'SCORE', 'GRADE',
       'GRADE DATE', 'INSPECTION TYPE']
null_cnt = df_train[null_var].isnull().sum()
null_cnt = null_cnt.to_frame()
null_cnt.columns = ['null_sum']
null_cnt['not_null'] = df_train.shape[0] - null_cnt['null_sum']
null_cnt.plot.bar(stacked = True)


# Violation and SCORE and GRADE are good criterion to evaluate the restuarant. However, there is that a half of data null data is in GRADE. The other variables were deemed to be not having null. But before dropping out the null rows of the viloation and score, I skimmed their local information given null violation and score.
# 

# In[4]:


null_test = df_train.columns[df_train.columns.str.contains('LATION')].tolist()
null_test2 = ['SCORE']
df_train[null_test+null_test2].isnull().sum()


# In[5]:


local_lst = ['BORO','ZIPCODE']
f = plt.figure(figsize = (12,4))

tmp_null = df_train[null_test].isnull()
tmp_null = (tmp_null.iloc[:,0] | tmp_null.iloc[:,1])

ax = f.add_subplot(1,2,1)
local = local_lst[0]
tmp = df_train.loc[tmp_null, local].value_counts().to_frame()
tmp2 = df_train.loc[:,local].value_counts().to_frame()
null_ratio = (tmp/tmp2.loc[tmp.index,:])
ax.scatter(x = np.arange(null_ratio.shape[0]) ,y = null_ratio.iloc[:,0], alpha = 0.5)
ax.set_title(local)
ax.set_ylabel('Null Probability')
ax.set_xticklabels(['wow'] + tmp.index.tolist())
ax.tick_params('x', rotation = 70)

ax = f.add_subplot(1,2,2)
local = local_lst[1]
tmp = df_train.loc[tmp_null, local].value_counts().to_frame()
tmp2 = df_train.loc[:,local].value_counts().to_frame()
null_ratio = (tmp/tmp2.loc[tmp.index,:])
ax.scatter(x = np.arange(null_ratio.shape[0]) ,y = null_ratio.iloc[:,0], alpha = 0.5)
ax.set_title(local)
ax.set_ylabel('Null Probability')
ax.tick_params('x', rotation = 70)

plt.show()


# In[6]:


df_train.groupby('BORO')['BORO'].size()['Missing']


# The inspection center missed so much the restuarnt living in Missing Area and ZIPCODE 175~200. I drop out of the missing Boro, only 9 points and the place where the null probability is undr 0.2 on ZIPCODE.

# In[7]:


df_train = df_train.loc[df_train.ZIPCODE.isin(null_ratio[null_ratio.sort_values('ZIPCODE') < 0.2].index),:]
df_train = df_train.loc[(df_train.BORO != 'Missing'),:]


# In[8]:


df_train.shape


# ### 2. Variable Interpretation

# #### Grouping Variables: Var(Rounded Unique Number)
# - Local Variables 7: BORO(6) -> zipcode(200) -> street(3300) -> bulding(7200) -> phone(25000), DBA(20000), CAMIS(26000)
# - Menu Variables 1: CUISINE DESCRIPTION  (84)
# - Score Variables 8: CRTICIAL FLAG(3), ACTION(5), GRADE(6), INSPECTION TYPE(34), VIOLATIO lvl(15) & type(18), DESCRIPTION(93), SCORE(120)
# - DATE 3: GRADE DATE  (1325), INSPETION DATE
# 
# #### Transformation
# - Drop out of RECORD DATE since there is just one date
# - VIOLATION CODE == 'lvl' + 'type' where lvl two decimal numbers, type 'a-z' so that divide it into VIO_lvl and VIO_type
# - VIOLATION DESCRIPTION == ?

# In[9]:


def level_code(row):
    if type(row) == float: return 99, 'No'
    return int(row[:2]), row[2]
df_train['VIO_lvl'],df_train['VIO_type'] = zip(*df_train['VIOLATION CODE'].apply(lambda row: level_code(row)))
#df_train['RECORD_DATE'] = pd.to_datetime(df_train['RECORD DATE'], format = '%m/%d/%Y', errors='coerce')
df_train['INSPECTION_DATE'] = pd.to_datetime(df_train['INSPECTION DATE'], format = '%m/%d/%Y', errors='coerce')
df_train['GRADE_DATE'] = pd.to_datetime(df_train['GRADE DATE'], format = '%m/%d/%Y', errors='coerce')
df_train.drop(['RECORD DATE', 'INSPECTION DATE', 'GRADE DATE'], axis = 1, inplace = True)

df_train.columns = ['_'.join(x.lower().split()) for x in df_train.columns]


# ### 3. Score Variables Relationship
# 
# #### Summary:
# 1. Only A grade restaruant didn't cacth up as a violated one.
# 2. P grade restaruant was already inspected as a violated one, then re-open now. So that the grade was lower on Critical ratio then the others.
# 3. Violation Lvl < 7, said they are the critical restaraunt.
# 4. Violation type, the character: A - J represented both positive and negative. K-T on vio_type, they represent critical or not critical. 

# In[10]:


tmp_tab = pd.crosstab(df_train['action'], df_train['grade'])
tmp_tab[['A', 'B', 'C', 'P', 'Z', 'Not Yet Graded']]


# ** DOHMH: Department of Health and Mental Hygiene - NYC.gov**  
# Please focus on the third to fifth rows. 
# - 3rd row: Where re-opened restuarants often received the bad grade. Especially P is the most part
# - 4th row: No violation! Awesome and Quality Restuarant is almost grade A
# - 5th row: Violation exists! But there is no P - restuarant. Can I think of P as 'paying attention restaruant since security?'

# In[11]:


f = plt.figure(figsize = (12,4))
ax = f.add_subplot(1,2,1)
tmp_tab = pd.crosstab(df_train['critical_flag'], df_train['grade'])
tmp_crit = tmp_tab[['A', 'B', 'C', 'P', 'Z', 'Not Yet Graded']].T
tmp_crit.plot.bar(stacked = True, ax = ax)
ax.set_title('Stacked Critical Flag')
ax.set_xlabel('Grade')

sum_ = tmp_crit.sum(axis = 1)
for col in tmp_crit.columns:
    tmp_crit[col] = tmp_crit[col].divide(sum_)
ax = f.add_subplot(1,2,2)
tmp_crit.plot.bar(stacked = True, ax = ax)
ax.set_title('Stacked Ratio Critical Flag')
ax.set_xlabel('Grade')

plt.show()


# As alluded above Restaraunt_P is 're-open'. So there is no dobut that the the ratio of 'not critical' is higher. In contrasts, B and C grade has high critical ratio. The maginitude of grade A is abosoutely bigger than the other values. And before diving into another variables, I would like to define the effect of the critical range toward Violation level.

# In[12]:


f = plt.figure(figsize = (12,8))
ax = plt.subplot2grid((2,4), (0,0), colspan = 3)
sns.distplot(df_train.vio_lvl.loc[df_train.critical_flag == 'Critical'], ax = ax, kde = False, color = 'r', label = 'Critical')
sns.distplot(df_train.vio_lvl.loc[df_train.critical_flag == 'Not Critical'], ax = ax, kde = False, color = 'b', label = 'Not Critical')
ax.set_title('Violation lvl of Critical Lvl')
ax.legend()
ax = plt.subplot2grid((2,4), (0,3))
sns.distplot(df_train.vio_lvl.loc[df_train.critical_flag == 'Not Applicable'], ax = ax, kde = False, color = 'g', label = 'Not Applicable')
ax.set_title('Violation lvl of Not Applicable')
ax.legend()

ax = plt.subplot2grid((2,4), (1,0), colspan = 4)
tmp_type = df_train[['critical_flag', 'vio_type']].groupby(['critical_flag', 'vio_type']).size().to_frame().reset_index()
tmp_type = tmp_type.pivot('critical_flag', 'vio_type').fillna(0)
tmp_type.columns = tmp_type.columns.droplevel(0)
tmp_type = tmp_type.T
sum_type = tmp_type.sum(axis = 1)
for col in tmp_type.columns:
    tmp_type[col] = tmp_type[col].divide(sum_type)
tmp_type.sort_values('Not Critical').plot.bar(stacked = True, ax = ax)
ax.set_title('Vio Type of Critical Flag')

plt.subplots_adjust(wspace = 0.3, hspace = 0.4)
plt.show()


# Violaiton lvl and type definitley affects to critical flag. 
# - Violation Lvl < 7 means including of 'Critical.'
# - Violation Type, (O, M, N, G, K..) looks so severe symptoms as Critical.
# - Not applicable is almost shown as vio_lvl 99, where is the mssing value, and vio_type No.

# In[13]:


print('OMN')
print(df_train.loc[df_train.vio_type.isin(('O', 'M', 'N')), 'vio_lvl'].value_counts())
print('KL')
print(df_train.loc[df_train.vio_type.isin(('K', 'L')), 'vio_lvl'].value_counts())


# - Wow! 'OMN' is on only one lvl 4 and 'KL' on (4,15)! Can we wrap the type of violation by Alpabetic order?

# In[14]:


tmp_cnt =df_train.groupby(['vio_type']).agg({'vio_lvl': pd.Series.nunique}).sort_values('vio_lvl').T
tmp_cnt[sorted(tmp_cnt.columns, reverse = False)]


# Umm, the early character tends to have various of vio_lvl! And Rerpesent (S,T): the positive part and (G,K,L,M,N,O): the negative one. The othere was mixed.

# ### 4. Score Variable & Location Variable

# In[ ]:




