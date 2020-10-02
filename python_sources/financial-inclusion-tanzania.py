#!/usr/bin/env python
# coding: utf-8

# # Import Libraries

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


# In[ ]:


import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from datetime import datetime
import ipywidgets as widgets
from ipywidgets import interact, interact_manual
# from ipywidgets import widgets
from IPython.display import display, clear_output, Image
# from plotly.widgets import GraphWidget
#%matplotlib inline
import seaborn as sns
import warnings
sns.set()

from scipy.stats import probplot # for a qqplot


# N/b: don't conflict 2 widget to avoid issues with showing

# In[ ]:


# %reset -f 


# # Read Data

# In[ ]:


train=pd.read_csv('/kaggle/input/Train_v2.csv')
test=pd.read_csv('/kaggle/input/Test_v2.csv')
# submission_file=pd.read_csv('/kaggle/input/SubmissionFile.csv')
# Variables=pd.read_csv('VariableDefinitions.csv')


# ## Combine train and test

# In[ ]:


all_data = pd.concat([train, test])


# In[ ]:


all_data.shape, test.shape, train.shape


# In[ ]:


all_data.isnull().sum()


# In[ ]:


## no missing values in the data


# Check the statistics of this dataset

# In[ ]:


all_data.describe()


# In[ ]:


# submission_file.to_csv('all_1 values.csv', index=False)


# In[ ]:


all_data.nunique()


# In[ ]:


train.info()


# 3 int and rest are object

# Check unique value for each value

# In[ ]:


for col in all_data.columns:
    print(col)
    print(all_data[col].value_counts())
    print('==========================')


# In[ ]:


for col in train.columns:
    print(col)
    print(train[col].value_counts())
    print('==========================')


# In[ ]:


test.education_level.value_counts()


# Categorical  value analyses

# In[ ]:


cat_col = train.select_dtypes(exclude=np.number).drop(['uniqueid', 'bank_account'], axis=1)
num_col = train.select_dtypes(include=np.number)


# ## Univariate Categorical value plot

# Barplot with percentage of occurance for each value

# In[ ]:


## count plot function
def cat_plot (col):
    
    ## create 2 plot one for train and another for test
    fig, ((ax),(ax1)) = plt.subplots(1, 2, sharex=False, sharey=False,figsize=(15,5))
    
    f = sns.countplot(x=col, data=train, ax=ax)
        ## write ontop of box snippet
    for p in f.patches:
        ## Get box location
        x=p.get_bbox().get_points()[:,0]
        y=p.get_bbox().get_points()[1,1]
        ## write percentage ontop of box
        f.annotate('{:.1f}%'.format(100.*p.get_height()/train.shape[0]), (x.mean(), y.mean()), 
                ha='center', va='bottom') # set the alignment of the text
    f.title.set_text('Bar plot of train ' + col)
    f.set_xticklabels(f.get_xticklabels(), rotation=40, ha="right")
   
    ## plot for test
    f1 = sns.countplot(x=col, data=test, ax=ax1)
        ## write ontop of box snippet
    for p in ax1.patches:
        ## Get box location
        x=p.get_bbox().get_points()[:,0]
        y=p.get_bbox().get_points()[1,1]
        ## write percentage ontop of box
        f1.annotate('{:.1f}%'.format(100.*p.get_height()/test.shape[0]), (x.mean(), y.mean()), 
                ha='center', va='bottom') # set the alignment of the text
    f1.title.set_text('Bar plot of test ' + col)
    f1.set_xticklabels(f1.get_xticklabels(), rotation=40, ha="right")
    fig.tight_layout()
    fig.show()  


# In[ ]:


## Gui of countplot
@interact
def show_cat_plot(col=cat_col.columns):
    cat_plot(col)


# Stupid way of data spliting

# Categorical value analysis comparison with train and test

# * ``Country`` : Rwanda has high with about 37.1%, and Uganda low with 8.9% of the data. Both train and test are accurately seperated
# * ``bank account`` : highly imbalance with no value having 85%
# * ``location`` : There is more of rural than urban 61%
# * ``cellphoneaccess`` : There is more of cellphone access 74%
# * ``gender`` : There is more of female than male 58%
# * ``head_status`` : There is more of head 54.5%, spouse 27%, child 9.5%, parent 4.6%, relative other 2.8%, non, 0.8%. Maybe i can join other and relative
# * ``marital status`` : There is more of married 45.5%, single 33%, divorced 8.8%, widowed 11.5%, add dont know to divorced
# * ``education_level`` : There is more of primary 54% ,no edu 19%, sec 18%, tet 4.9%, voc 3.4%,  dont know 
# * ``job_type`` : There is more of self employed 27% ,informally employed 23%, farming n fishing 23%, remittance denpendent 10%, other income 4%,  formally employ 4%, no income 2.7%,  
# 
# 
# 

# ## Univariate Numeric value plot

# ### Histogram 

# Histogram

# In[ ]:


num_col.nunique()


# In[ ]:


def num_plot(col):
    
    
    ## create 2 plot one for train and another for test
    fig, ((ax),(ax1)) = plt.subplots(1, 2, sharex=False, sharey=False,figsize=(15,5))
    
   ## Am using the min and max value of the col to set x and y axis
    if col == 'age_of_respondent':
        ax.set_xlim(13,100)
        ax.set_xticks(range(13,110,4))
        ax1.set_xlim(13,110)
        ax1.set_xticks(range(13,101,4))
    if col == 'household_size':
        ax.set_xlim(0,25)
        ax.set_xticks(range(0,26))
        ax1.set_xlim(0,26)
        ax1.set_xticks(range(0,26))
    f = sns.distplot(train[col], rug=True, ax=ax)
        ## write ontop of box snippet
    f.title.set_text('hist plot of train ' + col)
#     f.set_xticklabels(f.get_xticklabels(), rotation=40, ha="right")
   
    ## plot for test
    f1 = sns.distplot(test[col],  rug=True, ax=ax1)
    f1.title.set_text('hist plot of test ' + col)
#     f1.set_xticklabels(f1.get_xticklabels(), rotation=40, ha="right")
    fig.tight_layout()
    fig.show()  
    plt.show()


# In[ ]:


# num_col.household_size.describe()


# In[ ]:


# num_col['age_of_respondent'].describe()


# In[ ]:


@interact
def show_num_plot(col=num_col.columns):
    num_plot(col)


# * ``Year`` is categorical in nature, maybe i should make it categorical. There is  more  that in 2016 than 2017 and 2018
# * ``Ageofrespondent`` is kinda categorical in nature, maybe i should make it categorical. There is  more  of 20- 39 of age 
# * ``householdsize`` is categorical in nature, maybe i should make it categorical. There is  more  that is in range of 2 -7. With 2 value been the most occurance
# 
# Combine household > 10, 11, or 12 as a group

# In[ ]:


sns.distplot(np.log(train.age_of_respondent))


# In[ ]:


# Look deeply into this data since it is categorical in nature, i can use value count


# #### Age analyses 

# In[ ]:


## get counto 10 highest age of train
test.age_of_respondent.value_counts()[:10]


# In[ ]:


## get counto 10 highest age of test
num_col.age_of_respondent.value_counts()[:10]


# #### House_hold size analyses

# In[ ]:


num_col.household_size.value_counts()


# ### Cumulative density Plot

# In[ ]:


num_bins = 20
counts, bin_edges = np.histogram (num_col.age_of_respondent, bins=num_bins, normed=True)
cdf = np.cumsum (counts)
plt.plot (bin_edges[1:], cdf/cdf[-1])


# ### Quantile Plot

# In[ ]:


# statsmodels Q-Q plot on model residuals
for col in num_col.columns:
    probplot(num_col[col], dist="norm", plot=plt)
    plt.show()


# In[ ]:


# statsmodels Q-Q plot on model residuals
for col in num_col.columns:
    probplot(np.log(train[col]), dist="norm", plot=plt)
    plt.show()


# In[ ]:


# Possible solution to numeric col
# log age
# make year and child num cat


# # Bivariate Plot

# ### Cat wrt  target variable ``bank``
# 

# In[ ]:


## count plot function
def cat_plot (col):
    
    ## create 2 plot one for train and another for test
    fig, ((ax),(ax1)) = plt.subplots(1, 2, sharex=False, sharey=False,figsize=(15,5))
    
    f = sns.countplot(x=col, data=train, ax=ax, hue='bank_account')
        ## write ontop of box snippet
    for p in f.patches:
        ## Get box location
        x=p.get_bbox().get_points()[:,0]
        y=p.get_bbox().get_points()[1,1]
        ## write percentage ontop of box
        f.annotate('{:.1f}%'.format(100.*p.get_height()/train.shape[0]), (x.mean(), y.mean()), 
                ha='center', va='bottom') # set the alignment of the text
    f.title.set_text('Bar plot of train ' + col)
    f.set_xticklabels(f.get_xticklabels(), rotation=40, ha="right")
   
    ## plot for test
    f1 = sns.countplot(x=col, data=test, ax=ax1)
        ## write ontop of box snippet
    for p in ax1.patches:
        ## Get box location
        x=p.get_bbox().get_points()[:,0]
        y=p.get_bbox().get_points()[1,1]
        ## write percentage ontop of box
        f1.annotate('{:.1f}%'.format(100.*p.get_height()/test.shape[0]), (x.mean(), y.mean()), 
                ha='center', va='bottom') # set the alignment of the text
    f1.title.set_text('Bar plot of test ' + col)
    f1.set_xticklabels(f1.get_xticklabels(), rotation=40, ha="right")
    fig.tight_layout()
    fig.show()  


# In[ ]:


@interact
def cat_with_targrt(col=cat_col.columns):
    cat_plot(col)


# In[ ]:


train.job_type.unique()


# ### Who has no cell_access but have  a bank_account

# In[ ]:


# Old people dont have cell_access but have bank_account, why??
## This happens mostly in kenya 
## mostly in rural area


# In[ ]:


for col in train.columns:
    print(col)
    print (train.loc[(train.bank_account == 'Yes') & (train.cellphone_access == 'No') & (train.age_of_respondent > 69)][col].value_counts())
    print('===================')


# * Kenya has a great number of people with bank account
# * Rural n urban has same rate of bank account status
# * Those with no cellphone access barely uses/have bank acct
# * male are more likely to have an account
# * head of house n spose are more likely to have an acccount
# * Married couple n single guys rules here
# * Although sec andprimary rules here, but we tend to see in voc and tetriary that banks acct owners are more than non-bank account owners
# * Although self_employed and fishing, but we tend to see in formally employed and government agent that banks acct owners are more than non-bank account owners
# 
# ###  Analyses to check
# Who are the gov and formal employed without account
# The rest find out about the account owners
# 
# 

# In[ ]:


@interact
def cat_with_targrt(col=num_col.columns):
    cat_plot(col)


# Not surprisinng that niggas got wise in 2018, low value in 2017 might be due to small density of data in 2017

# In[ ]:


test.job_type.value_counts()


# ### Numeric wrt target variable

# In[ ]:


def num_plot(col):
    
    
    ## create 2 plot one for train and another for test
    fig, ((ax),(ax1)) = plt.subplots(1, 2, sharex=False, sharey=False,figsize=(15,5))
    
   ## Am using the min and max value of the col to set x and y axis
    if col == 'age_of_respondent':
        ax.set_xlim(13,100)
        ax.set_xticks(range(13,110,4))
        ax1.set_xlim(13,110)
        ax1.set_xticks(range(13,101,4))
    if col == 'household_size':
        ax.set_xlim(0,25)
        ax.set_xticks(range(0,26))
        ax1.set_xlim(0,26)
        ax1.set_xticks(range(0,26))
    f = sns.distplot(train.loc[train.bank_account == 'No'][col],hist=False, rug=True, ax=ax, label="No")
    f = sns.distplot(train.loc[train.bank_account == 'Yes'][col],hist=False, rug=True, ax=ax, label="Yes")
    
        ## write ontop of box snippet
    f.title.set_text('hist plot of train ' + col)
#     f.set_xticklabels(f.get_xticklabels(), rotation=40, ha="right")
   
    ## plot for test
    f1 = sns.distplot(test[col],  rug=True, ax=ax1)
    f1.title.set_text('hist plot of test ' + col)
#     f1.set_xticklabels(f1.get_xticklabels(), rotation=40, ha="right")
    fig.tight_layout()
    fig.show()  
    plt.show()


# In[ ]:


@interact
def num_wrt_target(col=num_col.columns):
    num_plot(col)


# In[ ]:


## does houeshold size really affect acct number??
## Those in their 20's tend to use a bank.
# What if i analyze those in 70's who have and analyse those in 20's who do not have


# I want to look at countries distribution

# ### Categorical Feature  wrt another Categorical feature

# #### Analyse wrt to country
# i know that kenya has a larger percentage, but can we look at what happens in this country, what age range are in tis country, and other stuffs

# In[ ]:


## count plot function
def cat_plot (col):
    
    ## create 2 plot one for train and another for test
    fig, ((ax),(ax1)) = plt.subplots(1, 2, sharex=False, sharey=False,figsize=(15,5))
    
    f = sns.countplot(x=col, data=train, ax=ax, hue='country')
        ## write ontop of box snippet
    for p in f.patches:
        ## Get box location
        x=p.get_bbox().get_points()[:,0]
        y=p.get_bbox().get_points()[1,1]
        ## write percentage ontop of box
        f.annotate('{:.1f}%'.format(100.*p.get_height()/train.shape[0]), (x.mean(), y.mean()), 
                ha='center', va='bottom') # set the alignment of the text
    f.title.set_text('Bar plot of train ' + col)
    f.set_xticklabels(f.get_xticklabels(), rotation=40, ha="right")
   
    ## plot for test
    f1 = sns.countplot(x=col, data=test, ax=ax1,hue='country')
        ## write ontop of box snippet
    for p in ax1.patches:
        ## Get box location
        x=p.get_bbox().get_points()[:,0]
        y=p.get_bbox().get_points()[1,1]
        ## write percentage ontop of box
        f1.annotate('{:.1f}%'.format(100.*p.get_height()/test.shape[0]), (x.mean(), y.mean()), 
                ha='center', va='bottom') # set the alignment of the text
    f1.title.set_text('Bar plot of test ' + col)
    f1.set_xticklabels(f1.get_xticklabels(), rotation=40, ha="right")
    fig.tight_layout()
    fig.show()  


# In[ ]:


@interact
def cat_with_targrt(col=cat_col.columns):
    cat_plot(col)


# * ``Location type``: More data was captured in the urban of ``tanzania`` than the rural area. ``Rwanda`` has a high level of rural capture than the urban. ``Kenya and Uganda``  looks almost balance in both region, though the rural is winning
# * ``Cellphone access``: Cellphone access is leading in this area. But ``tanzania``has a slight high level of no access , despite having high level of urban users. 
# * ``gender``: There is almost an equal share of this, though the femal still wins in all country
# * ``head of house status``: Tanzania tend to accomodate parent than child, compared to kenya who accomodate more child than parent, Uganda also accomodate parent
# * ``Marital_status``: Most people in Tanzania are single or divorced, but the rsre country has more mmarried people
# * ``education_level``:We have mor graduate in Tanzania than in any other country, Kenya has more sec school people
# * ``job_type``:More self employed people can be found in ``tanzania``:no farming and fishing nor formally employed there,``Rwanda`` are more of the fishing and farming type and also informally employed: no no income . Maybe this was due to high data captured in the rural than urban  area. ``Uganda`` guys no farming and fishing nor formally employed, remittane dependent, Informaly employed. Most guys there are sel_employed. The ``kenya`` guys has a bit of almost every jobs

# #### NUm

# In[ ]:


def num_plot(col):
    
    
    ## create 2 plot one for train and another for test
    fig, ((ax),(ax1),(ax2),(ax3)) = plt.subplots(2, 2, sharex=False, sharey=False,figsize=(15,5))
    
   ## Am using the min and max value of the col to set x and y axis
    if col == 'age_of_respondent':
        ax.set_xlim(13,100)
        ax.set_xticks(range(13,110,4))
        ax1.set_xlim(13,110)
        ax1.set_xticks(range(13,101,4))
    if col == 'household_size':
        ax.set_xlim(0,25)
        ax.set_xticks(range(0,26))
        ax1.set_xlim(0,26)
        ax1.set_xticks(range(0,26))
    f = sns.distplot(train.loc[train.country == 'Rwanda'][col],hist=False, rug=True, ax=ax, label="Rwanda")
    f = sns.distplot(train.loc[train.country == 'Tanzania'][col],hist=False, rug=True, ax=ax, label="Tanzania")
    f = sns.distplot(train.loc[train.country == 'Kenya'][col],hist=False, rug=True, ax=ax, label="Kenya")
    f = sns.distplot(train.loc[train.country == 'Uganda'][col],hist=False, rug=True, ax=ax, label="Uganda")

    
        ## write ontop of box snippet
    f.title.set_text('hist plot of train ' + col)
#     f.set_xticklabels(f.get_xticklabels(), rotation=40, ha="right")
   
    ## plot for test
    f1 = sns.distplot(train.loc[train.country == 'Tanzania'][col],  rug=True, ax=ax1)
    f1.title.set_text('hist plot of test ' + col)
#     f1.set_xticklabels(f1.get_xticklabels(), rotation=40, ha="right")
    fig.tight_layout()
    fig.show()  
    plt.show()


# In[ ]:


@interact
def num_wrt_target(col=num_col.columns):
    num_plot(col)


# In[ ]:


@interact
def cat_with_targrt(col=num_col.columns):
    cat_plot(col)


# Analyse bank account users in each country

# In[ ]:


train.loc[train.country == 'Uganda']['age_of_respondent'].value_counts()
# train.loc[train.country == 'Tanzania'][col],hist=False, rug=True, ax=ax, label="Tanzania")
#     f = sns.distplot(train.loc[train.country == 'Kenya'][col],hist=False, rug=True, ax=ax, label="Kenya")
#     f = sns.distplot(train.loc[train.country == 'Uganda'][col],hist=False, rug=True, ax=ax, label="Uganda")


# It seems the distribution of country comes from different year
# The Kenya guys dont want too much children, 1-3 is enough, Rwanda guys like 3 to 6 children, is it because they are in the rural area

# In[ ]:


train.country.value_counts()


# ### country seperately

# In[ ]:


def num_plot(col):
    
    
    ## create 2 plot one for train and another for test
    fig, ((ax,ax1),(ax2,ax3)) = plt.subplots(2, 2, sharex=False, sharey=False,figsize=(15,5))
    
   ## Am using the min and max value of the col to set x and y axis
    if col == 'age_of_respondent':
        ax.set_xlim(13,100)
        ax.set_xticks(range(13,110,4))
        ax1.set_xlim(13,110)
        ax1.set_xticks(range(13,101,4))
    if col == 'household_size':
        ax.set_xlim(0,25)
        ax.set_xticks(range(0,26))
        ax1.set_xlim(0,26)
        ax1.set_xticks(range(0,26))
#     f = sns.distplot(train.loc[train.country == 'Rwanda'][col],hist=False, rug=True, ax=ax, label="Rwanda")
#     f = sns.distplot(train.loc[train.country == 'Tanzania'][col],hist=False, rug=True, ax=ax, label="Tanzania")
#     f = sns.distplot(train.loc[train.country == 'Kenya'][col],hist=False, rug=True, ax=ax, label="Kenya")
#     f = sns.distplot(train.loc[train.country == 'Uganda'][col],hist=False, rug=True, ax=ax, label="Uganda")
    f = sns.countplot(x=train.loc[train.country == 'Rwanda'][col], data=train, ax=ax, hue='bank_account')
        ## write ontop of box snippet
    for p in f.patches:
        ## Get box location
        x=p.get_bbox().get_points()[:,0]
        y=p.get_bbox().get_points()[1,1]
        ## write percentage ontop of box
        f.annotate('{:.1f}%'.format(100.*p.get_height()/train.shape[0]), (x.mean(), y.mean()), 
                ha='center', va='bottom') # set the alignment of the text
    f.title.set_text('Bar plot of train Rwanda ' + col)
    f.set_xticklabels(f.get_xticklabels(), rotation=40, ha="right")
    
        ## write ontop of box snippet
#     f.title.set_text('hist plot of train ' + col)
#     f.set_xticklabels(f.get_xticklabels(), rotation=40, ha="right")
   
    ## plot for tanzania
    f1 = sns.countplot(x=train.loc[train.country == 'Tanzania'][col], data=train, ax=ax1, hue='bank_account')
        ## write ontop of box snippet
    for p in f1.patches:
        ## Get box location
        x=p.get_bbox().get_points()[:,0]
        y=p.get_bbox().get_points()[1,1]
        ## write percentage ontop of box
        f1.annotate('{:.1f}%'.format(100.*p.get_height()/train.shape[0]), (x.mean(), y.mean()), 
                ha='center', va='bottom') # set the alignment of the text
    f1.title.set_text('Bar plot of train Tanzania ' + col)
    f1.set_xticklabels(f1.get_xticklabels(), rotation=40, ha="right")
    
      
#     f1.set_xticklabels(f1.get_xticklabels(), rotation=40, ha="right")

    ## plot for tanzania
    f2 = sns.countplot(x=train.loc[train.country == 'Kenya'][col], data=train, ax=ax2, hue='bank_account')
        ## write ontop of box snippet
    for p in f2.patches:
        ## Get box location
        x=p.get_bbox().get_points()[:,0]
        y=p.get_bbox().get_points()[1,1]
        ## write percentage ontop of box
        f2.annotate('{:.1f}%'.format(100.*p.get_height()/train.shape[0]), (x.mean(), y.mean()), 
                ha='center', va='bottom') # set the alignment of the text
    f2.title.set_text('Bar plot of train Kenya ' + col)
    f2.set_xticklabels(f2.get_xticklabels(), rotation=40, ha="right")
    

    
    f3 = sns.countplot(x=train.loc[train.country == 'Uganda'][col], data=train, ax=ax3, hue='bank_account')
        ## write ontop of box snippet
    for p in f3.patches:
        ## Get box location
        x=p.get_bbox().get_points()[:,0]
        y=p.get_bbox().get_points()[1,1]
        ## write percentage ontop of box
        f3.annotate('{:.1f}%'.format(100.*p.get_height()/train.shape[0]), (x.mean(), y.mean()), 
                ha='center', va='bottom') # set the alignment of the text
    f3.title.set_text('Bar plot of train Uganda ' + col)
    f3.set_xticklabels(f3.get_xticklabels(), rotation=40, ha="right")
    
    fig.tight_layout()
    fig.show()  
    plt.show()


# In[ ]:


@interact
def num_wrt_target(col=train.columns):
    num_plot(col)


# In[ ]:


@interact
def num_wrt_target(col=num_col.columns):
    num_plot(col)


# Funny enough,the rural tends to have more bank account owners than the rural, except in kenya and Uganda where the urban guys win.
# Even tanzania that captures more of urban are lacking in bank_acct
# 
# * Cell phone access: Looking deeply, we see that no access to cell phone means that no account im Rwanda , kenya and Tanzania. Look deeply
# * Male or more likely to use bank account 
# * Head of house and sometimes spouse are more likely to use bank account 
# * Married and single are  likely to use bank account , but in Tanzania we are looking at the divorced and single winning this
# * Vocational specialist and tertiary education guy tend to use more acct, we can analyse more about this guys
# * formally employed private tend to win in Kenya
# 

# In[ ]:


train.loc[(train.country == 'Uganda') & (train.bank_account == 'Yes')]['job_type'].value_counts()


# From analysis, type of work(income)and cellphone access really affects bank accout and  also  level of education
