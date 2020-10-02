#!/usr/bin/env python
# coding: utf-8

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


import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


df = pd.read_csv('../input/loandata/Loan payments data.csv')


# In[ ]:


df.head()


# In[ ]:


print(df.info())
print('-------------')
print(df.shape)
print('The total rows are ',df.shape[0],' and the total features are ',df.shape[1]-1)
print('The predictor variable is loan status, divided into 3 cateogories as ',df['loan_status'].unique())


# In[ ]:


fig = plt.figure(figsize=(5,5))
ax = sns.countplot(df['loan_status'])
ax.set_title("Count of Loan Status")
for p in ax.patches:
    ax.annotate(p.get_height(), (p.get_x()+0.25, p.get_height()*1.01))


# In[ ]:


fig, axs = plt.subplots(1, 2, figsize=(16, 5))
sns.boxplot(x="loan_status", y="Principal", data=df, hue="loan_status", ax=axs[0])
sns.distplot(df['Principal'], ax=axs[1])


# In[ ]:


ax = sns.countplot(df['terms'])
for p in ax.patches:
    ax.annotate(p.get_height(), (p.get_x()+0.25, p.get_height()*1.01))


# In[ ]:


ax = sns.countplot(df['education'])
for p in ax.patches:
    ax.annotate(p.get_height(), (p.get_x()+0.25, p.get_height()*1.01))


# In[ ]:


ax = sns.countplot(df['education'],hue='loan_status',data=df)
for p in ax.patches:
    ax.annotate(p.get_height(), (p.get_x(), p.get_height()*1.01))


# In[ ]:


fig, axs = plt.subplots(1, 2, figsize=(15, 5))

ax = sns.countplot(x='Gender',data=df, ax=axs[0])
for p in ax.patches:
    ax.annotate(p.get_height(), (p.get_x(), p.get_height()*1.01)) 

ax = sns.countplot(x='loan_status', hue='Gender', data=df, ax=axs[1])
for p in ax.patches:
    ax.annotate(p.get_height(), (p.get_x(), p.get_height()*1.01))


# In[ ]:


ax = sns.countplot(df['education'],hue='Gender',data=df)
for p in ax.patches:
    ax.annotate(p.get_height(), (p.get_x(), p.get_height()*1.01))


# In[ ]:


fig = plt.figure(figsize=(15,5))
ax = sns.countplot(df['effective_date'],hue=df['Principal'],data=df)
for p in ax.patches:
    ax.annotate(p.get_height(), (p.get_x(), p.get_height()*1.01))


# The loan data is for Sept 8th to 14th. With most of the loans originating/drawn on Sept 11. Sep 10,11 and 12 make for the bulk of the business for the week. 400+ loans out of a total of 500 for the week.

# In[ ]:


fig = plt.figure(figsize=(15,5))
ax = sns.countplot(df['age'],hue=df['education'],data=df)
for p in ax.patches:
    ax.annotate(p.get_height(), (p.get_x(), p.get_height()*1.01))


# Interesting insights, the preference for universities was more among the older population and the younger demographics seem to prefer Colleges POTENTIAL REASON: Might still be pursuing/Rising tuition cost in last few years. 4 instances of masters also seen in older demographics.

# In[ ]:


fig, axs = plt.subplots(2, 1, figsize=(16, 7)) 

ax = sns.countplot(df['age'],ax=axs[0])
ax = sns.countplot(df['age'],hue=df['Principal'],data=df,ax =axs[1])


# The need for payday(bullet loans) seems to peak till 30 and then reduces as people tend to become financially stable. The probablity for credit and the amount required both seem to trend. Ages 24-38 tend to get loan of $$800 - $1000.

# In[ ]:


fig = plt.figure(figsize=(15,5))
ax = sns.countplot(df['Principal'],hue=df['terms'],data=df)
for p in ax.patches:
    ax.annotate(p.get_height(), (p.get_x(), p.get_height()*1.01))


# Most of the loans are 800 or 1000, with hardly any 900 loan which is odd. 800 are almost exclusively termed for biweekly while the $1000 are split 1:2.5 in ratio of biweekly/monthly terms.

# In[ ]:


fig = plt.figure(figsize=(15,5))
ax = sns.countplot(df['age'],hue=df['loan_status'],data=df)


# The loans in collections almost mimic the trend of loan seeking, thus correlation seems is based on demand for the loan.

# In[ ]:


df['day_to_pay'] = (pd.DatetimeIndex(df['paid_off_time']).normalize() - pd.DatetimeIndex(df['effective_date']).normalize())/ np.timedelta64(1, 'D')
fig = plt.figure(figsize=(25, 5))
ax = sns.countplot(x='day_to_pay', hue='terms', data=df)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
for p in ax.patches:
    ax.annotate(p.get_height(), (p.get_x(), p.get_height()*1.01))


# Spikes seen on 14 and 29 day, as the repayment is for biweekly and monthly loans.

# In[ ]:


df['paid_off_date'] = pd.DatetimeIndex(df['paid_off_time']).normalize()
df['paid_off_date']=df['paid_off_date'].dt.date      #removing the time component
fig = plt.figure(figsize=(16, 6))
ax = sns.countplot(x='paid_off_date', data=df.loc[df['loan_status'].isin(['COLLECTION_PAIDOFF', 'PAIDOFF'])] , hue='loan_status')
ax.set_xticklabels(ax.get_xticklabels(), rotation=90);


# In[ ]:


g = sns.catplot(x="terms", hue="effective_date", col="loan_status",data=df, kind="count",height=4, aspect=.7)


# Majority of loans seem to be paid of on time, the ones who paid after moving to collections also is a bit confusing as the graphs are identical with the Paid of loans. The loans with 7 days credit period are the paid off without any adverse behavior.

# In[ ]:




