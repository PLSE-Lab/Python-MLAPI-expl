#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sn
dataset=pd.read_csv('../input/churn_data.csv')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


dataset.head(5)


# In[ ]:


dataset.columns


# In[ ]:


dataset.describe()


# In[ ]:


dataset[dataset.credit_score<300]


# In[ ]:


dataset=dataset[dataset.credit_score>=300]


# In[ ]:


dataset.isna().any()


# In[ ]:





# In[ ]:


dataset.isna().sum()


# In[ ]:


dataset.drop(columns=['credit_score','rewards_earned'])


# In[ ]:


dataset2=dataset.drop(columns=['user','churn'])


# In[ ]:





# In[ ]:


dataset2 = dataset[['housing', 'is_referred', 'app_downloaded',
                    'web_user', 'app_web_user', 'ios_user',
                    'android_user', 'registered_phones', 'payment_type',
                    'waiting_4_loan', 'cancelled_loan',
                    'received_loan', 'rejected_loan', 'zodiac_sign',
                    'left_for_two_month_plus', 'left_for_one_month', 'is_referred']]
fig = plt.figure(figsize=(15, 12))
plt.suptitle('Pie Chart Distributions', fontsize=20)
for i in range(1, dataset2.shape[1] + 1):
    plt.subplot(6, 3, i)
    f = plt.gca()
    f.axes.get_yaxis().set_visible(False)
    f.set_title(dataset2.columns.values[i - 1])
   
    values = dataset2.iloc[:, i - 1].value_counts(normalize = True).values
    index = dataset2.iloc[:, i - 1].value_counts(normalize = True).index
    plt.pie(values, labels = index, autopct='%1.1f%%')
    plt.axis('equal')
fig.tight_layout(rect=[0, 0.03, 1, 0.95])


# In[ ]:


dataset[dataset2.waiting_4_loan==1].churn.value_counts()


# In[ ]:


dataset[dataset2.cancelled_loan == 1].churn.value_counts()


# In[ ]:


dataset[dataset2.received_loan == 1].churn.value_counts()


# In[ ]:


dataset[dataset2.rejected_loan == 1].churn.value_counts()


# In[ ]:


dataset[dataset2.left_for_one_month == 1].churn.value_counts()


# In[ ]:


dataset2.drop(columns = ['housing', 'payment_type',
                        'registered_phones', 'zodiac_sign']
   ).corrwith(dataset.churn).plot.bar(figsize=(20,10),
             title = 'Correlation with Response variable',
             fontsize = 15, rot = 45,
             grid = True)


# In[ ]:


sn.set(style="white")
corr = dataset.drop(columns = ['user', 'churn']).corr()
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
f, ax = plt.subplots(figsize=(18, 15))
cmap = sn.diverging_palette(220, 10, as_cmap=True)
sn.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})


# In[ ]:


dataset = dataset.drop(columns = ['app_web_user'])


# In[ ]:


dataset.to_csv('new_churn_data.csv', index = False)


# In[ ]:





# In[ ]:





# In[ ]:




