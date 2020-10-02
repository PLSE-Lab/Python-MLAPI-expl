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

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


#import packages
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


#Load data
restaurant_scores = pd.read_csv('../input/restaurant-scores-lives-standard.csv', header=0)


# In[ ]:


restaurant_scores.shape


# In[ ]:


restaurant_scores.head()


# In[ ]:


sns.countplot(x="risk_category", data=restaurant_scores);


# In[ ]:


restaurant_violation_measure = restaurant_scores[restaurant_scores.risk_category == 'High Risk'].groupby(['business_id', 'business_name'])['inspection_id'].count().reset_index(name='Severe_violation_count').sort_values(ascending=False, by='Severe_violation_count').head(10)


# In[ ]:


plt_major_violations = sns.barplot(x='Severe_violation_count', y='business_name', data=restaurant_violation_measure)
plt_major_violations.set_title("Restaurant with Major Violations")
plt_major_violations.set(xLabel="Violations Count", yLabel="Business name")
plt.plot();


# In[ ]:


tickets_count_by_inspection =  restaurant_scores.groupby(['inspection_id'])['inspection_id'].count().reset_index(name='tickets_count').sort_values(ascending=False, by='tickets_count').head(10)


# In[ ]:


plt_tickets_count = sns.barplot(x='tickets_count', y='inspection_id', data=ticket_count_by_inspection, orient='h')
plt_tickets_count.set_title("Ticket Counts by Inspector")
plt_tickets_count.set(xLabel="Tickets Count", yLabel="Inspection ID")
plt.plot();

