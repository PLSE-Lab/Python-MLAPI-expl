#!/usr/bin/env python
# coding: utf-8

# **Thanks for viewing my Kernel! If you like my work and find it useful, please leave an upvote! :)**
# 
# Some columns of the data are in JSON format. I have written a separate [kernel](https://www.kaggle.com/arunsankar/import-data-and-convert-into-table-format/notebook) to parse the data and extract train and test tabular datasets. I will use the output files directly to save execution time. 

# **Key insights:**
# * Only 1.4% of the visitors in training set have resulted in revenue. That is why it's critical to identify the customers with revenue potential
# * Mobile device visits generate revenue only from 0.4% of transactions whereas non-mobile devices generate revenue from 1.6% of transactions. Despite the industries' focus on mobile, mobile has really not come closer to non-mobile devices. 
# * Even the average revenue per transaction of non-mobile devices are twice more than that of mobile devices.
# * Chrome browser has the most number of transactions and the highest % of transactions with revenue (almost 3 times more than any other browser). On % of transactions with revenue, Chrome is followed by Internet Explorer, Edge, Firefox and then Safari. 
# 
# Note: I have commented the code for test and submission data explorations for faster execution. 

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import os
print(os.listdir("../input"))


# In[ ]:


train = pd.read_csv('../input/import-data-and-convert-into-table-format/df_train.csv', low_memory=False)
test = pd.read_csv('../input/import-data-and-convert-into-table-format/df_test.csv', low_memory=False)
sub = pd.read_csv('../input/ga-customer-revenue-prediction/sample_submission.csv', low_memory=False)

print('Train data: \nRows: {}\nCols: {}'.format(train.shape[0],train.shape[1]))
print(train.columns)

print('\nTest data: \nRows: {}\nCols: {}'.format(test.shape[0],test.shape[1]))
print(test.columns)

print('\nSubmission format: \nRows: {}\nCols: {}'.format(sub.shape[0],sub.shape[1]))
print(sub.columns)


# Submission has less records than test. That is because the objective is to predict log revenue for each unique 'fullVisitorId'.

# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


sub.head()


# In[ ]:


print('Unique visitor IDs in training data: {}'.format(train['fullVisitorId'].nunique()))
#print('Unique visitor IDs in test data: {}'.format(test['fullVisitorId'].nunique()))


# In[ ]:


train["totals.transactionRevenue"] = train["totals.transactionRevenue"].astype('float')
temp = train.groupby('fullVisitorId')['totals.transactionRevenue'].sum().reset_index()
print('Percentage of visitors with revenue is {:,.2%}'.format(temp[temp['totals.transactionRevenue'] > 0]['fullVisitorId'].count() / train['fullVisitorId'].nunique()))


# Only 1.4% of the visitors have resulted in some revenue. That is why it's very important to identify the customers with revenue potential

# Indicator to identify a non-revenue and revenue transaction.

# In[ ]:


train['revenue_flag'] = train['totals.transactionRevenue'].apply(lambda x: 1 if x>0 else 0)


# Mobile device visits generate revenue only from 0.4% of transactions whereas non-mobile devices generate revenue from 1.6% of transactions. Despite the industries' focus on mobile, mobile has really not come closer to non-mobile devices. 

# In[ ]:


mobile_revenue = train.pivot_table(train, index=['device.isMobile'], columns=['revenue_flag'], aggfunc=len).reset_index()[['device.isMobile','totals.transactionRevenue']]
mobile_revenue.columns = ['device.isMobile','No Revenue', 'Revenue']
mobile_revenue['Revenue %'] = mobile_revenue['Revenue'] / (mobile_revenue['Revenue'] + mobile_revenue['No Revenue'])

fig, ax = plt.subplots(figsize=(8,4))
sns.barplot(x='Revenue %', y='device.isMobile', data=mobile_revenue, color="lightskyblue", orient="h")

for p in ax.patches:
    ax.text(p.get_width() + 0.0006, 
            p.get_y() + (p.get_height()/2), 
            '{:,.1%}'.format(p.get_width()),
            ha="center")
    
ax.set_ylabel('Device is Mobile?', size=14, color="#0D47A1")
ax.set_xlabel('% of transactions with revenue', size=14, color="#0D47A1")
ax.set_title('Percentage of transactions with revenue by device type', size=18, color="#0D47A1")

plt.show()


# Even the average revenue per transaction of non-mobile devices are twice more than that of mobile devices.

# In[ ]:


mobile_revenue = train.pivot_table(train, index=['device.isMobile'], columns=['revenue_flag'], aggfunc=np.mean).reset_index()[['device.isMobile','totals.transactionRevenue']]
mobile_revenue.columns = ['device.isMobile','Avg Revenue']

fig, ax = plt.subplots(figsize=(8,4))
sns.barplot(x='Avg Revenue', y='device.isMobile', data=mobile_revenue, color="lightskyblue", orient="h")

for p in ax.patches:
    ax.text(p.get_width() + 10000000, 
            p.get_y() + (p.get_height()/2), 
            '${:,.0f}'.format(p.get_width()),
            ha="center")
    
ax.set_ylabel('Device is Mobile?', size=14, color="#0D47A1")
ax.set_xlabel('Average revenue', size=14, color="#0D47A1")
ax.set_title('Average revenue of by device type', size=18, color="#0D47A1")

plt.show()


# Chrome browser has the most number of transactions and the highest % of transactions with revenue (almost 3 times more than any other browser). On % of transactions with revenue, Chrome is followed by Internet Explorer, Edge, Firefox and then Safari. 

# In[ ]:


browser_revenue = train.pivot_table(train, index=['device.browser'], columns=['revenue_flag'], aggfunc=len).reset_index()[['device.browser','totals.transactionRevenue']]
browser_revenue.columns = ['device.browser','No Revenue', 'Revenue']
browser_revenue.fillna(0, inplace=True)
browser_revenue['Transactions'] = browser_revenue['Revenue'] + browser_revenue['No Revenue']
browser_revenue['Revenue %'] = browser_revenue['Revenue'] / browser_revenue['Transactions']
browser_revenue = browser_revenue.sort_values('Transactions',ascending=False).head(15)

fig, ax = plt.subplots(1, 2, figsize=(8,8), sharey=True)
a = sns.barplot(x='Transactions', y='device.browser', data=browser_revenue, color="lightskyblue", orient="h", ax=ax[0])
b = sns.barplot(x='Revenue %', y='device.browser', data=browser_revenue, color="lightskyblue", orient="h", ax=ax[1])

for p in ax[0].patches:
    ax[0].text(p.get_width() + 75000, 
            p.get_y() + (p.get_height()/2), 
            '{:,.0f}'.format(p.get_width()),
            ha="center")
    
for p in ax[1].patches:
    ax[1].text(p.get_width() + 0.001, 
            p.get_y() + (p.get_height()/2), 
            '{:,.1%}'.format(p.get_width()),
            ha="center")
    
ax[0].set_ylabel('Browsers', size=14, color="#0D47A1")
ax[0].set_xlabel('Total Number of Transactions', size=14, color="#0D47A1")

ax[1].set_ylabel('', size=14, color="#0D47A1")
ax[1].set_xlabel('% of Transactions with revenue', size=14, color="#0D47A1")

plt.show()


# **More to come..**
