#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
print(os.listdir("../input"))


# Import required libraries

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
sns.set()
import matplotlib.pyplot as plt
import datetime as dt

import warnings
warnings.filterwarnings('ignore')


# ### Load the datasets.

# In[ ]:


merchants = pd.read_csv('../input/merchants.csv')
historical_transactions = pd.read_csv('../input/historical_transactions.csv')
new_merchant_transactions = pd.read_csv('../input/new_merchant_transactions.csv')
train = pd.read_csv('../input/train.csv', parse_dates = ['first_active_month'])
test = pd.read_csv('../input/test.csv', parse_dates = ['first_active_month'])


# In[ ]:


print("Historical_transactions   :", historical_transactions.shape)
print("Merchants                 :", merchants.shape)
print("New_merchant_transactions :", new_merchant_transactions.shape)
print("Train set                 :", train.shape)
print("Test set                  :", test.shape)


# We will first explore our train and test dataset and mainly the target variable.

# ### Explore train dataset

# In[ ]:


print(pd.read_excel('../input/Data_Dictionary.xlsx', sheet_name = 'train'))


# In[ ]:


print('target:', pd.read_excel('../input/Data_Dictionary.xlsx', sheet_name = 'train')['Unnamed: 1'][7])


# So our target variable is the loyalty score (numerical). Though it is not cleared how it is calculated. We will look into it further.

# In[ ]:


train.head(2)


# In[ ]:


train.describe()


# Interestingly our target variable has max value of -33.xx and max of 17.xx. But they are not distributed normally. 25th to 72th quartile value lies in (+-)0.xx values. Lets look at them visually.

# In[ ]:


plt.figure(figsize = (10,6))
sns.distplot(train.target)
plt.show()


# In[ ]:


plt.figure(figsize = (10,5))
plt.scatter(train.index, np.sort(train.target.values))
plt.xlabel('Index')
plt.ylabel('Loyalty Score')
plt.show()


# So, the distribution seems normal. Though it contains an unexpected value of -33.xx. Lets see how much.

# In[ ]:


print((train.target == -33.21928095).sum())
print((train.target == -33.21928095).sum()*100 / len(train.target))


# 1% of data. We will look into it further. Lets look at the other variables in the train data.
# 
# We will look if there is something which Feature_x variables can find for the target variable.

# In[ ]:


plt.figure(figsize = (14,6))
sns.boxplot('feature_1', 'target', data = train)
plt.title('Feature_1 distribution (loyalty score)')
plt.show()

plt.figure(figsize = (12,6))
sns.boxplot('feature_2', 'target', data = train)
plt.title('Feature_2 distribution (loyalty score)')
plt.show()

plt.figure(figsize = (8,4))
sns.boxplot('feature_3', 'target', data = train)
plt.title('Feature_3 distribution (loyalty score)')
plt.show()


# Looks like Feature_x variables are not very good or helpful estimators for target variable. Here arises another question why not dig more our target variable. Loyalty score with negative value. There is definately some calculations behind the true loyalty score and to be it seems like an exponential or logarithmic function behind this due to -33.xx.

# After search I found this great kernel by @raddar Sir. Link to kernel : https://www.kaggle.com/raddar/towards-de-anonymizing-the-data-some-insights. Target value was revealed as 10^(target(log10(2))) which is simply 2**target.

# In[ ]:


train['new_target'] = 2**train.target


# In[ ]:


train.new_target.describe()


# In[ ]:


plt.figure(figsize = (10,6))
sns.distplot(train.new_target)
plt.show()


# In[ ]:


train.sort_values(by = 'new_target').head(5)


# In[ ]:


train.sort_values(by = 'new_target').tail(5)


# Looking at graph, I got some confused. But its clear now. Only 0.02% card_ids have loyalty score of 0.xx and thats expected (I expected even more than this).
# 
# So, target variable is set. Remeber we will have to reverse the calculation after the prediction in the test set.
# 
# Lets again plot Feature_x distribution with the new_target variable

# In[ ]:


plt.figure(figsize = (14,6))
sns.boxplot('feature_1', 'new_target', data = train)
plt.title('Feature_1 distribution (loyalty score)')
plt.show()

plt.figure(figsize = (12,6))
sns.boxplot('feature_2', 'new_target', data = train)
plt.title('Feature_2 distribution (loyalty score)')
plt.show()

plt.figure(figsize = (8,4))
sns.boxplot('feature_3', 'new_target', data = train)
plt.title('Feature_3 distribution (loyalty score)')
plt.show()


# Results are not great but yes, difference can be seen between each category of each variable. Thus, now, Feature_x variable will help some how for prediction in the test set.
# 
# Now, we are left with only one varible to explore i.e., first_active_month which gives us the customer first month of purchase.

# In[ ]:


train['year'] = train['first_active_month'].dt.year


# In[ ]:


plt.figure(figsize = (8,6))
plt.scatter(train.year, train.new_target, alpha = 0.5)
plt.xlabel('First Active Year')
plt.ylabel('Loyalty Score (new_target)')
plt.title('Loyalty Score vs First Active Year')
plt.show()


# So, loyalty score are high for the 2016 and 2017 first active users. 
# 
#  Lets look if the distribution of train and test dataset based on first_active_month is same or different.

# In[ ]:


month_count = train.first_active_month.dt.date.value_counts().sort_index()

plt.figure(figsize = (14,6))
sns.barplot(month_count.index, month_count.values, color = 'r')
plt.xticks(rotation = 'vertical')
plt.title('Train data distribution based on first active month')
plt.show()

plt.figure(figsize = (14,6))
month_count = test.first_active_month.dt.date.value_counts().sort_index()
sns.barplot(month_count.index, month_count.values, color = 'g')
plt.xticks(rotation = 'vertical')
plt.title('Test data distribution based on first active month')
plt.show()


# Yes, the distribution is same enough.

# ### New Merchant Transactions & Historical Transactions

# new_merchant_tranactions and historical transactions have same variables. Thus, I will only explore and work on new_merchant_transactions and will update the manipulation in the historical transactions.

# In[ ]:


print(pd.read_excel('../input/Data_Dictionary.xlsx', sheet_name = 'new_merchant_period'))


# In[ ]:


new_merchant_transactions.head()


# **purchase_amount**
# 
# Purchase amount should be positive but here it is not. Again, some sort of calculation is required to extract original purchase amount.
# 
# Again the kernel by [@raddar](http://kaggle.com/raddar) sir gave some magic numbers for transforming the purchase amount. But, I was not sure how this number came. Thanks to [@CPMP](http://kaggle.com/cpmpml) sir. His kernel helped to find pattern from the data and transform them using codes. Link to the kernel: https://www.kaggle.com/cpmpml/raddar-magic-explained-a-bit/
# 
# 

# In[ ]:


hist_new = pd.concat((historical_transactions, new_merchant_transactions), ignore_index = True)


# In[ ]:


hist_new.shape


# In[ ]:


hist_new.purchase_amount.describe()


# Make purchase amount non-negative.

# In[ ]:


hist_new['new_amount'] = hist_new.purchase_amount - hist_new.purchase_amount.min()


# In[ ]:


hist_new.new_amount.describe()


# Look at unique values we got after the new_amount

# In[ ]:


np.sort(hist_new.new_amount.unique())[:10]


# There looks a pattern here

# In[ ]:


np.diff(np.sort(hist_new.new_amount.unique()))


# Let do this steps in a proper way

# In[ ]:


hist_new_sorted = hist_new.groupby('new_amount').new_amount.first().to_frame().reset_index(drop=True)
hist_new_sorted['delta'] = hist_new_sorted.new_amount.diff()
hist_new_sorted[hist_new_sorted.delta >= 2e-5].head()


# In[ ]:


hist_new_sorted[1:52623].delta.mean()


# In[ ]:


hist_new['new_amount'] = np.round(hist_new['new_amount'] / (100 * hist_new_sorted[1:52623].delta.mean()), 2)


# In[ ]:


hist_new.new_amount.value_counts().head()


# In[ ]:


historical_transactions = hist_new[:29112361]
new_merchant_transactions = hist_new[29112361:]


# In[ ]:


historical_transactions.shape, new_merchant_transactions.shape


# # References 
# > https://www.kaggle.com/raddar/towards-de-anonymizing-the-data-some-insights
# 
# > https://www.kaggle.com/sudalairajkumar/simple-exploration-notebook-elo
# 
# > https://www.kaggle.com/cpmpml/raddar-magic-explained-a-bit/

# Thank you for reading. If you liked my kernel or was helpful to you kindly upvote it. Thanks!
