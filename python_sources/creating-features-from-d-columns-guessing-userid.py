#!/usr/bin/env python
# coding: utf-8

# ## Create features from D columns
# 
# Hi guys, this is my first ever kernel on Kaggle, hope it helps you!
# 

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


#import data
train_transaction = pd.read_csv('../input/ieee-fraud-detection/train_transaction.csv')
test_transaction = pd.read_csv('../input/ieee-fraud-detection/test_transaction.csv')


# In[ ]:


import matplotlib.pyplot as plt


# In[ ]:


SEED = 12345

#creating Day feature
train_transaction['TransactionDTday'] = (train_transaction['TransactionDT']/(60*60*24)).map(int)
test_transaction['TransactionDTday'] = (test_transaction['TransactionDT']/(60*60*24)).map(int)

#stolen from alijs and slightly modified: https://www.kaggle.com/alijs1/ieee-transaction-columns-reference
def timehist1_2(col,product):
    N = 8000 if col in ['TransactionAmt'] else 9999999999999999 # clip trans amount for better view
    train_transaction[(train_transaction['isFraud'] == 0) & (train_transaction['ProductCD'] == product)].set_index('TransactionDT')[col].clip(0, N).plot(style='.', title='Hist ' + col, figsize=(15, 3))
    train_transaction[(train_transaction['isFraud'] == 1) & (train_transaction['ProductCD'] == product)].set_index('TransactionDT')[col].clip(0, N).plot(style='.', title='Hist ' + col, figsize=(15, 3))
    test_transaction[test_transaction['ProductCD'] == product].set_index('TransactionDT')[col].clip(0, N).plot(style='.', title=col + ' values over time (blue=no-fraud, orange=fraud, green=test)', figsize=(15, 3))
    plt.show()


# Let's take the example of **D1** (but this approach works with other D features).
# 
# In the following cell, I plot the D1 feature by product. <br>
# If we take look at the orange dots (is Fraud=1), we can clearly see that some are forming lines.

# In[ ]:


products=train_transaction.ProductCD.unique().tolist()
col='D1'
for prod in products: 
    print("Product code:", prod)
    timehist1_2(col, prod)


# Again, to make things easier, we'll focus on a subset of the data. The S product category fits the needs of my demonstration.
# 
# D1 feature is a Datetime since a particular day, and increases therefore by day. Let's reverse this growth with the following variable and plot the result.

# In[ ]:


train_transaction['D1minusday'] = train_transaction['D1'] - train_transaction['TransactionDTday']
test_transaction['D1minusday'] = test_transaction['D1'] - test_transaction['TransactionDTday']


# In[ ]:


col='D1minusday'
for prod in ['S']: 
    print("Product code:", prod)
    timehist1_2(col, prod)


# The orange increasing line is now horizontal, we have corrected it from the slope, let's now find the intercept.

# In[ ]:


train_transaction[(train_transaction.isFraud==1) & (train_transaction.ProductCD=='S') & (train_transaction.D1minusday>50)]['D1minusday'].value_counts()


# The intercept of this line is 78. <br>
# Does it mean that 78 days before the start date, more cards were used than the other days, and more Frauds appear on this day (proportions being equals) (Frauds are plotted on top of non Fraud), or did we identity a fraudulent user? Let's check

# In[ ]:


import seaborn as sns
print('Blue: Frauds, Orange: Non-Fraud')
sns.distplot(train_transaction[(train_transaction.isFraud==1) & (train_transaction.ProductCD=='S')]['D1minusday'], hist=False, rug=False);
sns.distplot(train_transaction[(train_transaction.isFraud==0) & (train_transaction.ProductCD=='S')]['D1minusday'], hist=False, rug=False);


# It seems that the spike around day 78, corresponds to relatively more fraudulent cases!
# 
# I would not recommend to use this Feature for training, as it becomes a proxy of the Transaction day. Also, the original D1 feature is cliped, and many groups would not appear in the testing set, overfitting risk is increased (especially if you are using Kfolds as the future can easily predicts the past with this variable).
# 
# 
# **But** i would use it as a grouping feature (in our example: day 78 can help to identify a specific card). For instance, two cards with the same numbers (card1 to card6) that correspond to different cards can be distinguished thanks to this variable.

# In[ ]:


train_transaction[(train_transaction.isFraud==1) & (train_transaction.D1minusday==78)][['card1','card2','card3','card4','card5','card6','addr1',
 'addr2',
 'dist1',
 'dist2','P_emaildomain','R_emaildomain','TransactionDTday']]


# Also, we can derive other informations from this variable. For example about the stability of the emails, distances, addresses etc, to generate even more features
# 
# <img src="https://i.imgflip.com/3b4qm5.jpg">

# **I hope that you found this kernel useful. And by the way, i am looking for a team, I joined the competition lately and I need someone with a great knowledge of this competition**
