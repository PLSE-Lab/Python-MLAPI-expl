#!/usr/bin/env python
# coding: utf-8

# Competition is over and first of all congrats to all participants! 
# Big shout out to winners! Well deserved!!! :) 
# 
# In this kernel, I will try to explain one feature that gave me a great boost in this competition.
# 
# In my [previous kernel](https://www.kaggle.com/akasyanama13/eda-what-s-behind-d-features) I made an assumption regarding D features. It was a small hint for so-called 'magic' :)
# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import datetime
from dateutil.relativedelta import relativedelta
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


train = pd.read_csv('../input/ieee-fraud-detection/train_transaction.csv')


# The main idea was that **D1 represents the number of days from the first transaction** 
# and **D3 indicates the number of dates from the previous transaction**, also we know that 
# **TransactionDT is seconds from some starting point**. 
# Taking '2017-11-30' as a starting point we can calculate the first transaction date.

# In[ ]:


start_point=datetime.datetime.strptime('2017-11-30', '%Y-%m-%d')

train['D1'] = np.where(np.isnan(train['D1']), 0, train['D1'])
train['D3'] = np.where(np.isnan(train['D3']), 0, train['D3'])

# TransactionDT is a number of sec from some starting point, 2017-11-30 in our case.
train['TransactionDays'] = np.round(train['TransactionDT']/(60*60*24), 0)
train['TransactionDate'] = [start_point + relativedelta(seconds=secs) for secs in train['TransactionDT']]

# The first transaction date is nothing but difference between TransactionDate and D1
train['FirstTransaction'] = [dt1-relativedelta(days=days) for dt1, days in zip(train['TransactionDate'], train['D1'])]
train['FirstTransaction'] = [datetime.datetime.strftime(d, '%Y-%m-%d') for d in train['FirstTransaction']]


# In[ ]:


train.head()


# Now we can use **FirstTransaction** column as an addition to **card_x** features to identify **UserID**. But this is not the end. Some V_x features are also can be used for grouping. 
# I tried to add them successively and checked if on UserID level we calculate 'days from the previous transaction' and it matches to D3.

# In[ ]:


'''
This function creates UserID based on some pull of columns 
and then calculates the number of days from the previous transaction on UserID level
'''
def get_user_id(df, by, start_point=datetime.datetime.strptime('2017-11-30', '%Y-%m-%d')):

    df['TransactionDays'] = np.round(df['TransactionDT']/(60*60*24), 0)
    df['TransactionDate'] = [start_point + relativedelta(seconds=secs) for secs in df['TransactionDT']]
    df['D1'] = np.where(np.isnan(df['D1']), 0, df['D1'])

    df['FirstTransaction'] = [dt1-relativedelta(days=days) for dt1, days in zip(df['TransactionDate'], df['D1'])]
    df['FirstTransaction'] = [datetime.datetime.strftime(d, '%Y-%m-%d') for d in df['FirstTransaction']]
    df['D3'] = np.where(np.isnan(df['D3']), 0, df['D3'])

    df[by] = df[by].fillna(-99)

    grouped = df.groupby(by, as_index=False)['TransactionID'].min()
    grouped = grouped.rename(columns={'TransactionID': 'UserID'})

    df = pd.merge(df, grouped, on=by, how='left')

    df = df.sort_values(['TransactionDays'], ascending=True).groupby(['UserID']).head(df.shape[0])
    df['diffs'] = df.sort_values(['TransactionDT'], ascending=True).groupby(['UserID'])['TransactionDT'].transform(lambda x: x.diff())

    df['firstInRow'] = np.where(np.isnan(df['diffs']), 1, 0)
    df['diffs'] = np.where(np.isnan(df['diffs']), 0, df['diffs'])
    df['diffs'] = np.round(df['diffs'] / (60 * 60 * 24), 0)

    return df


# In[ ]:


# Let's check how it works. The example below creates UserID based on card_x, addr1, and FirstTransaction features.

by = ['card2', 'card3', 'card4', 'card5', 'card6', 'addr1', 'FirstTransaction']
train_u = get_user_id(train, by, start_point=datetime.datetime.strptime('2017-11-30', '%Y-%m-%d'))


# In[ ]:


# Now we can calculate how accurate this UserID
correctDiffD3 = train_u[(train_u['diffs'] == train_u['D3']) & (train_u['firstInRow'] == 0)].shape[0] / train_u[(train_u['firstInRow'] == 0)].shape[0]
correctDiffD3


# 88% - not bad! Let's try to add V1 into UserID

# In[ ]:


by = ['card2', 'card3', 'card4', 'card5', 'card6', 'addr1', 'FirstTransaction', 'V1']
train_u = get_user_id(train, by, start_point=datetime.datetime.strptime('2017-11-30', '%Y-%m-%d'))

correctDiffD3 = train_u[(train_u['diffs'] == train_u['D3']) & (train_u['firstInRow'] == 0)].shape[0] / train_u[(train_u['firstInRow'] == 0)].shape[0]
correctDiffD3


# V1 makes UserID worse...Let's try to add V29.

# In[ ]:


by = ['card2', 'card3', 'card4', 'card5', 'card6', 'addr1', 'FirstTransaction', 'V29']
train_u = get_user_id(train, by, start_point=datetime.datetime.strptime('2017-11-30', '%Y-%m-%d'))

correctDiffD3 = train_u[(train_u['diffs'] == train_u['D3']) & (train_u['firstInRow'] == 0)].shape[0] / train_u[(train_u['firstInRow'] == 0)].shape[0]
correctDiffD3


# V29 works! Following this logic I ended up with the next features for UserID:
# 
# ['card1', 'card2', 'card3', 'card4', 'card5', 'card6', 'addr1', 'FirstTransaction',
#  'V29', 'V53', 'V55', 'V57', 'V59', 'V61', 'V63', 'V69', 'V138', 'V139', 'V140', 'V141', 'V142',
#  'V144', 'V145', 'V146', 'V147', 'V148', 'V150', 'V151', 'V152', 'V157', 'V159', 'V160',
#  'V161', 'V162', 'V163', 'V164', 'V165', 'V166', 'V305', 'V311', 'V322', 'V323',
#  'V324', 'V325', 'V326', 'V327', 'V328', 'V329', 'V330',
#  'V331', 'V332', 'V333', 'V334', 'V335', 'V336', 'V337', 'V338', 'V339']
#  
# Adding UserID as categorical feature into catboost helped me a lot. Single catboost could give me 0.9566 on public leaderboard.

# I ended up in 42nd place...I know that it was tiring competition for everyone, with deadline extension and lots of arguing regarding disclosing magic but anyway thanks organizers for their continuous answers and clarifications on the forum.
# 
# The full solution can be found [here](https://github.com/Aziko13/IEEE-CIS-Fraud-Detection). Hope you find it useful.
