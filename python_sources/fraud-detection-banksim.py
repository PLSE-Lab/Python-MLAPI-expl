#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import average_precision_score
from xgboost import XGBClassifier, plot_importance
from sklearn.model_selection import train_test_split
import pickle

get_ipython().run_line_magic('matplotlib', 'inline')

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


# In[ ]:


df = pd.read_csv('/kaggle/input/banksim1/bs140513_032310.csv')


# In[ ]:


df['step'] = 1577836800 + df['step'] * 3600 * 24
df['step'] = pd.to_datetime(df['step'], unit='s')


# In[ ]:


df.columns


# In[ ]:


df


# In[ ]:


print('No. of fraud transactions: {}, No. of non-fraud transactions: {}'.format((df.fraud == 1).sum(), (df.fraud == 0).sum()))


# In[ ]:


df.loc[df.fraud == 1,'age'].value_counts()


# In[ ]:


df.loc[df.fraud == 1,'gender'].value_counts()


# In[ ]:


df.loc[df.fraud == 1,'category'].value_counts()


# In[ ]:


print('Min, Max amount of fraud transactions {}, {}'.format(df.loc[df.fraud == 1].amount.min(), df.loc[df.fraud == 1].amount.max()))
print('Min, Max amount of genuine transactions {}, {}'.format(df.loc[df.fraud == 0].amount.min(), df.loc[df.fraud == 0].amount.max()))


# In[ ]:


def last1DayTransactionCount(x):
    temp = pd.Series(x.index, index = x.step, name='count_1_day').sort_index()
    count_1_day = temp.rolling('1d').count() - 1
    count_1_day.index = temp.values
    x['count_1_day'] = count_1_day.reindex(x.index)
    return x
def last7DaysTransactionCount(x):
    temp = pd.Series(x.index, index = x.step, name='count_7_days').sort_index()
    count_7_days = temp.rolling('7d').count() - 1
    count_7_days.index = temp.values
    x['count_7_days'] = count_7_days.reindex(x.index)
    return x
def last30DaysTransactionCount(x):
    temp = pd.Series(x.index, index = x.step, name='count_30_days').sort_index()
    count_30_days = temp.rolling('30d').count() - 1
    count_30_days.index = temp.values
    x['count_30_days'] = count_30_days.reindex(x.index)
    return x


# In[ ]:


df = df.groupby('customer').apply(last1DayTransactionCount)
df = df.groupby('customer').apply(last7DaysTransactionCount)
df = df.groupby('customer').apply(last30DaysTransactionCount)


# In[ ]:


def last1DayCustMerchTransactionCount(x):
    temp = pd.Series(x.index, index = x.step, name='count_1_day').sort_index()
    count_1_day = temp.rolling('1d').count() - 1
    count_1_day.index = temp.values
    x['count_cust_merch_1_day'] = count_1_day.reindex(x.index)
    return x
def last7DaysCustMerchTransactionCount(x):
    temp = pd.Series(x.index, index = x.step, name='count_7_days').sort_index()
    count_7_days = temp.rolling('7d').count() - 1
    count_7_days.index = temp.values
    x['count_cust_merch_7_days'] = count_7_days.reindex(x.index)
    return x
def last30DaysCustMerchTransactionCount(x):
    temp = pd.Series(x.index, index = x.step, name='count_30_days').sort_index()
    count_30_days = temp.rolling('30d').count() - 1
    count_30_days.index = temp.values
    x['count_cust_merch_30_days'] = count_30_days.reindex(x.index)
    return x


# In[ ]:


df = df.groupby(['customer','merchant']).apply(last1DayCustMerchTransactionCount)
df = df.groupby(['customer','merchant']).apply(last7DaysCustMerchTransactionCount)
df = df.groupby(['customer','merchant']).apply(last30DaysCustMerchTransactionCount)


# In[ ]:


df


# In[ ]:


cust = df.groupby('customer').apply(lambda df:len(df))
sns.distplot(cust.values,kde=False)


# In[ ]:


df.count_1_day[df.fraud == 1].value_counts() / len(df.count_1_day[df.fraud == 1])


# In[ ]:


df.count_1_day[df.fraud == 0].value_counts() / len(df.count_1_day[df.fraud == 0])


# # Model Training

# In[ ]:


data = df.drop(['customer','merchant','zipcodeOri','zipMerchant','step'],axis=1)


# In[ ]:


cat_cols = ['age', 'gender', 'category']
enc = LabelEncoder()
for col in cat_cols:
    data[col] = enc.fit_transform(data[col])


# In[ ]:


data


# ## Standard Data without Feature Engineering

# In[ ]:


Y = data['fraud']
X = data.drop(['fraud','count_1_day','count_7_days','count_30_days','count_cust_merch_1_day','count_cust_merch_7_days','count_cust_merch_30_days'],axis=1)

Xtrain, Xtest, Ytrain, Ytest = train_test_split(X,Y,test_size=0.2, random_state = 1)

weights = (Y == 0).sum() / (Y == 1).sum()
clf = XGBClassifier(max_depth=3,scale_pos_weights=weights,n_jobs=4)
clf.fit(Xtrain,Ytrain)

print('AUPRC = {}'.format(average_precision_score(Ytest, clf.predict_proba(Xtest)[:,1])))


# ## Feature Engineered Data

# In[ ]:


Y = data['fraud']
X = data.drop('fraud',axis=1)

Xtrain, Xtest, Ytrain, Ytest = train_test_split(X,Y,test_size=0.2, random_state = 1)

weights = (Y == 0).sum() / (Y == 1).sum()
clf = XGBClassifier(max_depth=3,scale_pos_weights=weights,n_jobs=4)
clf.fit(Xtrain,Ytrain)

print('AUPRC = {}'.format(average_precision_score(Ytest, clf.predict_proba(Xtest)[:,1])))


# In[ ]:


fig = plt.figure(figsize = (14, 9))
ax = fig.add_subplot(111)

colours = plt.cm.Set1(np.linspace(0, 1, 9))

ax = plot_importance(clf, height = 1, color = colours, grid = False,                      show_values = False, importance_type = 'cover', ax = ax);
for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(2)
        
ax.set_xlabel('importance score', size = 16);
ax.set_ylabel('features', size = 16);
ax.set_yticklabels(ax.get_yticklabels(), size = 12);
ax.set_title('Ordering of features by importance to the model learnt', size = 20);


# In[ ]:


data


# In[ ]:


pickle.dump(clf,open('XGB-BankSim.pkl','wb'))


# In[ ]:


clf.score(Xtest.loc[Ytest==1],Ytest.loc[Ytest==1])


# In[ ]:




