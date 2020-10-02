#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score

from xgboost import XGBClassifier
from xgboost import plot_importance
from sklearn.preprocessing import LabelEncoder


# In[ ]:


import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


# # Exploratory Data Analysis

# In[ ]:


df = pd.read_csv('/kaggle/input/paysim1/PS_20174392719_1491204439457_log.csv')
df = df.rename(columns={'oldbalanceOrg':'oldBalanceOrig', 'newbalanceOrig':'newBalanceOrig',                         'oldbalanceDest':'oldBalanceDest', 'newbalanceDest':'newBalanceDest'})
print(df.head())


# In[ ]:


df = df.drop(['isFlaggedFraud'],axis=1)
df.columns


# In[ ]:


print("No. of fraud transactions: {}, No. of non-fraud transactions: {}".format((df.isFraud == 1).sum(),(df.isFraud == 0).sum()))


# In[ ]:


dfFraud = df.loc[df.isFraud == 1]
dfNonFraud = df.loc[df.isFraud == 0]


# In[ ]:


print("What are the types for fraud transactions? {}".format(dfFraud.type.drop_duplicates().values))
print("\nHow many unique origins are there in fraud transfer transactions? {} / {}".format(len(dfFraud.loc[dfFraud.type == 'TRANSFER'].nameOrig.drop_duplicates().values),len(dfFraud.loc[dfFraud.type == 'TRANSFER'])))
print("How many unique destinations are there in fraud transfer transactions? {} / {}".format(len(dfFraud.loc[dfFraud.type == 'TRANSFER'].nameDest.drop_duplicates().values),len(dfFraud.loc[dfFraud.type == 'TRANSFER'])))
print("\nHow many unique origins are there in fraud cash out transactions? {} / {}".format(len(dfFraud.loc[dfFraud.type == 'CASH_OUT'].nameOrig.drop_duplicates().values),len(dfFraud.loc[dfFraud.type == 'CASH_OUT'])))
print("How many unique destinations are there in fraud cash out transactions? {} / {}".format(len(dfFraud.loc[dfFraud.type == 'CASH_OUT'].nameDest.drop_duplicates().values),len(dfFraud.loc[dfFraud.type == 'CASH_OUT'])))


# In[ ]:


dfFraudTransfer = dfFraud.loc[dfFraud.type == 'TRANSFER']
dfFraudCashout = dfFraud.loc[dfFraud.type == 'CASH_OUT']

print("How many fraud transfer transactions have destinations which are origins in fraud cash out transactions? {}".     format(dfFraudTransfer.nameDest.isin(dfFraudCashout.nameOrig.unique()).sum()))

print("\nHow many fraud transfer transactions have destinations which are origins in genuine cash out transactions? {}".     format(dfFraudTransfer.nameDest.isin(dfNonFraud.loc[dfNonFraud.type == 'CASH_OUT'].nameOrig.unique()).sum()))
print("How many genuine transfer transactions have destinations which are origins in fraud cash out transactions? {}".     format(dfNonFraud.loc[dfNonFraud.type == 'TRANSFER'].nameDest.isin(dfFraudCashout.nameOrig.unique()).sum()))

print("\nHow many genuine transfer transactions have destinations which are destinations in fraud transfer transactions? {}".     format(dfNonFraud.loc[dfNonFraud.type == 'TRANSFER'].nameDest.isin(dfFraudTransfer.nameDest.unique()).sum()))
print("How many genuine transfer transactions have origins which are destinations in fraud transfer transactions? {}".     format(dfNonFraud.loc[dfNonFraud.type == 'TRANSFER'].nameOrig.isin(dfFraudTransfer.nameDest.unique()).sum()))
print("How many genuine transfer transactions have origins which are origins in fraud transfer transactions? {}".     format(dfNonFraud.loc[dfNonFraud.type == 'TRANSFER'].nameOrig.isin(dfFraudTransfer.nameOrig.unique()).sum()))
print("How many genuine transfer transactions have destinations which are origins in fraud transfer transactions? {}".     format(dfNonFraud.loc[dfNonFraud.type == 'TRANSFER'].nameDest.isin(dfFraudTransfer.nameOrig.unique()).sum()))


# In[ ]:


print('Min, Max of Fraud Transactions: {} - {}'.format(dfFraud.amount.min(),dfFraud.amount.max()))
print('Min, Max of Non-Fraud Transactions: {} - {}'.format(dfNonFraud.amount.min(),dfNonFraud.amount.max()))


# # Data Cleaning

# In[ ]:


df.isnull().values.any()


# In[ ]:


print('% of fraud transactions in which \'oldBalanceDest\' \'newBalanceDest\' and amount is non-zero: {}'.format(len(dfFraud.loc[(dfFraud.oldBalanceDest == 0) & (dfFraud.newBalanceDest == 0) & (dfFraud.amount != 0) & (dfFraud.type.isin(['TRANSFER','CASH_OUT']))]) / len(dfFraud.loc[(dfFraud.type.isin(['TRANSFER','CASH_OUT']))])))
print('% of genuine transactions in which \'oldBalanceDest\' \'newBalanceDest\' and amount is non-zero: {}'.format(len(dfNonFraud.loc[(dfNonFraud.oldBalanceDest == 0) & (dfNonFraud.newBalanceDest == 0) & (dfNonFraud.amount != 0) & (dfNonFraud.type.isin(['TRANSFER','CASH_OUT']))]) / len(dfNonFraud.loc[(dfNonFraud.type.isin(['TRANSFER','CASH_OUT']))])))


# In[ ]:


df['errorBalanceOrig'] = df.newBalanceOrig + df.amount - df.oldBalanceOrig
df['errorBalanceDest'] = df.oldBalanceDest + df.amount - df.newBalanceDest


# In[ ]:


print('Percentage of transactions with non-zero \'errorBalanceOrig\'')
pd.DataFrame(df.groupby('type').apply(lambda df:len(df.loc[df.errorBalanceOrig != 0]) / len(df)))


# In[ ]:


print('Percentage of transactions with non-zero \'errorBalanceDest\'')
pd.DataFrame(df.groupby('type').apply(lambda df:len(df.loc[df.errorBalanceDest != 0]) / len(df)))


# In[ ]:


df.groupby('type').apply(lambda df:df.loc[df.errorBalanceDest != 0])


# In[ ]:


df = df.drop(['nameOrig','nameDest'], axis=1)


# In[ ]:


enc = LabelEncoder()
df['type'] = enc.fit_transform(df['type'])


# In[ ]:


df.dtypes


# # Model Training

# In[ ]:


Y = df.isFraud
X = df.drop(['isFraud'],axis=1)


# In[ ]:


Xtrain, Xtest, Ytrain, Ytest = train_test_split(X,Y,test_size=0.2, random_state = 1)


# In[ ]:


# scale_pos_weight should be ratio of negative classes to positive classes
weights = (Y == 0).sum() / (Y == 1).sum()
clf = XGBClassifier(max_depth = 3, scale_pos_weight = weights, n_jobs = 4)

clf.fit(Xtrain,Ytrain)


# In[ ]:


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




