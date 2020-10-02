#!/usr/bin/env python
# coding: utf-8

# Let's get done with the shenanigans 

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

        
pd.set_option('display.max_columns',5000)
pd.set_option('display.max_rows',100)
pd.set_option('display.width',10000)
# Any results you write to the current directory are saved as output.


# Load the files 

# In[ ]:


import pandas as pd
sample_submission = pd.read_csv("../input/ieee-fraud-detection/sample_submission.csv")
test_identity = pd.read_csv("../input/ieee-fraud-detection/test_identity.csv")
test_transaction = pd.read_csv("../input/ieee-fraud-detection/test_transaction.csv")
train_identity = pd.read_csv("../input/ieee-fraud-detection/train_identity.csv")
train_transaction = pd.read_csv("../input/ieee-fraud-detection/train_transaction.csv")


# Let's start looking at the data 

# In[ ]:


test_identity.head(5)


# In[ ]:


train_identity.describe()


# In[ ]:


#lets look at nulls

train_identity.isnull().any()

train_identity.isnull().sum()


# In[ ]:


train_transaction.head(5)


# In[ ]:


train_transaction.shape


# #Before we move any further, let's make a basic baseline model and try to see how it performs 

# We are going to join the two dataframes to give us one result set

# In[ ]:


merge_df = pd.merge(left=train_transaction,right=train_identity,how="left",on="TransactionID")


# In[ ]:


merge_df_test = pd.merge(left=test_transaction,right=test_identity,how="left",on="TransactionID")


# Setting indexes which would be used later to submit the prediction 

# In[ ]:


merge_df.set_index(keys='TransactionID',inplace=True)


# In[ ]:


merge_df_test.set_index(keys='TransactionID',inplace=True)


# In[ ]:


#merge_df.head(5)
merge_df_test.head(5)


# In[ ]:


# check how many fraud transactions are there in the dataset

import seaborn as sns

sns.countplot(x='isFraud',data=merge_df)


# Checking what percentage of data is actually fraud

# In[ ]:


# so less then 4 perecent of the transaction is fraud 
## this would be interesting as class distribution has a lot of difference
round((len(merge_df[merge_df['isFraud']==1])/len(merge_df))*100,2)


# In[ ]:


##intutive column selection for now 
merge_df.columns


# In[ ]:


cols = ['TransactionID', 'isFraud', 'TransactionDT', 'TransactionAmt', 'ProductCD', 'card1', 'card2', 'card3', 'card4', 'card5',
         'card6','P_emaildomain','C1','C2','C3','C4','C5','C6','C7','C8','C9','C10','C11','C12','C13','C14','DeviceType', 'DeviceInfo']


# In[ ]:


merge_df['TransactionAmt'].isnull().any()


# In[ ]:


merge_df_test['TransactionAmt'].isnull().any()


# In[ ]:


# lets draw the histogram of transaction amount

from matplotlib import pyplot as plt
import seaborn as sns

#plt.hist(x=merge_df['TransactionAmt'])

#plt.scatter(x=merge_df['TransactionAmt'])

merge_df['TransactionAmt']

sns.boxplot(merge_df['TransactionAmt'])


# In[ ]:


from scipy.stats import iqr

iqr(merge_df['TransactionAmt'])


# In[ ]:


merge_df['ProductCD'].isnull().any()


# In[ ]:


merge_df_test['ProductCD'].isnull().any()


# In[ ]:


# let's look at the unique values

merge_df['ProductCD'].value_counts()


# In[ ]:


# plot to see if any pattern in product type and fraud 

sns.countplot(x='ProductCD',hue='isFraud',data=merge_df)

#doesn't seem like there is a pattern here 


# In[ ]:


merge_df[['card1','card2','card3','card4','card5','card6']].isnull().any()

#merge_df['card2'].value_counts()


# around 1.5% percent values are 


# In[ ]:


merge_df_test[['card1','card2','card3','card4','card5','card6']].isnull().any()


# Assigning these null values the most frequently occuring value in the system

# In[ ]:



merge_df.loc[merge_df['card2'].isna(),'card2']=321


# In[ ]:



merge_df_test.loc[merge_df_test['card2'].isna(),'card2']=321


# In[ ]:


merge_df.loc[merge_df['card3'].isna(),'card3']=150


# In[ ]:


merge_df_test.loc[merge_df_test['card3'].isna(),'card3']=150


# In[ ]:


merge_df.loc[merge_df['card4'].isna(),'card4']='visa' 


# In[ ]:


merge_df_test.loc[merge_df_test['card4'].isna(),'card4']='visa' 


# In[ ]:


merge_df.loc[merge_df['card5'].isna(),'card5']=226


# In[ ]:


merge_df_test.loc[merge_df_test['card5'].isna(),'card5']=226


# In[ ]:


merge_df.loc[merge_df['card6'].isna(),'card6']='debit'


# In[ ]:


merge_df_test.loc[merge_df_test['card6'].isna(),'card6']='debit'


# In[ ]:


merge_df[['card1','card2','card3','card4','card5','card6']].isnull().any()


# In[ ]:


merge_df_test[['card1','card2','card3','card4','card5','card6']].isnull().any()


# In[ ]:


cols = ['TransactionAmt','ProductCD','card1', 'card2', 'card3', 'card4', 'card5',
          'card6','isFraud']


# In[ ]:


# Lets get the subset of the columns

merge_df = merge_df[cols]


# In[ ]:


# Lets get the subset of the columns
cols_test = ['TransactionAmt','ProductCD','card1', 'card2', 'card3', 'card4', 'card5',
          'card6']
merge_df_test = merge_df_test[cols_test]


# Let's look at the correlation heat map

# In[ ]:


#print(merge_df)
cor = merge_df.corr()
#print(cor)
sns.heatmap(cor)


# In[ ]:


merge_df


# Convert stirngs to numerical features

# In[ ]:


from sklearn.preprocessing import LabelEncoder

le1 = LabelEncoder().fit(merge_df['ProductCD'])
merge_df['ProductCD'] = le1.transform(merge_df['ProductCD'])
merge_df_test['ProductCD'] = le1.transform(merge_df_test['ProductCD'])

le2 = LabelEncoder().fit(merge_df['card4'])
merge_df['card4'] = le2.transform(merge_df['card4'])
merge_df_test['card4'] = le2.transform(merge_df_test['card4'])

le3 = LabelEncoder().fit(merge_df['card6'])
merge_df['card6'] = le3.transform(merge_df['card6'])
merge_df_test['card6'] = le3.transform(merge_df_test['card6'])


# In[ ]:


print(merge_df)


# Create train test split

# In[ ]:


from sklearn.model_selection import train_test_split

X= merge_df.iloc[:,:-1]
y = merge_df.iloc[:,-1]
#print(y)

X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=0)


# Lets start by applying Random forest classifier , we would be using GridsearchCV to get the best model output 

# In[ ]:


# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import GridSearchCV

# rf = RandomForestClassifier()
# param = {'n_estimators':[150],'max_depth':[9]}

# grid_model = GridSearchCV(rf,param_grid=param,scoring='roc_auc')

# grid_model.fit(X,y)

# print(grid_model.best_estimator_)
# print(grid_model.best_score_)
# print(grid_model.best_params_)


# we are going to apply GBC next , the paramter values are select by running them through GridSearch for simplicity only placing the apply code here

# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier

gf = GradientBoostingClassifier(learning_rate=0.1,n_estimators=150,max_depth=9).fit(X_train,y_train)

y_pred = gf.predict(X_test)


# In[ ]:


print(gf.score(X_test,y_test))


# In[ ]:


## Let's predict using GBf

merge_df_test


# In[ ]:


# so we are getting a score of 86%
from sklearn.metrics import roc_auc_score
y_score = gf.predict_proba(X_test)[:,1]
print(roc_auc_score(y_test,y_score))


# All done let's get the score now 

# In[ ]:


y_test_predict = gf.predict_proba(merge_df_test)[:,1]


# In[ ]:


result_df = pd.DataFrame()
result_df['TransactionID'] = merge_df_test.index
result_df['isFraud'] = y_test_predict


# In[ ]:


result_df.to_csv('result1.csv',index=False)


# In[ ]:


len(result_df[result_df['isFraud']>0.5])/len(result_df)*100


# Let's try appling Neural Network to the problem on hand

# In[ ]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler().fit(X_train)
X_train_scaled =scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
merge_df_test_scaled = scaler.transform(merge_df_test)


# In[ ]:


# from sklearn.neural_network import MLPClassifier

# mlp = MLPClassifier(hidden_layer_sizes=(13,13,13),max_iter=500)
# mlp.fit(X_train_scaled,y_train)


# In[ ]:


# y_test_predict_nlp


# In[ ]:


# y_test_predict_nlp = mlp.predict_proba(merge_df_test_scaled)[:,1]
# result_df = pd.DataFrame()
# result_df['TransactionID'] = merge_df_test.index
# result_df['isFraud'] = y_test_predict_nlp

# result_df.to_csv('result2.csv',index=False)


# Lets try to apply XGBoost and see if results are better

# In[ ]:


import xgboost as xgb

model = xgb.XGBClassifier(n_estimators=500,
                        n_jobs=4,
                        max_depth=9,
                        learning_rate=0.05,
                        subsample=0.9,
                        colsample_bytree=0.9)

model.fit(X,y)



# In[ ]:


temp =model.predict_proba(merge_df_test)[:,1]
result_df3 = pd.DataFrame()
result_df3['TransactionID'] = merge_df_test.index
result_df3['isFraud'] = temp
result_df3.to_csv('result5.csv',index=False)

