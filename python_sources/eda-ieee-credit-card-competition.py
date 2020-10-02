#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns #visualization
import re #regular expressions, will be used when dealing with id_30 and id_31
import matplotlib.pyplot as plt #visualization
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.preprocessing import LabelEncoder #encoding categorical features
from category_encoders import target_encoder #We'll use Target Encoder for the emails
from sklearn.preprocessing import StandardScaler #PCA, dimensionality reducion
from sklearn.decomposition import PCA
from sklearn.experimental import enable_iterative_imputer #NaN imputation
from sklearn.impute import IterativeImputer #NaN imputation
from sklearn.impute import KNNImputer #NaN imputation

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


ss = pd.read_csv('/kaggle/input/ieee-fraud-detection/sample_submission.csv')
train_t = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_transaction.csv')
test_t = pd.read_csv('/kaggle/input/ieee-fraud-detection/test_transaction.csv')
train_i = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_identity.csv')
test_i = pd.read_csv('/kaggle/input/ieee-fraud-detection/test_identity.csv')
train = pd.merge(train_t, train_i, how = 'inner', on = 'TransactionID')
test = pd.merge(test_t, test_i, how = 'inner', on = 'TransactionID')


# ## Strategy of work ##
# 
# * *We have a credit card fraud detection problem. What does it mean? That this is a classification problem that is highly class-imbalanced, class-overlapping and in which the relationship between features and target is highly non-linear.*
# 
# 
# Our strategy will be the following:
# 
# 
# -> We need to create some **features**. Some of them are based in the basic paradigm of **RFM** (Recency, Frequency and Monetary Value), others are some useful features that are often used in the **literature** (type of currency, division between dollars and cents, days of week, hours of day) and the last group is based on **network** between (fraudulent) customers, merchants and transactions. This last approach is based on the APATE method, which outperformed the classical approach. Inspired by APATE, we'll divide all of our features into SHORT-TERM (ST, which will be measured in minutes within a day), MID-TERM (MT, which will be measured in hours within a week) and LONG-TERM (LT, which will be measured in days within a month). Maybe (with a very high probability) these features were already created by Vesta, but we need to check. After creating all the features, we'll drop the V-features that have a strong correlation (>0.95) with the created features.
# 
# -> There are different kinds of products (ProductCD) with different behaviours. We'll divide the dataset into 5 new datasets,one for each ProductCD.
# 
# -> After creating all features, we'll apply random forests as the main algorithm. Variations of other algorithms based on decision trees will be used if and only if they increase the score of random forests. Instead of using Information Gain to do the split, we'll use Hellinger Distance Decision Trees, which is more accurate for imbalanced datasets. To train the model, we'll use the EasyEnsemble algorithm to make the train set more balanced.
# 
# -> Since this is a time-series problem, our validation set will be the last month of the training data.
# 
# -> To generate the new features, we'll have to dig deep into un-anonymizing the features, because we'll need, at minimum, an unique ID for each customer (credit card) and an unique ID for each merchant.
# 
# -> Time is crucial to generate all of the features, so we'll analyze the D-j features carefully, among with the TransactionDT feature.
# 
# -> We'll start everything with EDA. In the EDA, we'll check the following: a) Target Distribution, b) Date Columns, c) C columns, d) Unique ID for customer generation, e) Unique ID for merchant generation, f) Drop Constant Columns
# 
# 
# 

# ## EDA ##

# In[ ]:


sns.distplot(train['isFraud'], kde = False)


# In[ ]:


vesta_features = []
for i in range(1, 340):
    vesta_features.append('V' + str(i))


# In[ ]:


##Let us drop all constant Vesta Features
dropped = []
for v_feature in vesta_features:
    if train[v_feature].nunique() <= 1:
        vesta_features.remove(v_feature)
        train.drop(v_feature, axis = 1, inplace = True)
        test.drop(v_feature, axis = 1, inplace = True)
        dropped.append(v_feature)
dropped


# ## Analysis of the Date features ##

# In[ ]:


date_features = ['TransactionDT']
for i in range(1, 16):
    date_features.append('D' + str(i))
train[date_features]


# In[ ]:


test[date_features]


# In[ ]:


for df in train, test:
    df['days'] = df['TransactionDT']//(86400)
    df['weeks'] = df['TransactionDT']//(7*86400) #Very long term 
    df['days_month'] = (((df['TransactionDT']//86400))%30) #Long term
    df['hours_day'] = (df['TransactionDT']%(3600*24)/3600//1) #Mid term
    df['minutes_hour'] = (df['TransactionDT']%(60*60)/60//1) #Short term
date_features.extend(['days_month','weeks','hours_day','minutes_hour', 'days'])
train[date_features]


# In[ ]:


train['D1'].value_counts()


# In[ ]:


train['D1'].nunique()


# In[ ]:


sns.distplot(train['D1'], kde = False)


# In[ ]:


train['D2'].value_counts()


# In[ ]:


sns.distplot(train['D2'], kde = False)


# In[ ]:


train['D3'].value_counts()


# In[ ]:


sns.distplot(train['D3'], kde = False)


# In[ ]:


train['D4'].value_counts()


# In[ ]:


sns.distplot(train['D4'], kde = False)


# In[ ]:


train['D5'].value_counts()


# In[ ]:


sns.distplot(train['D5'], kde = False)


# In[ ]:


train['D6'].value_counts()


# In[ ]:


sns.distplot(train['D6'], kde = False)


# In[ ]:


train['D7'].value_counts()


# In[ ]:


sns.distplot(train['D7'], kde = False)


# In[ ]:


train['D8'].value_counts()


# In[ ]:


sns.distplot(train['D8'], kde = False)


# In[ ]:


train['D9'].value_counts()


# In[ ]:


sns.distplot(train['D9'], kde = False)


# In[ ]:


train['D10'].value_counts()


# In[ ]:


sns.distplot(train['D10'], kde = False)


# In[ ]:


train['D11'].value_counts()


# In[ ]:


#We'll drop 'D11', since it is an empty feature
train.drop('D11', axis = 1, inplace = True)
test.drop('D11', axis = 1, inplace = True)
date_features.remove('D11')


# In[ ]:


train['D12'].value_counts()


# In[ ]:


sns.distplot(train['D12'], kde = False)


# In[ ]:


train['D13'].value_counts()


# In[ ]:


sns.distplot(train['D13'], kde = False)


# In[ ]:


train['D14'].value_counts()


# In[ ]:


sns.distplot(train['D14'], kde = False)


# In[ ]:


train['D15'].value_counts()


# In[ ]:


sns.distplot(train['D15'], kde = False)


# In[ ]:


sns.distplot(train['weeks'], kde = False)


# In[ ]:


sns.distplot(test['weeks'], kde = False)


# In[ ]:


sns.distplot(train['days_month'], kde = False)


# In[ ]:


train['hours_day'].value_counts()


# In[ ]:


sns.distplot(train['hours_day'], kde = False)


# In[ ]:


train['minutes_hour'].value_counts()


# In[ ]:


sns.distplot(train['minutes_hour'], kde = False)


# In[ ]:


#D9 and hours_day are pretty similar, aren't they ? Maybe the D9 feature is 'hours of day', but somehow normalized


# In[ ]:


train[['D9', 'hours_day']]


# In[ ]:


norm_const = train.loc[144228, 'D9']/train.loc[144228, 'hours_day']


# In[ ]:


train['D9'] = train['D9'].apply(lambda a: round(a/norm_const))
test['D9'] = test['D9'].apply(lambda a: round(a/norm_const))
train['D9'].value_counts() #Just as we expected


# In[ ]:


#Since the feature 'hours_day' doesn't have any Missing Values , we'll create a dummy variable 'D9_nan' that takes the value zero if 
#There is a NaN in D9 and one if D9 is not NaN. Then, we'll drop D9.
train['D9_nan'] = train['D9'].isnull().apply(lambda a: 1 if a == False else 0)
test['D9_nan'] = test['D9'].isnull().apply(lambda a: 1 if a == False else 0)
date_features.append('D9_nan')
train[['D9_nan', 'hours_day']]


# Except for hours_day, D9_nan, days_week and minutes_hour, all Dx distributions seem to be highly concentrated around zero. Since these features are timedeltas that relate to a specific transaction and their measure units are compatible with 'days' units, we will transform them by Dx = a * TransactionDT - Dx, where a is seconds -> days normalizing constant. 
# 
# 
# Furthermore, we see that D1, D2 and D12 features have maximum values around 600 and D3,D4,D5,D6,D7,D10,D12,D13,D14,D15 have maximum values around 800 and D8 have maximum values around 1750, so they must refer to similar things. For example, D1, D2 and D12 may relate to the time since the creation of the account,D3-D15 relate to the time since the last transaction (for an account, card, type of product etc). Let us see what our segmented analysis has to show to us:
# 
# #ProductCD == H
# * D1 and D2 have 99% positive correlation
# * D3 and D5 have 100% positive correlation
# * D3 and D7 have 100% positive correlation
# * D11 and D12 have 100% of missing values
# * D8 and D9 are in the same NaN group
# 
# #ProductCD == C
# * D4 and D6 have 99% positive correlation
# * D4 and D12 have 100% positive correlation
# * D4 and D6 have 100% positive correlation
# * D11 has 100% of missing values
# * D8 and D9 are in the same NaN group
# 
# #ProductCD == R
# * D1 and D2 have 100% positive correlation
# * D11 and D12 have 100% of missing values
# * D8 and D9 are in the same NaN group
# 
# #ProductCD == S (doesn't have the P_emaildomain column)
# * D1 doesn't have missing values
# * D1 and D2 have 100% positive correlation
# * D4, D5, D11 and D12 have 100% of missing values
# * D8 and D9 are in the same NaN group
# * D1 and D2 have 100% positive correlation
# 
# #ProductCD == W (doesn't have Identity Data nor R_emaildomain column)
# * D1 and D2 have 98% positive correlation
# * D6, D7, D8, D9, D11, D12, D13, D14 have 100% of missing values.
# 
# ###############################################################################
# 
# Hypothesis:
# 
# 1. D2 is a linear function of D1. Some of the D1 data has been replaced to NaN in D2, since the number of NaN in the former is greater than in the latter. If this is so, we'll replace the D2 column do D2_nan, where D2_nan is a function that takes value 1 if there is a non-nan missing value in columns D1 and D2 for the same row and zero otherwise.
# 2. D11 is an all-nan column (already shown)
# 3. D12 is an only-ProductCD == C column
# 4. D4 and D5 are related to an account , not a credit card (since they are not present in ProductCD == S, which doesn't have the P_emaildomain column). If so, we'll see the relationship between them and we'll apply the same nan-transformation that we have done to D9 and D2.
# 5. D8 and D9 are closely related because they are in the same NaN group for all ProductCD. Also, D8 is the only column that have non-integer values. Our hypoteshis: D8 is a sum of days and hours. We'll make round(D8-D9) and see what we can get.

# In[ ]:


# Testing hypothesis 1
sns.lmplot(x = 'D1', y = 'D2', data = train)

#Here, we can see that D2 is ALWAYS lower than or equal to D1.


# In[ ]:


#Testing hypothesis 1
train[~train['D2'].isnull()]['D1'].nunique()


# In[ ]:


#Testing hypothesis 1
train[train['D2'].isnull()]['D1'].nunique()


# In[ ]:


#Testing hypothesis 1
#From the two cells above, we can see that D2 isn't a function of D1


# In[ ]:


#Testing hypothesis 3
train[train['ProductCD'] == 'C']['D12'].nunique()


# In[ ]:


#Testing hypothesis 3
train[~(train['ProductCD'] == 'C')]['D12'].nunique()

#As we have supposed , D12 is related only to the C column


# In[ ]:


#Testing hypothesis 4
train[train['P_emaildomain'].isnull()]['D4'].nunique()


# In[ ]:


#Testing hypothesis 4
train[~(train['P_emaildomain'].isnull())]['D4'].nunique()


# In[ ]:


#Testing hypothesis 4
train[train['P_emaildomain'].isnull()]['D5'].nunique()


# In[ ]:


#Testing hypothesis 4
train[~(train['P_emaildomain'].isnull())]['D5'].nunique()


# In[ ]:


#From the cells above, maybe D4 or D5 is the time since the account was created.
#But, what is the relationship between D4 and D5?
#D5 is always less than or equal D4
sns.lmplot(x = 'D4', y = 'D5', data = train)


# In[ ]:


#Testing hypothesis 5

train['D8_rounded'] = round(train['D8'] - train['D9'])
date_features.append('D8_rounded')


# In[ ]:


#Testing hypothesis 5
train['D8_rounded'].value_counts()
#As we supposed


# In[ ]:


#Testing hypothesis 5
sns.distplot(train['D8_rounded'], kde = False)


# In[ ]:


train_sec = train['TransactionDT'].max() - train['TransactionDT'].min() 
train_min = train_sec//60 
train_hours = train_min//60
train_days = train_hours//24
train_weeks = train_days//7
time_periods_train = [train_sec,train_min,train_hours,train_days,train_weeks]
time_periods_train


# In[ ]:


test_sec = test['TransactionDT'].max() - test['TransactionDT'].min() 
test_min = test_sec//60 
test_hours = test_min//60
test_days = test_hours//24
test_weeks = test_days//7
time_periods_test = [test_sec,test_min,test_hours,test_days,test_weeks]
time_periods_test


# In[ ]:


for date_feature in date_features:
    print(train[date_feature].max())


# In[ ]:


for date_feature in date_features:
    print(train[date_feature].min())


# In[ ]:


for date_feature in date_features:
    print(train[date_feature].max() - train[date_feature].min())


# In[ ]:


# Since all Dx features , except the ones created by us and TransactionDT seems to be spanned in days, we'll apply our transformation
# With the constant 'a' being 1/(86400)
for date_feature in ['D1','D2','D3','D4','D5','D6','D7','D8','D10','D12','D13','D14','D15']:  
    test[date_feature] = test['TransactionDT']//86400 - test[date_feature]
    train[date_feature] = train['TransactionDT']//86400 - train[date_feature]


# * *Here, we haven't un-anonymized our Data Features yet. Some of them may be what we'll need to create the RFM features (time between transactions, for example), but we'll only know it when we have the Unique ID for customers.*

# ## EDA with transformed date features ##

# In[ ]:


sns.distplot(train['D1'], kde = False)


# In[ ]:


sns.distplot(train['D2'], kde = False)


# In[ ]:


sns.distplot(train['D3'], kde = False)


# In[ ]:


sns.distplot(train['D4'], kde = False)


# In[ ]:


sns.distplot(train['D5'], kde = False)


# In[ ]:


sns.distplot(train['D6'], kde = False)


# In[ ]:


sns.distplot(train['D7'], kde = False)


# In[ ]:


sns.distplot(train['D8'], kde = False)


# In[ ]:


sns.distplot(train['D10'], kde = False)


# In[ ]:


sns.distplot(train['D12'], kde = False)


# In[ ]:


sns.distplot(train['D13'], kde = False)


# In[ ]:


sns.distplot(train['D14'], kde = False)


# In[ ]:


sns.distplot(train['D15'], kde = False)


# In[ ]:


# Create Figure (empty canvas)
fig = plt.figure()

# Add set of axes to figure
axes = fig.add_axes([0.1, 0.1, 2.5, 2.5]) # left, bottom, width, height (range 0 to 1)
sns.barplot(x = 'weeks', y = 'isFraud', data = train, estimator=lambda x: len(x) / len(train) * 100)


# In[ ]:


# Create Figure (empty canvas)
fig = plt.figure()

# Add set of axes to figure
axes = fig.add_axes([0.1, 0.1, 2.5, 2.5]) # left, bottom, width, height (range 0 to 1)
sns.barplot(x = 'days_month', y = 'isFraud', data = train, estimator=lambda x: len(x) / len(train) * 100)


# In[ ]:


# Create Figure (empty canvas)
fig = plt.figure()

# Add set of axes to figure
axes = fig.add_axes([0.1, 0.1, 2.5, 2.5]) # left, bottom, width, height (range 0 to 1)
sns.barplot(x = 'hours_day', y = 'isFraud', data = train, estimator=lambda x: len(x) / len(train) * 100)


# In[ ]:


# Create Figure (empty canvas)
fig = plt.figure()

# Add set of axes to figure
axes = fig.add_axes([0.1, 0.1, 2.5, 2.5]) # left, bottom, width, height (range 0 to 1)

sns.barplot(x = 'minutes_hour', y = 'isFraud', data = train, estimator=lambda x: len(x) / len(train) * 100)


# In[ ]:


# Create Figure (empty canvas)
fig = plt.figure()

# Add set of axes to figure
axes = fig.add_axes([0.1, 0.1, 2.5, 2.5]) # left, bottom, width, height (range 0 to 1)
sns.heatmap(train[date_features].corr(), annot = True, square = True)


# In[ ]:


sns.lmplot(x = 'D6', y = 'D12', data = train, hue = 'isFraud', palette = 'pink')


# In[ ]:


sns.lmplot(x = 'D4', y = 'D12', data = train, hue = 'isFraud')


# In[ ]:


sns.lmplot(x = 'D5', y = 'D7', data = train, hue = 'isFraud')


# In[ ]:


sns.lmplot(x = 'D1', y = 'D2', data = train, hue = 'isFraud')


# In[ ]:


# It seems like the fraudulent behaviour is focused where the high-correlated features are equal. 
# Thus, we'll create deltaFeatures representing Dx - Dy
for df in train, test:
    df['delta1'] = df['D1'] - df['D2']
    df['delta2'] = df['D4'] - df['D6']
    df['delta3'] = df['D4'] - df['D12']
    df['delta4'] = df['D5'] - df['D7']
    df['delta5'] = df['D6'] - df['D12']


# In[ ]:


sns.distplot(train['delta1'], kde = False)


# In[ ]:


sns.distplot(train['delta2'], kde = False)


# In[ ]:


sns.distplot(train['delta3'], kde = False)


# In[ ]:


train['delta3'].unique()


# In[ ]:


train[train['D4'].isna() == True]


# In[ ]:


train[train['D12'].isna() == True]


# In[ ]:


train['D4_nan'] = train['D4'].isnull().apply(lambda a: 1 if a == True else 0)
test['D4_nan'] = test['D4'].isnull().apply(lambda a: 1 if a == True else 0)
date_features.append('D4_nan')
train[['D4_nan','D4', 'D12']].head(30)


# In[ ]:


train[['D4_nan','D4', 'D12']].tail(30)


# In[ ]:


sns.distplot(train['delta4'], kde = False)


# In[ ]:


sns.distplot(train['delta5'], kde = False)


# ## Summary on Dn features analysis ##
# 
# * D1 - D15 (except D9) seems to be spanned in days. We have transformed these features as Dn = TransactionDT(in days) - Dn
# * D9 is the HOURS OF DAY feature normalized. We've dropped this feature, since we've created a new feature with zero missing values called hours_day. We've created a new column, D9_nan which is zero wherever D9 has a missing value and 1 otherwise.
# * We've seen that the D8 column is the sum of some number of days + D9
# * We've seen that the D4 and D5 columns are related to an account. Maybe D4 is the time since the creation of the account, since it is >= D5 for all values.
# * D4 and D12 columns are equal, except for the number of missing values and 2 rows
# * We've created the features days, weeks, days_month,hours_day, weeks and minutes_hour, for the purpose of network analysis
# * We've seen that the fraudulent behaviour is pretty clear where the highly correlated date features are equal, so we've created new variables, 'deltaI' that represents Dn - Dm, where Dn and Dm are highly correlated date features
# * We've seen, also, that the number of fraud is approximately normal given the hours of day
# * We've seen, in the weeks feature, that there is some peak in the period analysed. Maybe some holiday(s). This will bring problems, since the test set also have a peak that doesn't match with that of train set.

# ## Analysis of Cn features ##

# In[ ]:


c_features = []
for i in range(1,15):
    c_features.append('C'+str(i))
train[c_features]


# In[ ]:


# Create Figure (empty canvas)
fig = plt.figure()

# Add set of axes to figure
axes = fig.add_axes([0.1, 0.1, 2.5, 2.5]) # left, bottom, width, height (range 0 to 1)

sns.heatmap(train[c_features].corr(), annot = True)


# In[ ]:


sns.pairplot(train[c_features])

#All C_features, except C3, have a linear-ish relationship. C5 and C9 seem to be constant.


# In[ ]:


sns.lmplot(x = 'C1', y = 'C4', data = train, hue = 'isFraud')


# In[ ]:


sns.lmplot(x = 'C1', y = 'C11', data = train, hue = 'isFraud')


# In[ ]:


sns.lmplot(x = 'C1', y = 'C2', data = train,hue = 'isFraud')


# In[ ]:


sns.lmplot(x = 'C4', y = 'C14', data = train, hue = 'isFraud')


# In[ ]:


train['C3'].value_counts()
#(This is some kind of counting from zero to 25?)


# In[ ]:


# Create Figure (empty canvas)
fig = plt.figure()

# Add set of axes to figure
axes = fig.add_axes([0.1, 0.1, 2.5, 2.5]) # left, bottom, width, height (range 0 to 1)

sns.barplot(x = 'C3', y = 'isFraud', data = train, estimator=lambda x: len(x) / len(train) * 100)


# In[ ]:


train['C1'].value_counts().index[0]


# In[ ]:


for c in c_features:
    print(c)
    print('Max = %f'%train[c].max())
    print('Min = %f'%train[c].min())
    print('Mode = %f'%train[c].value_counts().index[0])


# In[ ]:


train['C5'].value_counts()


# In[ ]:


train.drop('C5', axis = 1, inplace = True)
test.drop('C5', axis = 1, inplace = True)
c_features.remove('C5')


# In[ ]:


train['C9'].value_counts()


# In[ ]:


train.drop('C9', axis = 1, inplace = True)
test.drop('C9', axis = 1, inplace = True)
c_features.remove('C9')


# ## Summary of Cn features analysis ##
# 
# * Features C5 and C9 are constants. We've dropped them.
# * Feature C3 is some kind of counting highly concentrated around zero and has maximum value 26. It is highly uncorrelated with all other Cn features.
# * C1, C2, C4, C6, C8, C9, C10, C11, C12, C13, C14 are highly correlated features, also highly concentrated around zero and spanning around some thousands.
# * Nothing useful about the meaning of these features were discovered.

# ## Unique Card Identification (UCI) ##
# 
# It makes sense to use all the card variables as components of Unique Card Identification. Also, it makes sense to use addr1, addr2 and P_emaildomain. By the productCD segmentation analysis, however, we've seen that, for ProductCD == S we don't have P_emaildomain and for ProductCD == C we have P_emaildomain == R_emaildomain,so we won't use it to UCI identification. We need some Date variable that associates a card with an account, so we'll use D1 (transformed) as this variable. The full analysis behind this reasoning can be found here: https://www.kaggle.com/gabrielsantanna/projeto-vesta-fraudes

# In[ ]:


#Let us take a look into the card_n columns
train[['card1','card2','card3','card4','card5','card6']]


# In[ ]:


train['card1'].value_counts()


# In[ ]:


sns.distplot(train['card1'],kde = False)


# In[ ]:


train['card2'].value_counts()


# In[ ]:


sns.distplot(train['card2'], kde = False)


# In[ ]:


train['card3'].value_counts()


# In[ ]:


sns.distplot(train['card3'], kde = False)


# In[ ]:


train['card5'].value_counts()


# In[ ]:


sns.distplot(train['card5'], kde = False)


# In[ ]:


sns.heatmap(train[['card1','card2','card3','card4','card5','card6']].corr(), annot = True)


# In[ ]:


##It seems that there isn't a reason for us for don't put all card features into the UCI 


# In[ ]:


#So, let us do it
for df in train, test:
    df['uci'] = df['card1'].astype(str)+' '+df['card2'].astype(str)+' '+df['card3'].astype(str)+' '+df['card4'].astype(str)+' '+df['card5'].astype(str)+' '+df['card6'].astype(str)+' '+df['addr1'].astype(str)+' '+df['addr2'].astype(str)+' '+df['D1'].astype(str)
    


# In[ ]:


train['uci'].value_counts()


# In[ ]:


date_features.append('uci')
train[date_features].groupby('uci').std()


# In[ ]:


train[date_features].groupby('uci').nunique()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ## Merchant Unique Identification (MUI) ##
# 
# In this section, we'll try to construct a MUI. We'll start with the following features:
# 
# ProductCD

# ## Recency, Frequency and Monetary Value (RFM) features ##
# 
# In this section, we'll make the following features, related with the UCI:
# 
# 1. Average spending per transaction over a 30-day period on all transactions till this transaction
# 2. Average amount spent over the course of 1 week during the past 3 months 
# 3. Average spending per day over the past 30 days till this transaction
# 4. Average spending per day on a merchant type over the past 30 days till this transaction
# 5. Total number of transactions with the same merchant over a period of 30 days before a given transaction (HERE WE NEED MERCHANT UNIQUE IDENTIFICATION)
# 6. Average weekly spending on a merchant type during the past 3 months before a given transaction
# 7. Total amount spent with a credit card to the day of a given transaction
# 8. Total number of transactions in the day of a given transaction
# 9. Average amount per day spent over a 30-day period on all transactions up to this one on the same merchant of this transaction. 
# 10. Total number of transactions with the same merchant during the last month.
# 11. Average amount spent over a 30-day period on all transactions up to this transaction with the same currency.
# 12. Total number of transactions with the same merchant during the last month.
# 13. Average amount spent over a 30-day period in the same country.
# 14. Total number of transactions over a 30-day period in the same country.
# 15. Average amount spent over the course of 1 week during the past 3 months on the same merchant.
# 
# 
# * *Features 5, 9, 10, 12, 15 we'll only generate after creating the MUI, Merchant Unique Identification*

# In[ ]:





# In[ ]:


#1. AvspTr30
#How do I create averages?


# In[ ]:


#2. AvspWe90
#How do I create averages?


# In[ ]:


#3. AvspDay30
#How do I create averages?


# In[ ]:


#4. AvspMerTy30
#How do I create averages?


# In[ ]:


#6 AvspWeMerTy90
#How do I create averages?


# In[ ]:


#7 TotAmt

