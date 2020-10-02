#!/usr/bin/env python
# coding: utf-8

# # Home Credit Default Risk
# 
# This kernel will contain EDA, visualization, feature engineering and some modelling. Work currently in progress.

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns

import datetime
import lightgbm as lgb
from scipy import stats
from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold
from wordcloud import WordCloud
from collections import Counter
from nltk.corpus import stopwords
from nltk.util import ngrams
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
import os
import xgboost as xgb
import lightgbm as lgb

import os
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
pd.set_option('max_columns', 150)


# There are several files with data, let's go through them step by step.

# In[ ]:


folder = '../input/'
application_train = pd.read_csv(os.path.join(folder, 'application_train.csv'))
application_test = pd.read_csv(os.path.join(folder, 'application_test.csv'))
bureau = pd.read_csv(os.path.join(folder, 'bureau.csv'))
bureau_balance = pd.read_csv(os.path.join(folder, 'bureau_balance.csv'))
POS_CASH_balance = pd.read_csv(os.path.join(folder, 'POS_CASH_balance.csv'))
credit_card_balance = pd.read_csv(os.path.join(folder, 'credit_card_balance.csv'))
previous_application = pd.read_csv(os.path.join(folder, 'previous_application.csv'))
installments_payments = pd.read_csv(os.path.join(folder, 'installments_payments.csv'))
sample_submission = pd.read_csv(os.path.join(folder, 'sample_submission.csv'))


# ## Data Exploration

# ### application_train and application_test
# These are main files with data and technically we can use only them to make predictions. Obviously using additional data is necessary to improve score.

# In[ ]:


application_train.head()


# We have 122 columns in just main file! Let's take a look on some of them.

# #### Categorical features

# ##### Target

# In[ ]:


application_train.TARGET.value_counts(normalize=True)


# We have disbalanced target, though disbalance isn't really serious.

# ##### NAME_CONTRACT_TYPE

# In[ ]:


pd.crosstab(application_train.TARGET, application_train.NAME_CONTRACT_TYPE, dropna=False, normalize='all')


# We can see that there are two types of contract - cash loans and revolving loans. Most of the loans are cash loans which are defaulted.

# ##### CODE_GENDER

# In[ ]:


pd.crosstab(application_train.TARGET, application_train.CODE_GENDER, dropna=False)


# We can see that women take more loans and higher percentage of them repays the loans. And there are 4 people with unindentified gender, who repayed their loans :)

# ##### FLAG_OWN_CAR and FLAG_OWN_REALTY

# In[ ]:


print('There are {0} people with realty. {1}% of them repay loans.'.format(application_train[application_train.FLAG_OWN_REALTY == 'Y'].shape[0], np.round(application_train[application_train.FLAG_OWN_REALTY == 'Y'].TARGET.value_counts(normalize=True).values[1], 3) * 100))
print('There are {0} people with cars. {1}% of them repay loans.'.format(application_train[application_train.FLAG_OWN_CAR == 'Y'].shape[0], np.round(application_train[application_train.FLAG_OWN_CAR == 'Y'].TARGET.value_counts(normalize=True).values[1], 4) * 100))
print('Average age of the car is {:.2f} years.'.format(application_train.groupby(['FLAG_OWN_CAR'])['OWN_CAR_AGE'].mean().values[1]))


# ##### CNT_CHILDREN and NAME_FAMILY_STATUS

# In[ ]:


pd.crosstab(application_train.CNT_CHILDREN, application_train.NAME_FAMILY_STATUS, dropna=False)


# We can see that most of the people are married and have zero children. In face we can divide people into two group based on their family status - living together with their partner or single.

# In[ ]:


pd.crosstab(application_train.CNT_CHILDREN, application_train.CNT_FAM_MEMBERS, dropna=False)


# It isn't surprising that there are a lot of families consisting of two or one adults. Also there are families with two adults and 1-2 children.

# ##### NAME_TYPE_SUITE
# This feature shows who was accompanying client when he was applying for the loan.

# In[ ]:


application_train['NAME_TYPE_SUITE'].value_counts(dropna=False)


# In[ ]:


pd.crosstab(application_train.NAME_TYPE_SUITE, application_train.NAME_FAMILY_STATUS, dropna=False)


# It is interesting to see that these two variables sometimes contradict each other. For example, separated, single or widowed applicants were sometimes accompanied by their partner. I suppose this means unofficial relationships? Also sometimes children accompanied the applicant. Maybe these were adult childred?

# ##### NAME_INCOME_TYPE

# In[ ]:


application_train.groupby(['NAME_INCOME_TYPE']).agg({'AMT_INCOME_TOTAL': ['mean', 'median', 'count']})


# In[ ]:


application_train[application_train['NAME_INCOME_TYPE'] == 'Maternity leave']['CODE_GENDER'].value_counts()


# We can see that there are 4 categories with little amount of people in them: several high-income businessmen, 4 women and 1 man on maternity leave, and some unemployed/students. It is quite interesting that unemployed/students have quite a high income.
# And of course, most of the people work.

# In[ ]:


s = pd.crosstab(application_train.NAME_INCOME_TYPE, application_train.OCCUPATION_TYPE, dropna=False).style.background_gradient(cmap='viridis', low=.5, high=0).highlight_null('red')
s


# ##### AMT_GOODS_PRICE
# For consumer loans it is the price of the goods for which the loan is given

# In[ ]:


print('{0} zero values.'.format(application_train[application_train['AMT_GOODS_PRICE'].isnull()].shape[0]))


# So this means that only 278 loans have some other type. Let's fo deeper.

# In[ ]:


non_zero_good_price = application_train[application_train['AMT_GOODS_PRICE'].isnull() == False]
credit_to_good_price = non_zero_good_price['AMT_CREDIT'] / non_zero_good_price['AMT_GOODS_PRICE']
plt.boxplot(credit_to_good_price);
plt.title('Credit amount to goods price.');


# We can see that most of the loans have the amount which is similar to the goods price, but there are some outliers.

# ##### NAME_HOUSING_TYPE

# In[ ]:


sns.countplot(application_train['NAME_HOUSING_TYPE']);
plt.xticks(rotation=45);
plt.title('Counts of housing type')


# ##### Contact information
# There are 6 features showing that client provided some contact information, let's see how many ways of contact clients usually provide.

# In[ ]:


application_train['contact_info'] = application_train['FLAG_MOBIL'] + application_train['FLAG_EMP_PHONE'] + application_train['FLAG_WORK_PHONE'] + application_train['FLAG_CONT_MOBILE'] + application_train['FLAG_PHONE'] + application_train['FLAG_EMAIL']
sns.countplot(application_train['contact_info']);
plt.title('Count of ways to contact client');


# Most clients provide 3 ways to contact them and usually minimus is 2, if we don't consider several people who left only 1.

# # deliquencies
# 
# It is very important to see how many times clients was late with payments or defaulted his loans. I suppose info about his social circle is also important. I'll divide values into 2 groups: 0, 1 and more than 1.

# In[ ]:


application_train.loc[application_train['OBS_30_CNT_SOCIAL_CIRCLE'] > 1, 'OBS_30_CNT_SOCIAL_CIRCLE'] = '1+'
application_train.loc[application_train['DEF_30_CNT_SOCIAL_CIRCLE'] > 1, 'DEF_30_CNT_SOCIAL_CIRCLE'] = '1+'
application_train.loc[application_train['OBS_60_CNT_SOCIAL_CIRCLE'] > 1, 'OBS_60_CNT_SOCIAL_CIRCLE'] = '1+'
application_train.loc[application_train['DEF_60_CNT_SOCIAL_CIRCLE'] > 1, 'DEF_60_CNT_SOCIAL_CIRCLE'] = '1+'


# In[ ]:


fig, ax = plt.subplots(figsize = (30, 8))
plt.subplot(1, 4, 1)
sns.countplot(application_train['OBS_30_CNT_SOCIAL_CIRCLE']);
plt.subplot(1, 4, 2)
sns.countplot(application_train['DEF_30_CNT_SOCIAL_CIRCLE']);
plt.subplot(1, 4, 3)
sns.countplot(application_train['OBS_60_CNT_SOCIAL_CIRCLE']);
plt.subplot(1, 4, 4)
sns.countplot(application_train['DEF_60_CNT_SOCIAL_CIRCLE']);


# #### Continuous variables

# ##### AMT_INCOME_TOTAL

# In[ ]:


sns.boxplot(application_train['AMT_INCOME_TOTAL']);
plt.title('AMT_INCOME_TOTAL boxplot');


# In[ ]:


sns.boxplot(application_train[application_train['AMT_INCOME_TOTAL'] < np.percentile(application_train['AMT_INCOME_TOTAL'], 90)]['AMT_INCOME_TOTAL']);
plt.title('AMT_INCOME_TOTAL boxplot on data within 90 percentile');


# In[ ]:


application_train.groupby('TARGET').agg({'AMT_INCOME_TOTAL': ['mean', 'median', 'count']})


# In[ ]:


plt.hist(application_train['AMT_INCOME_TOTAL']);
plt.title('AMT_INCOME_TOTAL histogram');


# In[ ]:


plt.hist(application_train[application_train['AMT_INCOME_TOTAL'] < np.percentile(application_train['AMT_INCOME_TOTAL'], 90)]['AMT_INCOME_TOTAL']);
plt.title('AMT_INCOME_TOTAL histogram on data within 90 percentile');


# In[ ]:


plt.hist(np.log1p(application_train['AMT_INCOME_TOTAL']));
plt.title('AMT_INCOME_TOTAL histogram on data with log1p transformation');


# We can see following things from the information above:
# - income feature has some huge outliers. This could be due to rich individuals or due to errors in data;
# - average income is almost similar for those who repay the loans and those who don't;
# - if we leave only data within 90 percentile, it is almost normally distributed;
# - log transformation also helps;

# ##### AMT_CREDIT

# In[ ]:


sns.boxplot(application_train['AMT_CREDIT'], orient='v');
plt.title('AMT_CREDIT boxplot');


# In[ ]:


sns.boxplot(application_train[application_train['AMT_CREDIT'] < np.percentile(application_train['AMT_CREDIT'], 95)]['AMT_CREDIT'], orient='v');
plt.title('AMT_CREDIT boxplot on data within 90 percentile');


# In[ ]:


application_train.groupby('TARGET').agg({'AMT_CREDIT': ['mean', 'median', 'count']})


# In[ ]:


plt.hist(application_train['AMT_CREDIT']);
plt.title('AMT_CREDIT histogram');


# In[ ]:


plt.hist(application_train[application_train['AMT_CREDIT'] < np.percentile(application_train['AMT_CREDIT'], 90)]['AMT_CREDIT']);
plt.title('AMT_INCOME_TOTAL histogram on data within 90 percentile');


# In[ ]:


plt.hist(np.log1p(application_train['AMT_CREDIT']));
plt.title('AMT_CREDIT histogram on data with log1p transformation');


# This feature shows the amount of the loan in question.
# We can see following things from the information above:
# - income feature has some outliers. Maybe mortgage?;
# - average credit amoint is almost similar for those who repay the loans and those who don't;
# - if we leave only data within 95 percentile, it is almost normally distributed;
# - log transformation also helps;

# ##### DAYS_BIRTH

# In[ ]:


application_train['age'] = application_train['DAYS_BIRTH'] / -365
plt.hist(application_train['age']);
plt.title('Histogram of age in years.');


# We can see that age distribution is almost normal and most of the people are between 30 and 40 years.

# ##### DAYS_EMPLOYED

# In[ ]:


application_train.loc[application_train['DAYS_EMPLOYED'] == 365243, 'DAYS_EMPLOYED'] = 0
application_train['years_employed'] = application_train['DAYS_EMPLOYED'] / -365
plt.hist(application_train['years_employed']);
plt.title('Length of working at current workplace in years.');


# Ther was a strange value - 365243, it could mean empty values or some errors, so I replace it with zero.
# A lot of people don't work, but let's look deeper into this.

# In[ ]:


application_train.groupby(['NAME_INCOME_TYPE']).agg({'years_employed': ['mean', 'median', 'count', 'max'], 'age': ['median']})


# Well, it seems that a lot of non-working people are pensioners, which is normal. As for working people - they seem to work for several years at one place.

# Ther are so many features and so many possible angles from which we can analyze them. Let's see this for example:

# In[ ]:


application_train.groupby(['NAME_EDUCATION_TYPE', 'NAME_INCOME_TYPE']).agg({'AMT_INCOME_TOTAL': ['mean', 'median', 'count', 'max']})


# We can see that most of the loans are taken by working people with secondary education.

# ## Transforming and merging data

# In[ ]:


application_train['AMT_INCOME_TOTAL'] = np.log1p(application_train['AMT_INCOME_TOTAL'])
application_train['AMT_CREDIT'] = np.log1p(application_train['AMT_CREDIT'])
application_train['OWN_CAR_AGE'] = application_train['OWN_CAR_AGE'].fillna(0)
application_train['app AMT_CREDIT / AMT_ANNUITY'] = application_train['AMT_CREDIT'] / application_train['AMT_ANNUITY']
application_train['app EXT_SOURCE mean'] = application_train[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis = 1)
application_train['app EXT_SOURCE_1 / DAYS_BIRTH'] = application_train['EXT_SOURCE_1'] / application_train['DAYS_BIRTH']
application_train['app AMT_INCOME_TOTAL / 12 - AMT_ANNUITY'] = application_train['AMT_INCOME_TOTAL'] / 12. - application_train['AMT_ANNUITY']
application_train['app AMT_INCOME_TOTAL / AMT_ANNUITY'] = application_train['AMT_INCOME_TOTAL'] / application_train['AMT_ANNUITY']
application_train['app AMT_INCOME_TOTAL - AMT_GOODS_PRICE'] = application_train['AMT_INCOME_TOTAL'] - application_train['AMT_GOODS_PRICE']


# In[ ]:


application_test.loc[application_test['OBS_30_CNT_SOCIAL_CIRCLE'] > 1, 'OBS_30_CNT_SOCIAL_CIRCLE'] = '1+'
application_test.loc[application_test['DEF_30_CNT_SOCIAL_CIRCLE'] > 1, 'DEF_30_CNT_SOCIAL_CIRCLE'] = '1+'
application_test.loc[application_test['OBS_60_CNT_SOCIAL_CIRCLE'] > 1, 'OBS_60_CNT_SOCIAL_CIRCLE'] = '1+'
application_test.loc[application_test['DEF_60_CNT_SOCIAL_CIRCLE'] > 1, 'DEF_60_CNT_SOCIAL_CIRCLE'] = '1+'
np.log1p(application_test['AMT_INCOME_TOTAL'])
np.log1p(application_test['AMT_CREDIT'])
application_test['age'] = application_test['DAYS_BIRTH'] / -365
application_test.loc[application_test['DAYS_EMPLOYED'] == 365243, 'DAYS_EMPLOYED'] = 0
application_test['years_employed'] = application_test['DAYS_EMPLOYED'] / -365
application_test['AMT_INCOME_TOTAL'] = np.log1p(application_test['AMT_INCOME_TOTAL'])
application_test['AMT_CREDIT'] = np.log1p(application_test['AMT_CREDIT'])
application_test['OWN_CAR_AGE'] = application_test['OWN_CAR_AGE'].fillna(0)
application_test['app AMT_CREDIT / AMT_ANNUITY'] = application_test['AMT_CREDIT'] / application_test['AMT_ANNUITY']
application_test['app EXT_SOURCE mean'] = application_test[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis = 1)
application_test['app EXT_SOURCE_1 / DAYS_BIRTH'] = application_test['EXT_SOURCE_1'] / application_test['DAYS_BIRTH']
application_test['app AMT_INCOME_TOTAL / 12 - AMT_ANNUITY'] = application_test['AMT_INCOME_TOTAL'] / 12. - application_test['AMT_ANNUITY']
application_test['app AMT_INCOME_TOTAL / AMT_ANNUITY'] = application_test['AMT_INCOME_TOTAL'] / application_test['AMT_ANNUITY']
application_test['app AMT_INCOME_TOTAL - AMT_GOODS_PRICE'] = application_test['AMT_INCOME_TOTAL'] - application_test['AMT_GOODS_PRICE']


# In[ ]:


for col in ['FONDKAPREMONT_MODE', 'HOUSETYPE_MODE', 'WALLSMATERIAL_MODE', 'EMERGENCYSTATE_MODE', 'OBS_30_CNT_SOCIAL_CIRCLE', 'DEF_30_CNT_SOCIAL_CIRCLE', 'OBS_60_CNT_SOCIAL_CIRCLE', 'DEF_60_CNT_SOCIAL_CIRCLE',
           'NAME_CONTRACT_TYPE', 'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'NAME_TYPE_SUITE', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'OCCUPATION_TYPE',
            'WEEKDAY_APPR_PROCESS_START', 'ORGANIZATION_TYPE', 'WEEKDAY_APPR_PROCESS_START']:
    unique_values = list(set(list(application_train[col].astype(str).unique()) + list(application_test[col].astype(str).unique())))
    le.fit(unique_values)
    application_train[col] = le.transform(application_train[col].astype(str))
    application_test[col] = le.transform(application_test[col].astype(str))


# In[ ]:


train = application_train


# In[ ]:


train.head()


# In[ ]:


test = application_test


# In[ ]:


train = train.fillna(0)
test = test.fillna(0)


# In[ ]:


X = train.drop(['SK_ID_CURR','contact_info', 'TARGET'], axis=1)
y = train['TARGET']
X_test = test.drop(['SK_ID_CURR'], axis=1)


# ## Basic modelling, LGB

# In[ ]:


X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.20, random_state=42)
params = {
    'boosting': 'dart',
    'application': 'binary',
    'learning_rate': 0.01,
    'num_leaves': 34,
    'max_depth': 5,
    'feature_fraction': 0.9,
    'scale_pos_weight': 2,
    'reg_alpha': 0.05,
    'reg_lambda': 0.1}
model = lgb.train(params, lgb.Dataset(X_train, y_train), 1000, [lgb.Dataset(X_train, y_train), lgb.Dataset(X_valid, y_valid)], verbose_eval=10, early_stopping_rounds=20)


# In[ ]:


lgb.plot_importance(model, max_num_features=30, figsize=(24, 18))


# In[ ]:


folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
params={'colsample_bytree': 0.8,
 'learning_rate': 0.01,
 'num_leaves': 34,
 'subsample': 0.97,
 'max_depth': 8,
 'reg_alpha': 0.03,
 'reg_lambda': 0.07,
 'min_split_gain': 0.01,
 #'min_child_weight': 38
       }
prediction = np.zeros(X_test.shape[0])
for n_fold, (train_idx, valid_idx) in enumerate(folds.split(X, y)):
        train_x, train_y = X.iloc[train_idx], y.iloc[train_idx]
        valid_x, valid_y = X.iloc[valid_idx], y.iloc[valid_idx]
        clf = lgb.LGBMClassifier(**params)
        clf.fit(train_x, train_y, 
                eval_set = [(train_x, train_y), (valid_x, valid_y)], eval_metric = 'auc', 
                verbose = 100, early_stopping_rounds = 50)
        prediction += clf.predict(X_test)


# In[ ]:


sub = test[['SK_ID_CURR']].copy()
sub['TARGET'] = prediction / 10
sub.to_csv('sub.csv', index= False)


# This was EDA and basic feature engineering. I know that feature engineering and modelling could be much better, but decided to make EDA the main focus of this kernel. I'll do better feature engineering and modelling in the next one.
