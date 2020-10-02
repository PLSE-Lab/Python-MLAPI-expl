#!/usr/bin/env python
# coding: utf-8

# This is my introduction on the Home Credit Default Risk problem.  I first spend time reviewing the variables and examining the data in the training data.  Next, I move on to incorporating data from the other data files we have been given.  Then I begin paring down the data to the data I will use for modeling.

# # Introduction

# In[ ]:


# imports
import numpy as np 
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os
import gc
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score

# Suppress warnings 
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


# list of data files available
print(os.listdir('../input'))


# In[ ]:


# training data
app_train = pd.read_csv('../input/application_train.csv')
print('training data shape: ', app_train.shape)
app_train.head()


# In[ ]:


# testing data
app_test = pd.read_csv('../input/application_test.csv')
print('testing data shape: ', app_test.shape)
app_test.head()


# In[ ]:


# review target variable
app_train['TARGET'].value_counts()
app_train['TARGET'].astype(int).plot.hist();
(app_train['TARGET']).describe()


# In[ ]:


# summary of training data
app_train.describe()


# In[ ]:


# join training and testing data sets so we keep the same number of features in both
train_len = len(app_train)
dataset = pd.concat(objs=[app_train, app_test], axis=0).reset_index(drop=True)
# shape of combined dataset should be sum of rows in training and testing (307511 + 48744 = 356255) and 122 columns (testing data doesn't have target)
print('dataset data shape: ', dataset.shape)


# In[ ]:


# missing values
dataset.isnull().sum()


# This is a lot of missing values to deal with.  A number of variables are missing over two-thirds of the data.

# In[ ]:


# types of data
dataset.dtypes.value_counts()


# In[ ]:


# categorical data - how many different categories for each variable
dataset.select_dtypes('object').apply(pd.Series.nunique, axis = 0)


# In[ ]:


dataset.describe(include=[np.object])


# In[ ]:


# use label encoding for categorical variables with only two categories
le = LabelEncoder()
count = 0
le_vars = []
for col in dataset:
    if dataset[col].dtype == 'object':
        if len(list(dataset[col].unique())) == 2:
            le.fit(dataset[col])
            dataset[col] = le.transform(dataset[col])
            count += 1
            le_vars.append(col)
            
print('%d columns were label encoded' % count)
print(le_vars)


# Note that four columns had two categories, but only three were label encoded - EMERGENCYSTATE_MODE was not.
# This is because the three label encoded did not have any missing values, while EMERGENCYSTATE_MODE was missing values, so it really has three categories - Yes, No, and missing.

# In[ ]:


# use one-hot encoding for remaining categorical variables
dataset = pd.get_dummies(dataset)
print('dataset data shape: ', dataset.shape)


# Note that using one-hot encoding, we went from 122 variables to 243 variables - a significant increase.  At a later point, we will probably want to remove those that are not relevant.

# # Explore the Data

# For the non-indicator variables, let's look at individual variables for outliers or other interesting information.

# ## CNT_CHILDREN

# In[ ]:


(dataset['CNT_CHILDREN']).describe()


# Most applicants have no children or only one child.  However, the maximum is 20, which seems high.  Let's look closer at this data.

# In[ ]:


dataset['CNT_CHILDREN'].plot.hist(title = 'CNT_CHILDREN Histogram');
plt.xlabel('CNT_CHILDREN')


# In[ ]:


# plot CNT_CHILDREN against the TARGET to better understand the data
g = sns.factorplot(x='CNT_CHILDREN', y='TARGET', data=app_train, kind="bar", size = 6, palette = "muted")
g.despine(left=True)
g = g.set_ylabels("default probability")


# It appears that if CNT_CHILDREN is greater than six, the probability of default is higher.  From above, we know that on average the default rate is about 8.07%.  Let's look further at CNT_CHILDREN.

# In[ ]:


outlier_children = app_train[app_train['CNT_CHILDREN'] > 6]
print('count of outlier_children: ', len(outlier_children))
print('default probability of outlier_children: %0.2f%%' %(100 * outlier_children['TARGET'].mean()))
(outlier_children['CNT_CHILDREN']).describe()


# In[ ]:


# create a flag for outliers in the CNT_CHILDREN column, and then replace these values with nan
dataset['CNT_CHILDREN_outlier'] = dataset['CNT_CHILDREN'] > 6
for i in dataset['CNT_CHILDREN']:
    if i > 6:
        dataset['CNT_CHILDREN'].replace({i: np.nan}, inplace = True)


# In[ ]:


# review CNT_CHILDREN after our modifications
(dataset['CNT_CHILDREN']).describe()


# In[ ]:


dataset['CNT_CHILDREN'].plot.hist(title = 'CNT_CHILDREN Histogram');
plt.xlabel('CNT_CHILDREN')


# ## AMT_INCOME_TOTAL

# In[ ]:


(dataset['AMT_INCOME_TOTAL']).describe()


# In[ ]:


dataset['AMT_INCOME_TOTAL'].plot.hist(range = (1,1000000), title = 'AMT_INCOME_TOTAL Histogram');
plt.xlabel('AMT_INCOME_TOTAL')


# This looks to be close to what we would expect.  There are relatively fewer high incomes, and the mean is greater than the median.

# ## AMT_CREDIT

# In[ ]:


(dataset['AMT_CREDIT']).describe()


# In[ ]:


dataset['AMT_CREDIT'].plot.hist(title = 'AMT_CREDIT Histogram');
plt.xlabel('AMT_CREDIT')


# This distribution is definitely skewed, but looks like we would expect for a distribution of credit.

# ## AMT_ANNUITY

# In[ ]:


(dataset['AMT_ANNUITY']).describe()


# In[ ]:


dataset['AMT_ANNUITY'].plot.hist(title = 'AMT_ANNUITY Histogram');
plt.xlabel('AMT_ANNUITY')


# There do not appear to be any outliers here, though the distribution is definitely skewed.

# ## AMT_GOODS_PRICE

# In[ ]:


(dataset['AMT_GOODS_PRICE']).describe()


# In[ ]:


dataset['AMT_GOODS_PRICE'].plot.hist(title = 'AMT_GOODS_PRICE Histogram');
plt.xlabel('AMT_GOODS_PRICE')


# Similar to the above, there do not appear to be any outliers here, though the distribution is also definitely skewed.

# ## REGION_POPULATION_RELATIVE

# In[ ]:


(dataset['REGION_POPULATION_RELATIVE']).describe()


# In[ ]:


dataset['REGION_POPULATION_RELATIVE'].plot.hist(title = 'REGION_POPULATION_RELATIVE Histogram');
plt.xlabel('REGION_POPULATION_RELATIVE')


# In[ ]:


# plot REGION_POPULATION_RELATIVE against the TARGET to better understand the data
g = sns.lineplot(x='REGION_POPULATION_RELATIVE', y='TARGET', data=app_train, palette = "muted")


# The larger values do not appear to be correlated with higher-than-average or lower-than-average default rates.  For now, I will leave these as is.

# ### Let's review the correlation matrix for the variables examined so far:

# In[ ]:


g = sns.heatmap(app_train[['TARGET','CNT_CHILDREN','AMT_INCOME_TOTAL','AMT_CREDIT','AMT_ANNUITY','AMT_GOODS_PRICE','REGION_POPULATION_RELATIVE']].corr(),annot=True, fmt = ".2f", cmap = "coolwarm")


# None of the above variables are significantly correlated with TARGET.  However, AMT_CREDIT, AMT_ANNUITY, and AMT_GOODS_PRICE are all highly correlated with each other, especially AMT_CREDIT and AMT_GOODS_PRICE.  It appears that we may need to drop at least one of these two variables.

# ## DAYS_BIRTH

# In[ ]:


(dataset['DAYS_BIRTH']).describe()


# In[ ]:


# this variable appears to be equal to (date of birth) minus (date of application), which is producing negative numbers
# if we look again at the data transformed into positive numbers and into years (by dividing by -365.25) we get the following
(dataset['DAYS_BIRTH'] / -365.25).describe()


# This distribution of ages at application date seems reasonable - no small children and reasonable max age.

# In[ ]:


(dataset['DAYS_BIRTH'] / -365.25).plot.hist(title = 'DAYS_BIRTH Histogram');
plt.xlabel('DAYS_BIRTH')


# ## DAYS_EMPLOYED

# In[ ]:


(dataset['DAYS_EMPLOYED']).describe()


# In[ ]:


# this variable appears to be equal to (date of employment) minus (date of application), which is producing negative numbers
# if we look again at the data transformed into positive numbers and into years (by dividing by -365.25) we get the following
(dataset['DAYS_EMPLOYED'] / -365.25).describe()


# In[ ]:


# it appears that a dummy value was used, possibly for people who didn't have a date of employment to enter into the application
# this group had 365243 in the data, which is approximately -1000 years
# we should also look at the other side of the distribution - 49 years of employment is a long time
dataset['DAYS_EMPLOYED'].plot.hist(title = 'DAYS_EMPLOYED Histogram');
plt.xlabel('DAYS_EMPLOYED')


# In[ ]:


outlier_days_employed = app_train[app_train['DAYS_EMPLOYED'] == 365243]
print('count of outlier_days_employed: ', len(outlier_days_employed))
print('default probability of outlier_days_employed: %0.2f%%' %(100 * outlier_days_employed['TARGET'].mean()))
(outlier_days_employed['DAYS_EMPLOYED']).describe()


# In[ ]:


# create a flag for outliers in the DAYS_EMPLOYED column, and then replace these values with nan
dataset['DAYS_EMPLOYED_outlier'] = dataset['DAYS_EMPLOYED'] == 365243
dataset['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace = True)


# In[ ]:


# review DAYS_EMPLOYED after our modifications
(dataset['DAYS_EMPLOYED']).describe()


# In[ ]:


(dataset['DAYS_EMPLOYED'] / -365.25).plot.hist(title = 'DAYS_EMPLOYED Histogram');
plt.xlabel('DAYS_EMPLOYED')


# While the distribution is skewed, it is now what we expect - long-tenure employees are rare and short-tenure employees are much more common.

# ## DAYS_REGISTRATION

# In[ ]:


(dataset['DAYS_REGISTRATION']).describe()


# In[ ]:


# this variable appears to be equal to (date of registration) minus (date of application), which is producing negative numbers
# if we look again at the data transformed into positive numbers and into years (by dividing by 365.25) we get the following
(dataset['DAYS_REGISTRATION'] / -365.25).describe()


# In[ ]:


(dataset['DAYS_REGISTRATION'] / -365.25).plot.hist(title = 'DAYS_REGISTRATION Histogram');
plt.xlabel('DAYS_REGISTRATION')


# The above looks as expected.

# ## DAYS_ID_PUBLISH

# In[ ]:


(dataset['DAYS_ID_PUBLISH']).describe()


# In[ ]:


# convert to positive years again
(dataset['DAYS_ID_PUBLISH'] / -365.25).describe()


# In[ ]:


(dataset['DAYS_ID_PUBLISH'] / -365.25).plot.hist(title = 'DAYS_ID_PUBLISH Histogram');
plt.xlabel('DAYS_ID_PUBLISH')


# I don't see any outliers, although the shape of this distribution is different from the others we have seen above.

# ### Let's review the correlation matrix for the additional variables examined so far:

# In[ ]:


g = sns.heatmap(app_train[['TARGET','DAYS_BIRTH','DAYS_EMPLOYED','DAYS_REGISTRATION','DAYS_ID_PUBLISH']].corr(),annot=True, fmt = ".2f", cmap = "coolwarm")


# DAYS_BIRTH appears to be the most correlated with TARGET so far.

# ## OWN_CAR_AGE

# In[ ]:


(dataset['OWN_CAR_AGE']).describe()


# In[ ]:


dataset['OWN_CAR_AGE'].plot.hist(title = 'OWN_CAR_AGE Histogram');
plt.xlabel('OWN_CAR_AGE')


# In[ ]:


# it seems that there are some outliers on the car age, as the max is 91
# let's get a better look at the values in the tail and whether these have a higher probability of default than average
outlier_car_age = app_train[app_train['OWN_CAR_AGE'] > 60]
print('count of outlier_car_age: ', len(outlier_car_age))
print('default probability of outlier_car_age: %0.2f%%' %(100 * outlier_car_age['TARGET'].mean()))
(outlier_car_age['OWN_CAR_AGE']).describe()


# In[ ]:


outlier_car_age['OWN_CAR_AGE'].plot.hist(title = 'OWN_CAR_AGE Outlier Histogram');
plt.xlabel('OWN_CAR_AGE')


# In[ ]:


# create a flag for outliers in the OWN_CAR_AGE column, and then replace these values with nan
dataset['OWN_CAR_AGE_outlier'] = dataset['OWN_CAR_AGE'] > 60
for i in dataset['OWN_CAR_AGE']:
    if i > 60:
        dataset['OWN_CAR_AGE'].replace({i: np.nan}, inplace = True)


# In[ ]:


# review OWN_CAR_AGE after our modifications
(dataset['OWN_CAR_AGE']).describe()


# In[ ]:


# now this data should look more like we expect
dataset['OWN_CAR_AGE'].plot.hist(title = 'OWN_CAR_AGE Histogram');
plt.xlabel('OWN_CAR_AGE')


# ## CNT_FAM_MEMBERS

# In[ ]:


(dataset['CNT_FAM_MEMBERS']).describe()


# In[ ]:


dataset['CNT_FAM_MEMBERS'].plot.hist(title = 'CNT_FAM_MEMBERS Histogram');
plt.xlabel('CNT_FAM_MEMBERS')


# In[ ]:


# it seems that there are some outliers on the count of family members, as the max is 21
# let's look at the 99th percentile
print(np.nanpercentile(dataset['CNT_FAM_MEMBERS'], 99))


# In[ ]:


# let's get a better look at the values in the tail and whether these have a higher probability of default than average
outlier_fam_mem = app_train[app_train['CNT_FAM_MEMBERS'] > 5]
print('count of outlier_fam_mem: ', len(outlier_fam_mem))
print('default probability of outlier_fam_mem: %0.2f%%' %(100 * outlier_fam_mem['TARGET'].mean()))
(outlier_fam_mem['CNT_FAM_MEMBERS']).describe()


# From the above, it appears that the probability of default for our outliers is 13.23%, which far exceeds that of the entire training data of 8.07%.  Let's remove these as outliers and keep track of which records are outliers.

# In[ ]:


# create a flag for outliers in the CNT_FAM_MEMBERS column, and then replace these values with nan
dataset['CNT_FAM_MEMBERS_outlier'] = dataset['CNT_FAM_MEMBERS'] > 5
for i in dataset['CNT_FAM_MEMBERS']:
    if i > 5:
        dataset['CNT_FAM_MEMBERS'].replace({i: np.nan}, inplace = True)


# In[ ]:


# review CNT_FAM_MEMBERS after our modifications
(dataset['CNT_FAM_MEMBERS']).describe()


# In[ ]:


dataset['CNT_FAM_MEMBERS'].plot.hist(title = 'CNT_FAM_MEMBERS Histogram');
plt.xlabel('CNT_FAM_MEMBERS')


# ## REGION_RATING_CLIENT

# In[ ]:


(dataset['REGION_RATING_CLIENT']).describe()


# In[ ]:


dataset['REGION_RATING_CLIENT'].plot.hist(title = 'REGION_RATING_CLIENT Histogram');
plt.xlabel('REGION_RATING_CLIENT')


# ## REGION_RATING_CLIENT_W_CITY

# In[ ]:


(dataset['REGION_RATING_CLIENT_W_CITY']).describe()


# In[ ]:


dataset['REGION_RATING_CLIENT_W_CITY'].plot.hist(title = 'REGION_RATING_CLIENT_W_CITY Histogram');
plt.xlabel('REGION_RATING_CLIENT_W_CITY')


# In[ ]:


# how many are equal to -1 in the dataset?
dataset['REGION_RATING_CLIENT_W_CITY'].map(lambda s: 1 if s == -1 else 0).sum()


# In[ ]:


# this appears to be a data entry error
# let's set the value of -1 equal to 1 instead
for i in dataset['REGION_RATING_CLIENT_W_CITY']:
    if i == -1:
        dataset['REGION_RATING_CLIENT_W_CITY'].replace({i: 1}, inplace = True)


# In[ ]:


(dataset['REGION_RATING_CLIENT_W_CITY']).describe()


# ## HOUR_APPR_PROCESS_START

# In[ ]:


(dataset['HOUR_APPR_PROCESS_START']).describe()


# In[ ]:


dataset['HOUR_APPR_PROCESS_START'].plot.hist(title = 'HOUR_APPR_PROCESS_START Histogram');
plt.xlabel('HOUR_APPR_PROCESS_START')


# ### Let's review the correlation matrix for the additional variables examined so far:

# In[ ]:


g = sns.heatmap(app_train[['TARGET','OWN_CAR_AGE','CNT_FAM_MEMBERS','REGION_RATING_CLIENT','REGION_RATING_CLIENT_W_CITY','HOUR_APPR_PROCESS_START']].corr(),annot=True, fmt = ".2f", cmap = "coolwarm")


# REGION_RATING_CLIENT and REGION_RATING_CLIENT_W_CITY are the most correlated with our TARGET, though these two variables are highly correlated with each other.  We will probably want to include only one of these two.

# ## EXT_SOURCE_1

# In[ ]:


(dataset['EXT_SOURCE_1']).describe()


# In[ ]:


# this data looks pretty good from the above, check the histogram
dataset['EXT_SOURCE_1'].plot.hist(title = 'EXT_SOURCE_1 Histogram');
plt.xlabel('EXT_SOURCE_1')


# ## EXT_SOURCE_2

# In[ ]:


(dataset['EXT_SOURCE_2']).describe()


# In[ ]:


# this data also looks pretty good from the above, check the histogram
dataset['EXT_SOURCE_2'].plot.hist(title = 'EXT_SOURCE_2 Histogram');
plt.xlabel('EXT_SOURCE_2')


# ## EXT_SOURCE_3

# In[ ]:


(dataset['EXT_SOURCE_3']).describe()


# In[ ]:


# this doesn't appear to have issues either, check the histogram
dataset['EXT_SOURCE_3'].plot.hist(title = 'EXT_SOURCE_3 Histogram');
plt.xlabel('EXT_SOURCE_3')


# These three external sources variables appear similar but have different distributions.  Let's look at their correlations with each other and with our TARGET.

# In[ ]:


g = sns.heatmap(app_train[['TARGET','EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3']].corr(),annot=True, fmt = ".2f", cmap = "coolwarm")


# These three are all strongly correlated with our TARGET, and they are not too highly correlated with each other.  These look like important variables for later.

# ## OBS_30_CNT_SOCIAL_CIRCLE

# In[ ]:


(dataset['OBS_30_CNT_SOCIAL_CIRCLE']).describe(percentiles=[0.25,0.5,0.75,0.99,0.999,0.9999,0.99999])


# In[ ]:


# there appear to be some outliers here we may need to deal with (max = 354??)
dataset['OBS_30_CNT_SOCIAL_CIRCLE'].plot.hist(title = 'OBS_30_CNT_SOCIAL_CIRCLE Histogram');
plt.xlabel('OBS_30_CNT_SOCIAL_CIRCLE')


# In[ ]:


# let's look more at these outliers and whether these have a higher probability of default than average
outlier_obs_30_social = app_train[app_train['OBS_30_CNT_SOCIAL_CIRCLE'] > 17]
print('count of outlier_obs_30_social: ', len(outlier_obs_30_social))
print('default probability of outlier_obs_30_social: %0.2f%%' %(100 * outlier_obs_30_social['TARGET'].mean()))
(outlier_obs_30_social['OBS_30_CNT_SOCIAL_CIRCLE']).describe()


# In[ ]:


# create a flag for outliers in the OBS_30_CNT_SOCIAL_CIRCLE column, and then replace these values with nan
dataset['OBS_30_CNT_SOCIAL_CIRCLE_outlier'] = dataset['OBS_30_CNT_SOCIAL_CIRCLE'] > 17
for i in dataset['OBS_30_CNT_SOCIAL_CIRCLE']:
    if i > 17:
        dataset['OBS_30_CNT_SOCIAL_CIRCLE'].replace({i: np.nan}, inplace = True)


# In[ ]:


# review OBS_30_CNT_SOCIAL_CIRCLE after our modifications
(dataset['OBS_30_CNT_SOCIAL_CIRCLE']).describe()


# In[ ]:


dataset['OBS_30_CNT_SOCIAL_CIRCLE'].plot.hist(title = 'OBS_30_CNT_SOCIAL_CIRCLE Histogram');
plt.xlabel('OBS_30_CNT_SOCIAL_CIRCLE')


# ## DEF_30_CNT_SOCIAL_CIRCLE

# In[ ]:


(dataset['DEF_30_CNT_SOCIAL_CIRCLE']).describe(percentiles=[0.25,0.5,0.75,0.99,0.999,0.9999,0.99999])


# In[ ]:


dataset['DEF_30_CNT_SOCIAL_CIRCLE'].plot.hist(title = 'DEF_30_CNT_SOCIAL_CIRCLE Histogram');
plt.xlabel('DEF_30_CNT_SOCIAL_CIRCLE')


# In[ ]:


outlier_def_30_social = app_train[app_train['DEF_30_CNT_SOCIAL_CIRCLE'] > 5]
print('count of outlier_def_30_social: ', len(outlier_def_30_social))
print('default probability of outlier_def_30_social: %0.2f%%' %(100 * outlier_def_30_social['TARGET'].mean()))
(outlier_def_30_social['DEF_30_CNT_SOCIAL_CIRCLE']).describe()


# In[ ]:


dataset['DEF_30_CNT_SOCIAL_CIRCLE_outlier'] = dataset['DEF_30_CNT_SOCIAL_CIRCLE'] > 5
for i in dataset['DEF_30_CNT_SOCIAL_CIRCLE']:
    if i > 5:
        dataset['DEF_30_CNT_SOCIAL_CIRCLE'].replace({i: np.nan}, inplace = True)


# In[ ]:


(dataset['DEF_30_CNT_SOCIAL_CIRCLE']).describe()


# In[ ]:


dataset['DEF_30_CNT_SOCIAL_CIRCLE'].plot.hist(title = 'DEF_30_CNT_SOCIAL_CIRCLE Histogram');
plt.xlabel('DEF_30_CNT_SOCIAL_CIRCLE')


# ## OBS_60_CNT_SOCIAL_CIRCLE

# In[ ]:


(dataset['OBS_60_CNT_SOCIAL_CIRCLE']).describe(percentiles=[0.25,0.5,0.75,0.99,0.999,0.9999,0.99999])


# In[ ]:


dataset['OBS_60_CNT_SOCIAL_CIRCLE'].plot.hist(title = 'OBS_60_CNT_SOCIAL_CIRCLE Histogram');
plt.xlabel('OBS_60_CNT_SOCIAL_CIRCLE')


# In[ ]:


outlier_obs_60_social = app_train[app_train['OBS_60_CNT_SOCIAL_CIRCLE'] > 16]
print('count of outlier_obs_60_social: ', len(outlier_obs_60_social))
print('default probability of outlier_obs_60_social: %0.2f%%' %(100 * outlier_obs_60_social['TARGET'].mean()))
(outlier_obs_60_social['OBS_60_CNT_SOCIAL_CIRCLE']).describe()


# In[ ]:


dataset['OBS_60_CNT_SOCIAL_CIRCLE_outlier'] = dataset['OBS_60_CNT_SOCIAL_CIRCLE'] > 16
for i in dataset['OBS_60_CNT_SOCIAL_CIRCLE']:
    if i > 16:
        dataset['OBS_60_CNT_SOCIAL_CIRCLE'].replace({i: np.nan}, inplace = True)


# In[ ]:


(dataset['OBS_60_CNT_SOCIAL_CIRCLE']).describe()


# In[ ]:


dataset['OBS_60_CNT_SOCIAL_CIRCLE'].plot.hist(title = 'OBS_60_CNT_SOCIAL_CIRCLE Histogram');
plt.xlabel('OBS_60_CNT_SOCIAL_CIRCLE')


# ## DEF_60_CNT_SOCIAL_CIRCLE

# In[ ]:


(dataset['DEF_60_CNT_SOCIAL_CIRCLE']).describe(percentiles=[0.25,0.5,0.75,0.99,0.999,0.9999,0.99999])


# In[ ]:


dataset['DEF_60_CNT_SOCIAL_CIRCLE'].plot.hist(title = 'DEF_60_CNT_SOCIAL_CIRCLE Histogram');
plt.xlabel('DEF_60_CNT_SOCIAL_CIRCLE')


# In[ ]:


outlier_def_60_social = app_train[app_train['DEF_60_CNT_SOCIAL_CIRCLE'] > 4]
print('count of outlier_def_60_social: ', len(outlier_def_60_social))
print('default probability of outlier_def_60_social: %0.2f%%' %(100 * outlier_def_60_social['TARGET'].mean()))
(outlier_def_60_social['DEF_60_CNT_SOCIAL_CIRCLE']).describe()


# In[ ]:


dataset['DEF_60_CNT_SOCIAL_CIRCLE_outlier'] = dataset['DEF_60_CNT_SOCIAL_CIRCLE'] > 4
for i in dataset['DEF_60_CNT_SOCIAL_CIRCLE']:
    if i > 4:
        dataset['DEF_60_CNT_SOCIAL_CIRCLE'].replace({i: np.nan}, inplace = True)


# In[ ]:


(dataset['DEF_60_CNT_SOCIAL_CIRCLE']).describe()


# In[ ]:


dataset['DEF_60_CNT_SOCIAL_CIRCLE'].plot.hist(title = 'DEF_60_CNT_SOCIAL_CIRCLE Histogram');
plt.xlabel('DEF_60_CNT_SOCIAL_CIRCLE')


# Let's look at the correlations between these social variables and our TARGET.

# In[ ]:


g = sns.heatmap(app_train[['TARGET','OBS_30_CNT_SOCIAL_CIRCLE','DEF_30_CNT_SOCIAL_CIRCLE','OBS_60_CNT_SOCIAL_CIRCLE','DEF_60_CNT_SOCIAL_CIRCLE']].corr(),annot=True, fmt = ".2f", cmap = "coolwarm")


# The DEF counts are highly correlated with each other, but none of these are very correlated with our TARGET.

# ## DAYS_LAST_PHONE_CHANGE

# In[ ]:


(dataset['DAYS_LAST_PHONE_CHANGE']).describe()


# In[ ]:


# let's transform this into positive years, as we did with the other DAYS_ variables above
(dataset['DAYS_LAST_PHONE_CHANGE'] / -365.25).describe()


# In[ ]:


(dataset['DAYS_LAST_PHONE_CHANGE'] / -365.25).plot.hist(title = 'DAYS_LAST_PHONE_CHANGE Histogram');
plt.xlabel('DAYS_LAST_PHONE_CHANGE')


# I am going to treat all of these credit bureau variables similarly.  I am going to remove the outliers but create a flag for the outliers.  As the period of evaluation increases (from hour to day to week, etc.), the cutoff for the outliers will also increase.

# ## AMT_REQ_CREDIT_BUREAU_HOUR

# In[ ]:


(dataset['AMT_REQ_CREDIT_BUREAU_HOUR']).describe()


# In[ ]:


dataset['AMT_REQ_CREDIT_BUREAU_HOUR'].plot.hist(title = 'AMT_REQ_CREDIT_BUREAU_HOUR Histogram');
plt.xlabel('AMT_REQ_CREDIT_BUREAU_HOUR')


# In[ ]:


dataset['AMT_REQ_CREDIT_BUREAU_HOUR_outlier'] = dataset['AMT_REQ_CREDIT_BUREAU_HOUR'] > 1
for i in dataset['AMT_REQ_CREDIT_BUREAU_HOUR']:
    if i > 1:
        dataset['AMT_REQ_CREDIT_BUREAU_HOUR'].replace({i: np.nan}, inplace = True)


# ## AMT_REQ_CREDIT_BUREAU_DAY

# In[ ]:


(dataset['AMT_REQ_CREDIT_BUREAU_DAY']).describe()


# In[ ]:


dataset['AMT_REQ_CREDIT_BUREAU_DAY'].plot.hist(title = 'AMT_REQ_CREDIT_BUREAU_DAY Histogram');
plt.xlabel('AMT_REQ_CREDIT_BUREAU_DAY')


# In[ ]:


dataset['AMT_REQ_CREDIT_BUREAU_DAY_outlier'] = dataset['AMT_REQ_CREDIT_BUREAU_DAY'] > 2
for i in dataset['AMT_REQ_CREDIT_BUREAU_DAY']:
    if i > 2:
        dataset['AMT_REQ_CREDIT_BUREAU_DAY'].replace({i: np.nan}, inplace = True)


# ## AMT_REQ_CREDIT_BUREAU_WEEK

# In[ ]:


(dataset['AMT_REQ_CREDIT_BUREAU_WEEK']).describe()


# In[ ]:


dataset['AMT_REQ_CREDIT_BUREAU_WEEK'].plot.hist(title = 'AMT_REQ_CREDIT_BUREAU_WEEK Histogram');
plt.xlabel('AMT_REQ_CREDIT_BUREAU_WEEK')


# In[ ]:


dataset['AMT_REQ_CREDIT_BUREAU_WEEK_outlier'] = dataset['AMT_REQ_CREDIT_BUREAU_WEEK'] > 2
for i in dataset['AMT_REQ_CREDIT_BUREAU_WEEK']:
    if i > 2:
        dataset['AMT_REQ_CREDIT_BUREAU_WEEK'].replace({i: np.nan}, inplace = True)


# ## AMT_REQ_CREDIT_BUREAU_MON

# In[ ]:


(dataset['AMT_REQ_CREDIT_BUREAU_MON']).describe()


# In[ ]:


dataset['AMT_REQ_CREDIT_BUREAU_MON'].plot.hist(title = 'AMT_REQ_CREDIT_BUREAU_MON Histogram');
plt.xlabel('AMT_REQ_CREDIT_BUREAU_MON')


# In[ ]:


dataset['AMT_REQ_CREDIT_BUREAU_MON_outlier'] = dataset['AMT_REQ_CREDIT_BUREAU_MON'] > 5
for i in dataset['AMT_REQ_CREDIT_BUREAU_MON']:
    if i > 5:
        dataset['AMT_REQ_CREDIT_BUREAU_MON'].replace({i: np.nan}, inplace = True)


# ## AMT_REQ_CREDIT_BUREAU_QRT

# In[ ]:


(dataset['AMT_REQ_CREDIT_BUREAU_QRT']).describe()


# In[ ]:


dataset['AMT_REQ_CREDIT_BUREAU_QRT'].plot.hist(title = 'AMT_REQ_CREDIT_BUREAU_QRT Histogram');
plt.xlabel('AMT_REQ_CREDIT_BUREAU_QRT')


# In[ ]:


dataset['AMT_REQ_CREDIT_BUREAU_QRT_outlier'] = dataset['AMT_REQ_CREDIT_BUREAU_QRT'] > 5
for i in dataset['AMT_REQ_CREDIT_BUREAU_QRT']:
    if i > 5:
        dataset['AMT_REQ_CREDIT_BUREAU_QRT'].replace({i: np.nan}, inplace = True)


# ## AMT_REQ_CREDIT_BUREAU_YEAR

# In[ ]:


(dataset['AMT_REQ_CREDIT_BUREAU_YEAR']).describe()


# In[ ]:


dataset['AMT_REQ_CREDIT_BUREAU_YEAR'].plot.hist(title = 'AMT_REQ_CREDIT_BUREAU_YEAR Histogram');
plt.xlabel('AMT_REQ_CREDIT_BUREAU_YEAR')


# In[ ]:


dataset['AMT_REQ_CREDIT_BUREAU_YEAR_outlier'] = dataset['AMT_REQ_CREDIT_BUREAU_YEAR'] > 10
for i in dataset['AMT_REQ_CREDIT_BUREAU_YEAR']:
    if i > 10:
        dataset['AMT_REQ_CREDIT_BUREAU_YEAR'].replace({i: np.nan}, inplace = True)


# Let's look at the correlations of this last group of variables.

# In[ ]:


g = sns.heatmap(app_train[['TARGET','DAYS_LAST_PHONE_CHANGE','AMT_REQ_CREDIT_BUREAU_HOUR','AMT_REQ_CREDIT_BUREAU_DAY','AMT_REQ_CREDIT_BUREAU_WEEK','AMT_REQ_CREDIT_BUREAU_MON','AMT_REQ_CREDIT_BUREAU_QRT','AMT_REQ_CREDIT_BUREAU_YEAR']].corr(),annot=True, fmt = ".2f", cmap = "coolwarm")


# DAYS_LAST_PHONE_CHANGE is more correlated with our TARGET, while the credit bureau variables don't seem to be correlated with TARGET much at all.

# # Additional Features to Add from the Training Data

# Now that we have reviewed the initial data provided, there are a few more variables we can create from the existing data.  We can later determine if any of these interactions can add to our model.
# 
# New variables to try:
# * EMPLOY_AGE = DAYS_EMPLOYED / DAYS_BIRTH: how long was the applicant employed relative to how old the applicant was - employed for a larger portion may indicate reliability
# * INCOME_AGE = AMT_INCOME_TOTAL / DAYS_BIRTH: how large is the income relative to how old the applicant was - may indicate potential for income to rise and may repayment easier in the future
# * CREDIT_AGE = AMT_CREDIT / DAYS_BIRTH: how much credit relative to how old the applicant was - may indicate sources of other financial stress
# * CREDIT_INCOME = AMT_CREDIT / AMT_INCOME_TOTAL: how much credit relative to total income - too much credit may be too risky
# * ANNUITY_INCOME = AMT_ANNUITY / AMT_INCOME_TOTAL: how large are the loan payments relative to total income - too large of payments may not be sustainable
# * ANNUITY_CREDIT = AMT_ANNUITY / AMT_CREDIT: how large are the loan payments relative to the credit amount (how long will it take to pay it back, without accounting for different interest rates)
# 
# These may or may not be good predictors, but these were the ones I thought could be useful in the model.

# In[ ]:


# create new variables
dataset['EMPLOY_AGE'] = dataset['DAYS_EMPLOYED'] / dataset['DAYS_BIRTH']
dataset['INCOME_AGE'] = dataset['AMT_INCOME_TOTAL'] / dataset['DAYS_BIRTH']
dataset['CREDIT_AGE'] = dataset['AMT_CREDIT'] / dataset['DAYS_BIRTH']
dataset['CREDIT_INCOME'] = dataset['AMT_CREDIT'] / dataset['AMT_INCOME_TOTAL']
dataset['ANNUITY_INCOME'] = dataset['AMT_ANNUITY'] / dataset['AMT_INCOME_TOTAL']
dataset['ANNUITY_CREDIT'] = dataset['AMT_ANNUITY'] / dataset['AMT_CREDIT']


# In[ ]:


# let's look at the correlations of the new variables we created along with TARGET
g = sns.heatmap(dataset[['TARGET','EMPLOY_AGE','INCOME_AGE','CREDIT_AGE','CREDIT_INCOME','ANNUITY_INCOME','ANNUITY_CREDIT']].corr(),annot=True, fmt = ".2f", cmap = "coolwarm")


# In[ ]:


# EMPLOY_AGE seems to be most correlated with TARGET of our new variables
# we can plot EMPLOY_AGE relative to TARGET using KDE
plt.figure(figsize = (8, 6))
sns.kdeplot(dataset.loc[dataset['TARGET'] == 0, 'EMPLOY_AGE'], label = 'TARGET == 0')
sns.kdeplot(dataset.loc[dataset['TARGET'] == 1, 'EMPLOY_AGE'], label = 'TARGET == 1')
plt.xlabel('EMPLOY_AGE'); plt.ylabel('Density'); plt.title('KDE of EMPLOY_AGE');


# # Adding Features from the Other Data Files

# In this section, I will begin adding to my data, incorporating the information from the other data files we were provided.

# In[ ]:


# the first file we will investigate is bureau
bureau = pd.read_csv('../input/bureau.csv')
bureau.head()


# In[ ]:


print('bureau data shape: ', bureau.shape)


# In[ ]:


bureau.describe()


# I will need to determine how to incorporate each of these items into the data.  Generally, for time measurements (numbers of days since something), I will want to use the max or min.  For other items, like amounts, I will want to use the mean instead.  I may also want to use count or sum, depending on the item.
# 
# To get this data into the main dataset file, I will need to group the data in the new file by SK_ID_CURR.  I will then apply the max or mean (or other function) to the data and merge this into the dataset file.

# ## COUNT

# In[ ]:


# the first item to look at is how many records are in this for each applicant
BUREAU_count = bureau.groupby('SK_ID_CURR', as_index=False)['SK_ID_BUREAU'].count().rename(columns = {'SK_ID_BUREAU': 'bureau_count'})
(BUREAU_count['bureau_count']).describe()


# In[ ]:


dataset = dataset.merge(BUREAU_count, on = 'SK_ID_CURR', how = 'left')
corr = dataset['TARGET'].corr(dataset['bureau_count'])
print('%.4f' % corr)


# ## DAYS_CREDIT

# In[ ]:


# review the data, and divide by -365.25 to turn this into positive years
(bureau['DAYS_CREDIT'] / -365.25).describe()


# In[ ]:


(bureau['DAYS_CREDIT'] / -365.25).plot.hist(title = 'DAYS_CREDIT Histogram');
plt.xlabel('DAYS_CREDIT')


# In[ ]:


DAYS_CREDIT_max = bureau.groupby('SK_ID_CURR', as_index=False)['DAYS_CREDIT'].max().rename(columns = {'DAYS_CREDIT': 'bureau_DAYS_CREDIT_max'})
DAYS_CREDIT_max.head()


# In[ ]:


# merge with the dataset
dataset = dataset.merge(DAYS_CREDIT_max, on = 'SK_ID_CURR', how = 'left')


# In[ ]:


# what is the correlation of our new variable with TARGET
corr = dataset['TARGET'].corr(dataset['bureau_DAYS_CREDIT_max'])
print('%.4f' % corr)


# In[ ]:


# evaluate the new variable with a KDE plot
plt.figure(figsize = (8, 6))
sns.kdeplot(dataset.loc[dataset['TARGET'] == 0, 'bureau_DAYS_CREDIT_max'], label = 'TARGET == 0')
sns.kdeplot(dataset.loc[dataset['TARGET'] == 1, 'bureau_DAYS_CREDIT_max'], label = 'TARGET == 1')
plt.xlabel('bureau_DAYS_CREDIT_max'); plt.ylabel('Density'); plt.title('KDE of bureau_DAYS_CREDIT_max');


# ## CREDIT_DAY_OVERDUE

# In[ ]:


(bureau['CREDIT_DAY_OVERDUE']).describe([.25, .5, .75, .9, .99, .999])


# In[ ]:


# this looks like virtually all are zero, but there are some outliers
bureau['CREDIT_DAY_OVERDUE'].plot.hist(title = 'CREDIT_DAY_OVERDUE Histogram');
plt.xlabel('CREDIT_DAY_OVERDUE')


# In[ ]:


# let's take the max of this variable
CREDIT_DAY_OVERDUE_max = bureau.groupby('SK_ID_CURR', as_index=False)['CREDIT_DAY_OVERDUE'].max().rename(columns = {'CREDIT_DAY_OVERDUE': 'bureau_CREDIT_DAY_OVERDUE_max'})


# In[ ]:


(CREDIT_DAY_OVERDUE_max['bureau_CREDIT_DAY_OVERDUE_max']).describe([.25, .5, .75, .9, .99, .999])


# In[ ]:


# most of the data in this column is zero
# how many non-zero items exist?
CREDIT_DAY_OVERDUE_max[CREDIT_DAY_OVERDUE_max['bureau_CREDIT_DAY_OVERDUE_max'] > 0].count()


# In[ ]:


# let's turn this into a flag, since 99% of the data is zero
CREDIT_DAY_OVERDUE_max['bureau_CREDIT_DAY_OVERDUE_max_flag'] = CREDIT_DAY_OVERDUE_max['bureau_CREDIT_DAY_OVERDUE_max'].where(CREDIT_DAY_OVERDUE_max['bureau_CREDIT_DAY_OVERDUE_max']==0,other=1)


# In[ ]:


# drop the max variable and merge in the flag
CREDIT_DAY_OVERDUE_max = CREDIT_DAY_OVERDUE_max.drop('bureau_CREDIT_DAY_OVERDUE_max', axis=1)
dataset = dataset.merge(CREDIT_DAY_OVERDUE_max, on = 'SK_ID_CURR', how = 'left')


# In[ ]:


corr = dataset['TARGET'].corr(dataset['bureau_CREDIT_DAY_OVERDUE_max_flag'])
print('%.4f' % corr)


# ## DAYS_CREDIT_ENDDATE

# In[ ]:


(bureau['DAYS_CREDIT_ENDDATE']).describe()


# In[ ]:


bureau['DAYS_CREDIT_ENDDATE'].plot.hist(title = 'DAYS_CREDIT_ENDDATE Histogram');
plt.xlabel('DAYS_CREDIT_ENDDATE')


# In[ ]:


# let's take the max of this variable
DAYS_CREDIT_ENDDATE_max = bureau.groupby('SK_ID_CURR', as_index=False)['DAYS_CREDIT_ENDDATE'].max().rename(columns = {'DAYS_CREDIT_ENDDATE': 'bureau_DAYS_CREDIT_ENDDATE_max'})
(DAYS_CREDIT_ENDDATE_max['bureau_DAYS_CREDIT_ENDDATE_max']).describe()


# In[ ]:


# it appears that we have a few outliers around -41875
DAYS_CREDIT_ENDDATE_max['bureau_DAYS_CREDIT_ENDDATE_max_outlier'] = DAYS_CREDIT_ENDDATE_max['bureau_DAYS_CREDIT_ENDDATE_max'] < -10000
for i in DAYS_CREDIT_ENDDATE_max['bureau_DAYS_CREDIT_ENDDATE_max']:
    if i < -10000:
        DAYS_CREDIT_ENDDATE_max['bureau_DAYS_CREDIT_ENDDATE_max'].replace({i: np.nan}, inplace = True)


# In[ ]:


# merge both our max variable and outlier flag into the dataset
dataset = dataset.merge(DAYS_CREDIT_ENDDATE_max, on = 'SK_ID_CURR', how = 'left')


# In[ ]:


corr = dataset['TARGET'].corr(dataset['bureau_DAYS_CREDIT_ENDDATE_max'])
print('%.4f' % corr)


# ## DAYS_ENDDATE_FACT

# In[ ]:


(bureau['DAYS_ENDDATE_FACT'] / -365.25).describe()


# In[ ]:


(bureau['DAYS_ENDDATE_FACT'] / -365.25).plot.hist(title = 'DAYS_ENDDATE_FACT Histogram');
plt.xlabel('DAYS_ENDDATE_FACT')


# In[ ]:


# let's take the average of this variable
DAYS_ENDDATE_FACT_mean = bureau.groupby('SK_ID_CURR', as_index=False)['DAYS_ENDDATE_FACT'].mean().rename(columns = {'DAYS_ENDDATE_FACT': 'bureau_DAYS_ENDDATE_FACT_mean'})
(DAYS_ENDDATE_FACT_mean['bureau_DAYS_ENDDATE_FACT_mean']).describe()


# In[ ]:


(DAYS_ENDDATE_FACT_mean['bureau_DAYS_ENDDATE_FACT_mean']).plot.hist(title = 'bureau_DAYS_ENDDATE_FACT_mean Histogram');
plt.xlabel('bureau_DAYS_ENDDATE_FACT_mean')


# In[ ]:


# it appears that we have a few outliers around -8000 days that we can handle
DAYS_ENDDATE_FACT_mean['bureau_DAYS_ENDDATE_FACT_mean_outlier'] = DAYS_ENDDATE_FACT_mean['bureau_DAYS_ENDDATE_FACT_mean'] < -4000
for i in DAYS_ENDDATE_FACT_mean['bureau_DAYS_ENDDATE_FACT_mean']:
    if i < -4000:
        DAYS_ENDDATE_FACT_mean['bureau_DAYS_ENDDATE_FACT_mean'].replace({i: np.nan}, inplace = True)


# In[ ]:


# merge both our mean variable and outlier flag into the dataset
dataset = dataset.merge(DAYS_ENDDATE_FACT_mean, on = 'SK_ID_CURR', how = 'left')


# In[ ]:


corr = dataset['TARGET'].corr(dataset['bureau_DAYS_ENDDATE_FACT_mean'])
print('%.4f' % corr)


# In[ ]:


# evaluate the new variable with a KDE plot
plt.figure(figsize = (8, 6))
sns.kdeplot(dataset.loc[dataset['TARGET'] == 0, 'bureau_DAYS_ENDDATE_FACT_mean'], label = 'TARGET == 0')
sns.kdeplot(dataset.loc[dataset['TARGET'] == 1, 'bureau_DAYS_ENDDATE_FACT_mean'], label = 'TARGET == 1')
plt.xlabel('bureau_DAYS_ENDDATE_FACT_mean'); plt.ylabel('Density'); plt.title('KDE of bureau_DAYS_ENDDATE_FACT_mean');


# ## AMT_CREDIT_MAX_OVERDUE

# In[ ]:


(bureau['AMT_CREDIT_MAX_OVERDUE']).describe()


# In[ ]:


# let's take the max of this variable
AMT_CREDIT_MAX_OVERDUE_max = bureau.groupby('SK_ID_CURR', as_index=False)['AMT_CREDIT_MAX_OVERDUE'].max().rename(columns = {'AMT_CREDIT_MAX_OVERDUE': 'bureau_AMT_CREDIT_MAX_OVERDUE_max'})
(AMT_CREDIT_MAX_OVERDUE_max['bureau_AMT_CREDIT_MAX_OVERDUE_max']).describe()


# In[ ]:


# I'm also curious on the average of this variable
AMT_CREDIT_MAX_OVERDUE_mean = bureau.groupby('SK_ID_CURR', as_index=False)['AMT_CREDIT_MAX_OVERDUE'].mean().rename(columns = {'AMT_CREDIT_MAX_OVERDUE': 'bureau_AMT_CREDIT_MAX_OVERDUE_mean'})
(AMT_CREDIT_MAX_OVERDUE_mean['bureau_AMT_CREDIT_MAX_OVERDUE_mean']).describe()


# In[ ]:


# I'm not sure which of these two variables may work better in this case, so let's bring them both into the dataset for now
dataset = dataset.merge(AMT_CREDIT_MAX_OVERDUE_max, on = 'SK_ID_CURR', how = 'left')
dataset = dataset.merge(AMT_CREDIT_MAX_OVERDUE_mean, on = 'SK_ID_CURR', how = 'left')


# In[ ]:


corr_max = dataset['TARGET'].corr(dataset['bureau_AMT_CREDIT_MAX_OVERDUE_max'])
corr_mean = dataset['TARGET'].corr(dataset['bureau_AMT_CREDIT_MAX_OVERDUE_mean'])
print('correlation for max variable: %.4f' % corr_max)
print('correlation for mean variable: %.4f' % corr_mean)


# These two are very similarly correlated with our TARGET.  For now, we will keep both of these, but we may remove one later on.

# ## CNT_CREDIT_PROLONG

# In[ ]:


(bureau['CNT_CREDIT_PROLONG']).describe()


# In[ ]:


# since these are counts, let's sum this variable
CNT_CREDIT_PROLONG_sum = bureau.groupby('SK_ID_CURR', as_index=False)['CNT_CREDIT_PROLONG'].sum().rename(columns = {'CNT_CREDIT_PROLONG': 'bureau_CNT_CREDIT_PROLONG_sum'})
(CNT_CREDIT_PROLONG_sum['bureau_CNT_CREDIT_PROLONG_sum']).describe()


# In[ ]:


# merge into our dataset
dataset = dataset.merge(CNT_CREDIT_PROLONG_sum, on = 'SK_ID_CURR', how = 'left')


# In[ ]:


corr = dataset['TARGET'].corr(dataset['bureau_CNT_CREDIT_PROLONG_sum'])
print('%.4f' % corr)


# ## AMT_CREDIT_SUM

# In[ ]:


(bureau['AMT_CREDIT_SUM']).describe()


# In[ ]:


# since these are amounts, let's average this variable
AMT_CREDIT_SUM_mean = bureau.groupby('SK_ID_CURR', as_index=False)['AMT_CREDIT_SUM'].mean().rename(columns = {'AMT_CREDIT_SUM': 'bureau_AMT_CREDIT_SUM_mean'})
(AMT_CREDIT_SUM_mean['bureau_AMT_CREDIT_SUM_mean']).describe()


# In[ ]:


# merge into our dataset
dataset = dataset.merge(AMT_CREDIT_SUM_mean, on = 'SK_ID_CURR', how = 'left')


# In[ ]:


corr = dataset['TARGET'].corr(dataset['bureau_AMT_CREDIT_SUM_mean'])
print('%.4f' % corr)


# ## AMT_CREDIT_SUM_DEBT

# In[ ]:


(bureau['AMT_CREDIT_SUM_DEBT']).describe()


# In[ ]:


# since these are amounts, let's average this variable
AMT_CREDIT_SUM_DEBT_mean = bureau.groupby('SK_ID_CURR', as_index=False)['AMT_CREDIT_SUM_DEBT'].mean().rename(columns = {'AMT_CREDIT_SUM_DEBT': 'bureau_AMT_CREDIT_SUM_DEBT_mean'})
(AMT_CREDIT_SUM_DEBT_mean['bureau_AMT_CREDIT_SUM_DEBT_mean']).describe()


# In[ ]:


# merge into our dataset and look at the correlation
dataset = dataset.merge(AMT_CREDIT_SUM_DEBT_mean, on = 'SK_ID_CURR', how = 'left')
corr = dataset['TARGET'].corr(dataset['bureau_AMT_CREDIT_SUM_DEBT_mean'])
print('%.4f' % corr)


# ## AMT_CREDIT_SUM_LIMIT

# In[ ]:


(bureau['AMT_CREDIT_SUM_LIMIT']).describe()


# In[ ]:


# since these are amounts, let's average this variable
AMT_CREDIT_SUM_LIMIT_mean = bureau.groupby('SK_ID_CURR', as_index=False)['AMT_CREDIT_SUM_LIMIT'].mean().rename(columns = {'AMT_CREDIT_SUM_LIMIT': 'bureau_AMT_CREDIT_SUM_LIMIT_mean'})
(AMT_CREDIT_SUM_LIMIT_mean['bureau_AMT_CREDIT_SUM_LIMIT_mean']).describe()


# In[ ]:


# merge into our dataset and look at the correlation
dataset = dataset.merge(AMT_CREDIT_SUM_LIMIT_mean, on = 'SK_ID_CURR', how = 'left')
corr = dataset['TARGET'].corr(dataset['bureau_AMT_CREDIT_SUM_LIMIT_mean'])
print('%.4f' % corr)


# ## AMT_CREDIT_SUM_OVERDUE

# In[ ]:


(bureau['AMT_CREDIT_SUM_OVERDUE']).describe()


# In[ ]:


# since these are amounts, let's average this variable
AMT_CREDIT_SUM_OVERDUE_mean = bureau.groupby('SK_ID_CURR', as_index=False)['AMT_CREDIT_SUM_OVERDUE'].mean().rename(columns = {'AMT_CREDIT_SUM_OVERDUE': 'bureau_AMT_CREDIT_SUM_OVERDUE_mean'})
(AMT_CREDIT_SUM_OVERDUE_mean['bureau_AMT_CREDIT_SUM_OVERDUE_mean']).describe()


# In[ ]:


# merge into our dataset and look at the correlation
dataset = dataset.merge(AMT_CREDIT_SUM_OVERDUE_mean, on = 'SK_ID_CURR', how = 'left')
corr = dataset['TARGET'].corr(dataset['bureau_AMT_CREDIT_SUM_OVERDUE_mean'])
print('%.4f' % corr)


# ## DAYS_CREDIT_UPDATE

# In[ ]:


# divide by -365.25 to turn this into positive years
(bureau['DAYS_CREDIT_UPDATE'] / -365.25).describe()


# In[ ]:


# since this is a days variable, let's use max
DAYS_CREDIT_UPDATE_max = bureau.groupby('SK_ID_CURR', as_index=False)['DAYS_CREDIT_UPDATE'].max().rename(columns = {'DAYS_CREDIT_UPDATE': 'bureau_DAYS_CREDIT_UPDATE_max'})
(DAYS_CREDIT_UPDATE_max['bureau_DAYS_CREDIT_UPDATE_max']).describe()


# In[ ]:


# merge into our dataset and look at the correlation
dataset = dataset.merge(DAYS_CREDIT_UPDATE_max, on = 'SK_ID_CURR', how = 'left')
corr = dataset['TARGET'].corr(dataset['bureau_DAYS_CREDIT_UPDATE_max'])
print('%.4f' % corr)


# ## AMT_ANNUITY

# In[ ]:


(bureau['AMT_ANNUITY']).describe()


# In[ ]:


# since these are amounts, let's average this variable
AMT_ANNUITY_mean = bureau.groupby('SK_ID_CURR', as_index=False)['AMT_ANNUITY'].mean().rename(columns = {'AMT_ANNUITY': 'bureau_AMT_ANNUITY_mean'})
(AMT_ANNUITY_mean['bureau_AMT_ANNUITY_mean']).describe()


# In[ ]:


# merge into our dataset and look at the correlation
dataset = dataset.merge(AMT_ANNUITY_mean, on = 'SK_ID_CURR', how = 'left')
corr = dataset['TARGET'].corr(dataset['bureau_AMT_ANNUITY_mean'])
print('%.4f' % corr)


# ## Categoricals in Bureau data
# 
# Now let's look at the categorical variables that are left in this data.  There are three of them that we need to deal with.

# In[ ]:


bureau.describe(include=[np.object])


# In[ ]:


# let's use one-hot encoding on these variables
bureau_cats = pd.get_dummies(bureau.select_dtypes('object'))
bureau_cats['SK_ID_CURR'] = bureau['SK_ID_CURR']
bureau_cats.head()


# In[ ]:


bureau_cats_grouped = bureau_cats.groupby('SK_ID_CURR').agg('sum')
bureau_cats_grouped.head()


# In[ ]:


#merge into our dataset
dataset = dataset.merge(bureau_cats_grouped, on = 'SK_ID_CURR', right_index = True, how = 'left')
dataset.head()


# In[ ]:


# the next file we will investigate is bureau_balance
bureau_balance = pd.read_csv('../input/bureau_balance.csv')
bureau_balance.head()


# ## MONTHS_BALANCE

# In[ ]:


# this appears to be the number of months of balance relative to the application date
# let's start with the count
MONTHS_BALANCE_count = bureau_balance.groupby('SK_ID_BUREAU', as_index=False)['MONTHS_BALANCE'].count().rename(columns = {'MONTHS_BALANCE': 'bureau_bal_MONTHS_BALANCE_count'})
(MONTHS_BALANCE_count['bureau_bal_MONTHS_BALANCE_count']).describe()


# In[ ]:


# let's also look at the mean
MONTHS_BALANCE_mean = bureau_balance.groupby('SK_ID_BUREAU', as_index=False)['MONTHS_BALANCE'].mean().rename(columns = {'MONTHS_BALANCE': 'bureau_bal_MONTHS_BALANCE_mean'})
(MONTHS_BALANCE_mean['bureau_bal_MONTHS_BALANCE_mean']).describe()


# In[ ]:


MONTHS_BAL = MONTHS_BALANCE_mean.merge(MONTHS_BALANCE_count, on = 'SK_ID_BUREAU', right_index = True, how = 'inner')
MONTHS_BAL.head()


# In[ ]:


# now let's get our categoricals
bureau_bal_cats = pd.get_dummies(bureau_balance.select_dtypes('object'))
bureau_bal_cats['SK_ID_BUREAU'] = bureau_balance['SK_ID_BUREAU']
bureau_bal_cats.head()


# In[ ]:


bureau_bal_cats_grouped = bureau_bal_cats.groupby('SK_ID_BUREAU').agg('sum')
bureau_bal_cats_grouped.head()


# In[ ]:


# now let's merge the MONTHS_BAL with our categoricals by SK_ID_BUREAU, then merge with bureau to add in SK_ID_CURR
bureau_bal_merged = MONTHS_BAL.merge(bureau_bal_cats_grouped, right_index = True, left_on = 'SK_ID_BUREAU', how = 'outer')
bureau_bal_merged = bureau_bal_merged.merge(bureau[['SK_ID_BUREAU', 'SK_ID_CURR']], on = 'SK_ID_BUREAU', how = 'left')
bureau_bal_merged.head()


# Now we will take the mean when grouping by SK_ID_CURR for each of the above variables and then add them to our dataset.  There are more possibilities here (min, max, count, sum, etc.) but we will just do mean for now.

# In[ ]:


bureau_bal_MONTHS_BALANCE_mean_mean = bureau_bal_merged.groupby('SK_ID_CURR', as_index=False)['bureau_bal_MONTHS_BALANCE_mean'].mean().rename(columns = {'bureau_bal_MONTHS_BALANCE_mean': 'bureau_bal_MONTHS_BALANCE_mean_mean'})
dataset = dataset.merge(bureau_bal_MONTHS_BALANCE_mean_mean, on = 'SK_ID_CURR', how = 'left')
corr = dataset['TARGET'].corr(dataset['bureau_bal_MONTHS_BALANCE_mean_mean'])
print('%.4f' % corr)


# In[ ]:


bureau_bal_MONTHS_BALANCE_count_mean = bureau_bal_merged.groupby('SK_ID_CURR', as_index=False)['bureau_bal_MONTHS_BALANCE_count'].mean().rename(columns = {'bureau_bal_MONTHS_BALANCE_count': 'bureau_bal_MONTHS_BALANCE_count_mean'})
dataset = dataset.merge(bureau_bal_MONTHS_BALANCE_count_mean, on = 'SK_ID_CURR', how = 'left')
corr = dataset['TARGET'].corr(dataset['bureau_bal_MONTHS_BALANCE_count_mean'])
print('%.4f' % corr)


# In[ ]:


bureau_bal_STATUS_0_mean = bureau_bal_merged.groupby('SK_ID_CURR', as_index=False)['STATUS_0'].mean().rename(columns = {'STATUS_0': 'bureau_bal_STATUS_0_mean'})
dataset = dataset.merge(bureau_bal_STATUS_0_mean, on = 'SK_ID_CURR', how = 'left')
corr = dataset['TARGET'].corr(dataset['bureau_bal_STATUS_0_mean'])
print('%.4f' % corr)


# In[ ]:


bureau_bal_STATUS_1_mean = bureau_bal_merged.groupby('SK_ID_CURR', as_index=False)['STATUS_1'].mean().rename(columns = {'STATUS_1': 'bureau_bal_STATUS_1_mean'})
dataset = dataset.merge(bureau_bal_STATUS_1_mean, on = 'SK_ID_CURR', how = 'left')
corr = dataset['TARGET'].corr(dataset['bureau_bal_STATUS_1_mean'])
print('%.4f' % corr)


# In[ ]:


bureau_bal_STATUS_2_mean = bureau_bal_merged.groupby('SK_ID_CURR', as_index=False)['STATUS_2'].mean().rename(columns = {'STATUS_2': 'bureau_bal_STATUS_2_mean'})
dataset = dataset.merge(bureau_bal_STATUS_2_mean, on = 'SK_ID_CURR', how = 'left')
corr = dataset['TARGET'].corr(dataset['bureau_bal_STATUS_2_mean'])
print('%.4f' % corr)


# In[ ]:


bureau_bal_STATUS_3_mean = bureau_bal_merged.groupby('SK_ID_CURR', as_index=False)['STATUS_3'].mean().rename(columns = {'STATUS_3': 'bureau_bal_STATUS_3_mean'})
dataset = dataset.merge(bureau_bal_STATUS_3_mean, on = 'SK_ID_CURR', how = 'left')
corr = dataset['TARGET'].corr(dataset['bureau_bal_STATUS_3_mean'])
print('%.4f' % corr)


# In[ ]:


bureau_bal_STATUS_4_mean = bureau_bal_merged.groupby('SK_ID_CURR', as_index=False)['STATUS_4'].mean().rename(columns = {'STATUS_4': 'bureau_bal_STATUS_4_mean'})
dataset = dataset.merge(bureau_bal_STATUS_4_mean, on = 'SK_ID_CURR', how = 'left')
corr = dataset['TARGET'].corr(dataset['bureau_bal_STATUS_4_mean'])
print('%.4f' % corr)


# In[ ]:


bureau_bal_STATUS_5_mean = bureau_bal_merged.groupby('SK_ID_CURR', as_index=False)['STATUS_5'].mean().rename(columns = {'STATUS_5': 'bureau_bal_STATUS_5_mean'})
dataset = dataset.merge(bureau_bal_STATUS_5_mean, on = 'SK_ID_CURR', how = 'left')
corr = dataset['TARGET'].corr(dataset['bureau_bal_STATUS_5_mean'])
print('%.4f' % corr)


# In[ ]:


bureau_bal_STATUS_C_mean = bureau_bal_merged.groupby('SK_ID_CURR', as_index=False)['STATUS_C'].mean().rename(columns = {'STATUS_C': 'bureau_bal_STATUS_C_mean'})
dataset = dataset.merge(bureau_bal_STATUS_C_mean, on = 'SK_ID_CURR', how = 'left')
corr = dataset['TARGET'].corr(dataset['bureau_bal_STATUS_C_mean'])
print('%.4f' % corr)


# In[ ]:


bureau_bal_STATUS_X_mean = bureau_bal_merged.groupby('SK_ID_CURR', as_index=False)['STATUS_X'].mean().rename(columns = {'STATUS_X': 'bureau_bal_STATUS_X_mean'})
dataset = dataset.merge(bureau_bal_STATUS_X_mean, on = 'SK_ID_CURR', how = 'left')
corr = dataset['TARGET'].corr(dataset['bureau_bal_STATUS_X_mean'])
print('%.4f' % corr)


# In[ ]:


dataset.head()


# In[ ]:


# let's free up some memory by deleting some of the dataframes we are done with
gc.enable()
del bureau, BUREAU_count, DAYS_CREDIT_max, CREDIT_DAY_OVERDUE_max, DAYS_CREDIT_ENDDATE_max, DAYS_ENDDATE_FACT_mean, AMT_CREDIT_MAX_OVERDUE_max, 
AMT_CREDIT_MAX_OVERDUE_mean, CNT_CREDIT_PROLONG_sum, AMT_CREDIT_SUM_mean, AMT_CREDIT_SUM_DEBT_mean, AMT_CREDIT_SUM_LIMIT_mean, AMT_CREDIT_SUM_OVERDUE_mean, 
DAYS_CREDIT_UPDATE_max, AMT_ANNUITY_mean, bureau_cats, bureau_cats_grouped, bureau_balance, MONTHS_BALANCE_count, MONTHS_BALANCE_mean, MONTHS_BAL, bureau_bal_cats, 
bureau_bal_cats_grouped, bureau_bal_merged, bureau_bal_MONTHS_BALANCE_mean_mean, bureau_bal_MONTHS_BALANCE_count_mean, bureau_bal_STATUS_0_mean, 
bureau_bal_STATUS_1_mean, bureau_bal_STATUS_2_mean, bureau_bal_STATUS_3_mean, bureau_bal_STATUS_4_mean, bureau_bal_STATUS_5_mean, bureau_bal_STATUS_C_mean, 
bureau_bal_STATUS_X_mean
gc.collect()


# In[ ]:


# the next file we will investigate is credit_card_balance
credit = pd.read_csv('../input/credit_card_balance.csv')
credit.head()


# In[ ]:


credit_stats_by_prev = credit[['SK_ID_PREV', 'SK_ID_CURR']]


# In[ ]:


credit_MONTHS_BALANCE_count = credit.groupby('SK_ID_PREV', as_index=False)['MONTHS_BALANCE'].count().rename(columns = {'MONTHS_BALANCE': 'credit_MONTHS_BALANCE_count'})
credit_MONTHS_BALANCE_mean = credit.groupby('SK_ID_PREV', as_index=False)['MONTHS_BALANCE'].mean().rename(columns = {'MONTHS_BALANCE': 'credit_MONTHS_BALANCE_mean'})


# In[ ]:


credit_stats_by_prev = credit_stats_by_prev.merge(credit_MONTHS_BALANCE_count, on = 'SK_ID_PREV', how = 'left')
credit_stats_by_prev = credit_stats_by_prev.merge(credit_MONTHS_BALANCE_mean, on = 'SK_ID_PREV', how = 'left')
credit_stats_by_prev.head()


# In[ ]:


gc.enable()
del credit_MONTHS_BALANCE_count, credit_MONTHS_BALANCE_mean
gc.collect()


# In[ ]:


credit_AMT_BALANCE_mean = credit.groupby('SK_ID_PREV', as_index=False)['AMT_BALANCE'].mean().rename(columns = {'AMT_BALANCE': 'credit_AMT_BALANCE_mean'})
credit_stats_by_prev = credit_stats_by_prev.merge(credit_AMT_BALANCE_mean, on = 'SK_ID_PREV', how = 'left')
gc.enable()
del credit_AMT_BALANCE_mean
gc.collect()


# In[ ]:


credit_AMT_CREDIT_LIMIT_ACTUAL_mean = credit.groupby('SK_ID_PREV', as_index=False)['AMT_CREDIT_LIMIT_ACTUAL'].mean().rename(columns = {'AMT_CREDIT_LIMIT_ACTUAL': 'credit_AMT_CREDIT_LIMIT_ACTUAL_mean'})
credit_stats_by_prev = credit_stats_by_prev.merge(credit_AMT_CREDIT_LIMIT_ACTUAL_mean, on = 'SK_ID_PREV', how = 'left')
gc.enable()
del credit_AMT_CREDIT_LIMIT_ACTUAL_mean
gc.collect()


# In[ ]:


credit_AMT_DRAWINGS_ATM_CURRENT_mean = credit.groupby('SK_ID_PREV', as_index=False)['AMT_DRAWINGS_ATM_CURRENT'].mean().rename(columns = {'AMT_DRAWINGS_ATM_CURRENT': 'credit_AMT_DRAWINGS_ATM_CURRENT_mean'})
credit_stats_by_prev = credit_stats_by_prev.merge(credit_AMT_DRAWINGS_ATM_CURRENT_mean, on = 'SK_ID_PREV', how = 'left')
gc.enable()
del credit_AMT_DRAWINGS_ATM_CURRENT_mean
gc.collect()


# In[ ]:


credit_AMT_DRAWINGS_CURRENT_mean = credit.groupby('SK_ID_PREV', as_index=False)['AMT_DRAWINGS_CURRENT'].mean().rename(columns = {'AMT_DRAWINGS_CURRENT': 'credit_AMT_DRAWINGS_CURRENT_mean'})
credit_stats_by_prev = credit_stats_by_prev.merge(credit_AMT_DRAWINGS_CURRENT_mean, on = 'SK_ID_PREV', how = 'left')
gc.enable()
del credit_AMT_DRAWINGS_CURRENT_mean
gc.collect()


# In[ ]:


credit_AMT_DRAWINGS_OTHER_CURRENT_mean = credit.groupby('SK_ID_PREV', as_index=False)['AMT_DRAWINGS_OTHER_CURRENT'].mean().rename(columns = {'AMT_DRAWINGS_OTHER_CURRENT': 'credit_AMT_DRAWINGS_OTHER_CURRENT_mean'})
credit_stats_by_prev = credit_stats_by_prev.merge(credit_AMT_DRAWINGS_OTHER_CURRENT_mean, on = 'SK_ID_PREV', how = 'left')
gc.enable()
del credit_AMT_DRAWINGS_OTHER_CURRENT_mean
gc.collect()


# In[ ]:


credit_AMT_DRAWINGS_POS_CURRENT_mean = credit.groupby('SK_ID_PREV', as_index=False)['AMT_DRAWINGS_POS_CURRENT'].mean().rename(columns = {'AMT_DRAWINGS_POS_CURRENT': 'credit_AMT_DRAWINGS_POS_CURRENT_mean'})
credit_stats_by_prev = credit_stats_by_prev.merge(credit_AMT_DRAWINGS_POS_CURRENT_mean, on = 'SK_ID_PREV', how = 'left')
gc.enable()
del credit_AMT_DRAWINGS_POS_CURRENT_mean
gc.collect()


# In[ ]:


credit_AMT_INST_MIN_REGULARITY_mean = credit.groupby('SK_ID_PREV', as_index=False)['AMT_INST_MIN_REGULARITY'].mean().rename(columns = {'AMT_INST_MIN_REGULARITY': 'credit_AMT_INST_MIN_REGULARITY_mean'})
credit_stats_by_prev = credit_stats_by_prev.merge(credit_AMT_INST_MIN_REGULARITY_mean, on = 'SK_ID_PREV', how = 'left')
gc.enable()
del credit_AMT_INST_MIN_REGULARITY_mean
gc.collect()


# In[ ]:


credit_AMT_PAYMENT_CURRENT_mean = credit.groupby('SK_ID_PREV', as_index=False)['AMT_PAYMENT_CURRENT'].mean().rename(columns = {'AMT_PAYMENT_CURRENT': 'credit_AMT_PAYMENT_CURRENT_mean'})
credit_stats_by_prev = credit_stats_by_prev.merge(credit_AMT_PAYMENT_CURRENT_mean, on = 'SK_ID_PREV', how = 'left')
gc.enable()
del credit_AMT_PAYMENT_CURRENT_mean
gc.collect()


# In[ ]:


credit_AMT_PAYMENT_TOTAL_CURRENT_mean = credit.groupby('SK_ID_PREV', as_index=False)['AMT_PAYMENT_TOTAL_CURRENT'].mean().rename(columns = {'AMT_PAYMENT_TOTAL_CURRENT': 'credit_AMT_PAYMENT_TOTAL_CURRENT_mean'})
credit_stats_by_prev = credit_stats_by_prev.merge(credit_AMT_PAYMENT_TOTAL_CURRENT_mean, on = 'SK_ID_PREV', how = 'left')
gc.enable()
del credit_AMT_PAYMENT_TOTAL_CURRENT_mean
gc.collect()


# In[ ]:


credit_AMT_RECEIVABLE_PRINCIPAL_mean = credit.groupby('SK_ID_PREV', as_index=False)['AMT_RECEIVABLE_PRINCIPAL'].mean().rename(columns = {'AMT_RECEIVABLE_PRINCIPAL': 'credit_AMT_RECEIVABLE_PRINCIPAL_mean'})
credit_stats_by_prev = credit_stats_by_prev.merge(credit_AMT_RECEIVABLE_PRINCIPAL_mean, on = 'SK_ID_PREV', how = 'left')
gc.enable()
del credit_AMT_RECEIVABLE_PRINCIPAL_mean
gc.collect()


# In[ ]:


credit_AMT_RECIVABLE_mean = credit.groupby('SK_ID_PREV', as_index=False)['AMT_RECIVABLE'].mean().rename(columns = {'AMT_RECIVABLE': 'credit_AMT_RECIVABLE_mean'})
credit_stats_by_prev = credit_stats_by_prev.merge(credit_AMT_RECIVABLE_mean, on = 'SK_ID_PREV', how = 'left')
gc.enable()
del credit_AMT_RECIVABLE_mean
gc.collect()


# In[ ]:


credit_AMT_TOTAL_RECEIVABLE_mean = credit.groupby('SK_ID_PREV', as_index=False)['AMT_TOTAL_RECEIVABLE'].mean().rename(columns = {'AMT_TOTAL_RECEIVABLE': 'credit_AMT_TOTAL_RECEIVABLE_mean'})
credit_stats_by_prev = credit_stats_by_prev.merge(credit_AMT_TOTAL_RECEIVABLE_mean, on = 'SK_ID_PREV', how = 'left')
gc.enable()
del credit_AMT_TOTAL_RECEIVABLE_mean
gc.collect()


# In[ ]:


credit_CNT_DRAWINGS_ATM_CURRENT_mean = credit.groupby('SK_ID_PREV', as_index=False)['CNT_DRAWINGS_ATM_CURRENT'].mean().rename(columns = {'CNT_DRAWINGS_ATM_CURRENT': 'credit_CNT_DRAWINGS_ATM_CURRENT_mean'})
credit_stats_by_prev = credit_stats_by_prev.merge(credit_CNT_DRAWINGS_ATM_CURRENT_mean, on = 'SK_ID_PREV', how = 'left')
gc.enable()
del credit_CNT_DRAWINGS_ATM_CURRENT_mean
gc.collect()


# In[ ]:


credit_CNT_DRAWINGS_CURRENT_mean = credit.groupby('SK_ID_PREV', as_index=False)['CNT_DRAWINGS_CURRENT'].mean().rename(columns = {'CNT_DRAWINGS_CURRENT': 'credit_CNT_DRAWINGS_CURRENT_mean'})
credit_stats_by_prev = credit_stats_by_prev.merge(credit_CNT_DRAWINGS_CURRENT_mean, on = 'SK_ID_PREV', how = 'left')
gc.enable()
del credit_CNT_DRAWINGS_CURRENT_mean
gc.collect()


# In[ ]:


credit_CNT_DRAWINGS_OTHER_CURRENT_mean = credit.groupby('SK_ID_PREV', as_index=False)['CNT_DRAWINGS_OTHER_CURRENT'].mean().rename(columns = {'CNT_DRAWINGS_OTHER_CURRENT': 'credit_CNT_DRAWINGS_OTHER_CURRENT_mean'})
credit_stats_by_prev = credit_stats_by_prev.merge(credit_CNT_DRAWINGS_OTHER_CURRENT_mean, on = 'SK_ID_PREV', how = 'left')
gc.enable()
del credit_CNT_DRAWINGS_OTHER_CURRENT_mean
gc.collect()


# In[ ]:


credit_CNT_DRAWINGS_POS_CURRENT_mean = credit.groupby('SK_ID_PREV', as_index=False)['CNT_DRAWINGS_POS_CURRENT'].mean().rename(columns = {'CNT_DRAWINGS_POS_CURRENT': 'credit_CNT_DRAWINGS_POS_CURRENT_mean'})
credit_stats_by_prev = credit_stats_by_prev.merge(credit_CNT_DRAWINGS_POS_CURRENT_mean, on = 'SK_ID_PREV', how = 'left')
gc.enable()
del credit_CNT_DRAWINGS_POS_CURRENT_mean
gc.collect()


# In[ ]:


credit_CNT_INSTALMENT_MATURE_CUM_mean = credit.groupby('SK_ID_PREV', as_index=False)['CNT_INSTALMENT_MATURE_CUM'].mean().rename(columns = {'CNT_INSTALMENT_MATURE_CUM': 'credit_CNT_INSTALMENT_MATURE_CUM_mean'})
credit_stats_by_prev = credit_stats_by_prev.merge(credit_CNT_INSTALMENT_MATURE_CUM_mean, on = 'SK_ID_PREV', how = 'left')
gc.enable()
del credit_CNT_INSTALMENT_MATURE_CUM_mean
gc.collect()


# In[ ]:


credit_SK_DPD_mean = credit.groupby('SK_ID_PREV', as_index=False)['SK_DPD'].mean().rename(columns = {'SK_DPD': 'credit_SK_DPD_mean'})
credit_stats_by_prev = credit_stats_by_prev.merge(credit_SK_DPD_mean, on = 'SK_ID_PREV', how = 'left')
gc.enable()
del credit_SK_DPD_mean
gc.collect()


# In[ ]:


credit_SK_DPD_DEF_mean = credit.groupby('SK_ID_PREV', as_index=False)['SK_DPD_DEF'].mean().rename(columns = {'SK_DPD_DEF': 'credit_SK_DPD_DEF_mean'})
credit_stats_by_prev = credit_stats_by_prev.merge(credit_SK_DPD_DEF_mean, on = 'SK_ID_PREV', how = 'left')
gc.enable()
del credit_SK_DPD_DEF_mean
gc.collect()


# In[ ]:


# now let's deal with our one categorical variable, NAME_CONTRACT_STATUS
credit_cats = pd.get_dummies(credit.select_dtypes('object'))
credit_cats['SK_ID_PREV'] = credit['SK_ID_PREV']
credit_cats.head()


# In[ ]:


credit_cats_grouped = credit_cats.groupby('SK_ID_PREV').agg('sum')
credit_cats_grouped.head()


# In[ ]:


credit_stats_by_prev = credit_stats_by_prev.merge(credit_cats_grouped, on = 'SK_ID_PREV', how = 'left')
gc.enable()
del credit_cats_grouped, credit_cats
gc.collect()


# In[ ]:


credit_stats_by_prev.head()


# Now we have all of this data by previous ID, but we need to aggregate this on SK_ID_CURR.  We will repeat what we did above for the bureau balance data, averaging these variables for each applicant.

# In[ ]:


credit_MONTHS_BALANCE_count_mean = credit_stats_by_prev.groupby('SK_ID_CURR', as_index=False)['credit_MONTHS_BALANCE_count'].mean().rename(columns = {'credit_MONTHS_BALANCE_count': 'credit_MONTHS_BALANCE_count_mean'})
dataset = dataset.merge(credit_MONTHS_BALANCE_count_mean, on = 'SK_ID_CURR', how = 'left')
gc.enable()
del credit_MONTHS_BALANCE_count_mean
gc.collect()


# In[ ]:


credit_MONTHS_BALANCE_mean_mean = credit_stats_by_prev.groupby('SK_ID_CURR', as_index=False)['credit_MONTHS_BALANCE_mean'].mean().rename(columns = {'credit_MONTHS_BALANCE_mean': 'credit_MONTHS_BALANCE_mean_mean'})
dataset = dataset.merge(credit_MONTHS_BALANCE_mean_mean, on = 'SK_ID_CURR', how = 'left')
gc.enable()
del credit_MONTHS_BALANCE_mean_mean
gc.collect()


# In[ ]:


credit_AMT_BALANCE_mean_mean = credit_stats_by_prev.groupby('SK_ID_CURR', as_index=False)['credit_AMT_BALANCE_mean'].mean().rename(columns = {'credit_AMT_BALANCE_mean': 'credit_AMT_BALANCE_mean_mean'})
dataset = dataset.merge(credit_AMT_BALANCE_mean_mean, on = 'SK_ID_CURR', how = 'left')
gc.enable()
del credit_AMT_BALANCE_mean_mean
gc.collect()


# In[ ]:


credit_AMT_CREDIT_LIMIT_ACTUAL_mean_mean = credit_stats_by_prev.groupby('SK_ID_CURR', as_index=False)['credit_AMT_CREDIT_LIMIT_ACTUAL_mean'].mean().rename(columns = {'credit_AMT_CREDIT_LIMIT_ACTUAL_mean': 'credit_AMT_CREDIT_LIMIT_ACTUAL_mean_mean'})
dataset = dataset.merge(credit_AMT_CREDIT_LIMIT_ACTUAL_mean_mean, on = 'SK_ID_CURR', how = 'left')
gc.enable()
del credit_AMT_CREDIT_LIMIT_ACTUAL_mean_mean
gc.collect()


# In[ ]:


credit_AMT_DRAWINGS_ATM_CURRENT_mean_mean = credit_stats_by_prev.groupby('SK_ID_CURR', as_index=False)['credit_AMT_DRAWINGS_ATM_CURRENT_mean'].mean().rename(columns = {'credit_AMT_DRAWINGS_ATM_CURRENT_mean': 'credit_AMT_DRAWINGS_ATM_CURRENT_mean_mean'})
dataset = dataset.merge(credit_AMT_DRAWINGS_ATM_CURRENT_mean_mean, on = 'SK_ID_CURR', how = 'left')
gc.enable()
del credit_AMT_DRAWINGS_ATM_CURRENT_mean_mean
gc.collect()


# In[ ]:


credit_AMT_DRAWINGS_CURRENT_mean_mean = credit_stats_by_prev.groupby('SK_ID_CURR', as_index=False)['credit_AMT_DRAWINGS_CURRENT_mean'].mean().rename(columns = {'credit_AMT_DRAWINGS_CURRENT_mean': 'credit_AMT_DRAWINGS_CURRENT_mean_mean'})
dataset = dataset.merge(credit_AMT_DRAWINGS_CURRENT_mean_mean, on = 'SK_ID_CURR', how = 'left')
gc.enable()
del credit_AMT_DRAWINGS_CURRENT_mean_mean
gc.collect()


# In[ ]:


credit_AMT_DRAWINGS_OTHER_CURRENT_mean_mean = credit_stats_by_prev.groupby('SK_ID_CURR', as_index=False)['credit_AMT_DRAWINGS_OTHER_CURRENT_mean'].mean().rename(columns = {'credit_AMT_DRAWINGS_OTHER_CURRENT_mean': 'credit_AMT_DRAWINGS_OTHER_CURRENT_mean_mean'})
dataset = dataset.merge(credit_AMT_DRAWINGS_OTHER_CURRENT_mean_mean, on = 'SK_ID_CURR', how = 'left')
gc.enable()
del credit_AMT_DRAWINGS_OTHER_CURRENT_mean_mean
gc.collect()


# In[ ]:


credit_AMT_DRAWINGS_POS_CURRENT_mean_mean = credit_stats_by_prev.groupby('SK_ID_CURR', as_index=False)['credit_AMT_DRAWINGS_POS_CURRENT_mean'].mean().rename(columns = {'credit_AMT_DRAWINGS_POS_CURRENT_mean': 'credit_AMT_DRAWINGS_POS_CURRENT_mean_mean'})
dataset = dataset.merge(credit_AMT_DRAWINGS_POS_CURRENT_mean_mean, on = 'SK_ID_CURR', how = 'left')
gc.enable()
del credit_AMT_DRAWINGS_POS_CURRENT_mean_mean
gc.collect()


# In[ ]:


credit_AMT_INST_MIN_REGULARITY_mean_mean = credit_stats_by_prev.groupby('SK_ID_CURR', as_index=False)['credit_AMT_INST_MIN_REGULARITY_mean'].mean().rename(columns = {'credit_AMT_INST_MIN_REGULARITY_mean': 'credit_AMT_INST_MIN_REGULARITY_mean_mean'})
dataset = dataset.merge(credit_AMT_INST_MIN_REGULARITY_mean_mean, on = 'SK_ID_CURR', how = 'left')
gc.enable()
del credit_AMT_INST_MIN_REGULARITY_mean_mean
gc.collect()


# In[ ]:


credit_AMT_PAYMENT_CURRENT_mean_mean = credit_stats_by_prev.groupby('SK_ID_CURR', as_index=False)['credit_AMT_PAYMENT_CURRENT_mean'].mean().rename(columns = {'credit_AMT_PAYMENT_CURRENT_mean': 'credit_AMT_PAYMENT_CURRENT_mean_mean'})
dataset = dataset.merge(credit_AMT_PAYMENT_CURRENT_mean_mean, on = 'SK_ID_CURR', how = 'left')
gc.enable()
del credit_AMT_PAYMENT_CURRENT_mean_mean
gc.collect()


# In[ ]:


credit_AMT_PAYMENT_TOTAL_CURRENT_mean_mean = credit_stats_by_prev.groupby('SK_ID_CURR', as_index=False)['credit_AMT_PAYMENT_TOTAL_CURRENT_mean'].mean().rename(columns = {'credit_AMT_PAYMENT_TOTAL_CURRENT_mean': 'credit_AMT_PAYMENT_TOTAL_CURRENT_mean_mean'})
dataset = dataset.merge(credit_AMT_PAYMENT_TOTAL_CURRENT_mean_mean, on = 'SK_ID_CURR', how = 'left')
gc.enable()
del credit_AMT_PAYMENT_TOTAL_CURRENT_mean_mean
gc.collect()


# In[ ]:


credit_AMT_RECEIVABLE_PRINCIPAL_mean_mean = credit_stats_by_prev.groupby('SK_ID_CURR', as_index=False)['credit_AMT_RECEIVABLE_PRINCIPAL_mean'].mean().rename(columns = {'credit_AMT_RECEIVABLE_PRINCIPAL_mean': 'credit_AMT_RECEIVABLE_PRINCIPAL_mean_mean'})
dataset = dataset.merge(credit_AMT_RECEIVABLE_PRINCIPAL_mean_mean, on = 'SK_ID_CURR', how = 'left')
gc.enable()
del credit_AMT_RECEIVABLE_PRINCIPAL_mean_mean
gc.collect()


# In[ ]:


credit_AMT_RECIVABLE_mean_mean = credit_stats_by_prev.groupby('SK_ID_CURR', as_index=False)['credit_AMT_RECIVABLE_mean'].mean().rename(columns = {'credit_AMT_RECIVABLE_mean': 'credit_AMT_RECIVABLE_mean_mean'})
dataset = dataset.merge(credit_AMT_RECIVABLE_mean_mean, on = 'SK_ID_CURR', how = 'left')
gc.enable()
del credit_AMT_RECIVABLE_mean_mean
gc.collect()


# In[ ]:


credit_AMT_TOTAL_RECEIVABLE_mean_mean = credit_stats_by_prev.groupby('SK_ID_CURR', as_index=False)['credit_AMT_TOTAL_RECEIVABLE_mean'].mean().rename(columns = {'credit_AMT_TOTAL_RECEIVABLE_mean': 'credit_AMT_TOTAL_RECEIVABLE_mean_mean'})
dataset = dataset.merge(credit_AMT_TOTAL_RECEIVABLE_mean_mean, on = 'SK_ID_CURR', how = 'left')
gc.enable()
del credit_AMT_TOTAL_RECEIVABLE_mean_mean
gc.collect()


# In[ ]:


credit_CNT_DRAWINGS_ATM_CURRENT_mean_mean = credit_stats_by_prev.groupby('SK_ID_CURR', as_index=False)['credit_CNT_DRAWINGS_ATM_CURRENT_mean'].mean().rename(columns = {'credit_CNT_DRAWINGS_ATM_CURRENT_mean': 'credit_CNT_DRAWINGS_ATM_CURRENT_mean_mean'})
dataset = dataset.merge(credit_CNT_DRAWINGS_ATM_CURRENT_mean_mean, on = 'SK_ID_CURR', how = 'left')
gc.enable()
del credit_CNT_DRAWINGS_ATM_CURRENT_mean_mean
gc.collect()


# In[ ]:


credit_CNT_DRAWINGS_CURRENT_mean_mean = credit_stats_by_prev.groupby('SK_ID_CURR', as_index=False)['credit_CNT_DRAWINGS_CURRENT_mean'].mean().rename(columns = {'credit_CNT_DRAWINGS_CURRENT_mean': 'credit_CNT_DRAWINGS_CURRENT_mean_mean'})
dataset = dataset.merge(credit_CNT_DRAWINGS_CURRENT_mean_mean, on = 'SK_ID_CURR', how = 'left')
gc.enable()
del credit_CNT_DRAWINGS_CURRENT_mean_mean
gc.collect()


# In[ ]:


credit_CNT_DRAWINGS_OTHER_CURRENT_mean_mean = credit_stats_by_prev.groupby('SK_ID_CURR', as_index=False)['credit_CNT_DRAWINGS_OTHER_CURRENT_mean'].mean().rename(columns = {'credit_CNT_DRAWINGS_OTHER_CURRENT_mean': 'credit_CNT_DRAWINGS_OTHER_CURRENT_mean_mean'})
dataset = dataset.merge(credit_CNT_DRAWINGS_OTHER_CURRENT_mean_mean, on = 'SK_ID_CURR', how = 'left')
gc.enable()
del credit_CNT_DRAWINGS_OTHER_CURRENT_mean_mean
gc.collect()


# In[ ]:


credit_CNT_DRAWINGS_POS_CURRENT_mean_mean = credit_stats_by_prev.groupby('SK_ID_CURR', as_index=False)['credit_CNT_DRAWINGS_POS_CURRENT_mean'].mean().rename(columns = {'credit_CNT_DRAWINGS_POS_CURRENT_mean': 'credit_CNT_DRAWINGS_POS_CURRENT_mean_mean'})
dataset = dataset.merge(credit_CNT_DRAWINGS_POS_CURRENT_mean_mean, on = 'SK_ID_CURR', how = 'left')
gc.enable()
del credit_CNT_DRAWINGS_POS_CURRENT_mean_mean
gc.collect()


# In[ ]:


credit_CNT_INSTALMENT_MATURE_CUM_mean_mean = credit_stats_by_prev.groupby('SK_ID_CURR', as_index=False)['credit_CNT_INSTALMENT_MATURE_CUM_mean'].mean().rename(columns = {'credit_CNT_INSTALMENT_MATURE_CUM_mean': 'credit_CNT_INSTALMENT_MATURE_CUM_mean_mean'})
dataset = dataset.merge(credit_CNT_INSTALMENT_MATURE_CUM_mean_mean, on = 'SK_ID_CURR', how = 'left')
gc.enable()
del credit_CNT_INSTALMENT_MATURE_CUM_mean_mean
gc.collect()


# In[ ]:


credit_SK_DPD_mean_mean = credit_stats_by_prev.groupby('SK_ID_CURR', as_index=False)['credit_SK_DPD_mean'].mean().rename(columns = {'credit_SK_DPD_mean': 'credit_SK_DPD_mean_mean'})
dataset = dataset.merge(credit_SK_DPD_mean_mean, on = 'SK_ID_CURR', how = 'left')
gc.enable()
del credit_SK_DPD_mean_mean
gc.collect()


# In[ ]:


credit_SK_DPD_DEF_mean_mean = credit_stats_by_prev.groupby('SK_ID_CURR', as_index=False)['credit_SK_DPD_DEF_mean'].mean().rename(columns = {'credit_SK_DPD_DEF_mean': 'credit_SK_DPD_DEF_mean_mean'})
dataset = dataset.merge(credit_SK_DPD_DEF_mean_mean, on = 'SK_ID_CURR', how = 'left')
gc.enable()
del credit_SK_DPD_DEF_mean_mean
gc.collect()


# In[ ]:


credit_NAME_CONTRACT_STATUS_Active_mean = credit_stats_by_prev.groupby('SK_ID_CURR', as_index=False)['NAME_CONTRACT_STATUS_Active'].mean().rename(columns = {'NAME_CONTRACT_STATUS_Active': 'credit_NAME_CONTRACT_STATUS_Active_mean'})
dataset = dataset.merge(credit_NAME_CONTRACT_STATUS_Active_mean, on = 'SK_ID_CURR', how = 'left')
gc.enable()
del credit_NAME_CONTRACT_STATUS_Active_mean
gc.collect()


# In[ ]:


credit_NAME_CONTRACT_STATUS_Approved_mean = credit_stats_by_prev.groupby('SK_ID_CURR', as_index=False)['NAME_CONTRACT_STATUS_Approved'].mean().rename(columns = {'NAME_CONTRACT_STATUS_Approved': 'credit_NAME_CONTRACT_STATUS_Approved_mean'})
dataset = dataset.merge(credit_NAME_CONTRACT_STATUS_Approved_mean, on = 'SK_ID_CURR', how = 'left')
gc.enable()
del credit_NAME_CONTRACT_STATUS_Approved_mean
gc.collect()


# In[ ]:


credit_NAME_CONTRACT_STATUS_Completed_mean = credit_stats_by_prev.groupby('SK_ID_CURR', as_index=False)['NAME_CONTRACT_STATUS_Completed'].mean().rename(columns = {'NAME_CONTRACT_STATUS_Completed': 'credit_NAME_CONTRACT_STATUS_Completed_mean'})
dataset = dataset.merge(credit_NAME_CONTRACT_STATUS_Completed_mean, on = 'SK_ID_CURR', how = 'left')
gc.enable()
del credit_NAME_CONTRACT_STATUS_Completed_mean
gc.collect()


# In[ ]:


credit_NAME_CONTRACT_STATUS_Demand_mean = credit_stats_by_prev.groupby('SK_ID_CURR', as_index=False)['NAME_CONTRACT_STATUS_Demand'].mean().rename(columns = {'NAME_CONTRACT_STATUS_Demand': 'credit_NAME_CONTRACT_STATUS_Demand_mean'})
dataset = dataset.merge(credit_NAME_CONTRACT_STATUS_Demand_mean, on = 'SK_ID_CURR', how = 'left')
gc.enable()
del credit_NAME_CONTRACT_STATUS_Demand_mean
gc.collect()


# In[ ]:


credit_NAME_CONTRACT_STATUS_Refused_mean = credit_stats_by_prev.groupby('SK_ID_CURR', as_index=False)['NAME_CONTRACT_STATUS_Refused'].mean().rename(columns = {'NAME_CONTRACT_STATUS_Refused': 'credit_NAME_CONTRACT_STATUS_Refused_mean'})
dataset = dataset.merge(credit_NAME_CONTRACT_STATUS_Refused_mean, on = 'SK_ID_CURR', how = 'left')
gc.enable()
del credit_NAME_CONTRACT_STATUS_Refused_mean
gc.collect()


# In[ ]:


credit_NAME_CONTRACT_STATUS_Sent_mean = credit_stats_by_prev.groupby('SK_ID_CURR', as_index=False)['NAME_CONTRACT_STATUS_Sent proposal'].mean().rename(columns = {'NAME_CONTRACT_STATUS_Sent proposal': 'credit_NAME_CONTRACT_STATUS_Sent_mean'})
dataset = dataset.merge(credit_NAME_CONTRACT_STATUS_Sent_mean, on = 'SK_ID_CURR', how = 'left')
gc.enable()
del credit_NAME_CONTRACT_STATUS_Sent_mean
gc.collect()


# In[ ]:


credit_NAME_CONTRACT_STATUS_Signed_mean = credit_stats_by_prev.groupby('SK_ID_CURR', as_index=False)['NAME_CONTRACT_STATUS_Signed'].mean().rename(columns = {'NAME_CONTRACT_STATUS_Signed': 'credit_NAME_CONTRACT_STATUS_Signed_mean'})
dataset = dataset.merge(credit_NAME_CONTRACT_STATUS_Signed_mean, on = 'SK_ID_CURR', how = 'left')
gc.enable()
del credit_NAME_CONTRACT_STATUS_Signed_mean
gc.collect()


# In[ ]:


print('dataset data shape: ', dataset.shape)
dataset.head()


# In[ ]:


# let's free up some memory by deleting some of the dataframes we are done with
gc.enable()
del credit, credit_stats_by_prev
gc.collect()


# In[ ]:


# the next file we will investigate is installments_payments
install = pd.read_csv('../input/installments_payments.csv')
install.head()


# For this file, I think we want to look at creating a couple of additional variables.  For the dates of payments, I would like to see the difference between the due date and the actual payment date.  For the amounts of payments, I would like to see the difference between the amount owed and the actual amount paid.  So let's create a few new variables.

# In[ ]:


# create the additional variables
install['DAYS_DIFF'] = install['DAYS_INSTALMENT'] - install['DAYS_ENTRY_PAYMENT']
install['AMT_DIFF'] = install['AMT_INSTALMENT'] - install['AMT_PAYMENT']
install.head()


# In[ ]:


install_stats_by_prev = install[['SK_ID_PREV', 'SK_ID_CURR']]


# In[ ]:


install_NUM_INSTALMENT_VERSION_count = install.groupby('SK_ID_PREV', as_index=False)['NUM_INSTALMENT_VERSION'].count().rename(columns = {'NUM_INSTALMENT_VERSION': 'install_NUM_INSTALMENT_VERSION_count'})
install_NUM_INSTALMENT_VERSION_max = install.groupby('SK_ID_PREV', as_index=False)['NUM_INSTALMENT_VERSION'].max().rename(columns = {'NUM_INSTALMENT_VERSION': 'install_NUM_INSTALMENT_VERSION_max'})
install_stats_by_prev = install_stats_by_prev.merge(install_NUM_INSTALMENT_VERSION_count, on = 'SK_ID_PREV', how = 'left')
install_stats_by_prev = install_stats_by_prev.merge(install_NUM_INSTALMENT_VERSION_max, on = 'SK_ID_PREV', how = 'left')


# In[ ]:


gc.enable()
del install_NUM_INSTALMENT_VERSION_count, install_NUM_INSTALMENT_VERSION_max
gc.collect()


# In[ ]:


install_DAYS_INSTALMENT_mean = install.groupby('SK_ID_PREV', as_index=False)['DAYS_INSTALMENT'].mean().rename(columns = {'DAYS_INSTALMENT': 'install_DAYS_INSTALMENT_mean'})
install_stats_by_prev = install_stats_by_prev.merge(install_DAYS_INSTALMENT_mean, on = 'SK_ID_PREV', how = 'left')
gc.enable()
del install_DAYS_INSTALMENT_mean
gc.collect()


# In[ ]:


install_DAYS_ENTRY_PAYMENT_mean = install.groupby('SK_ID_PREV', as_index=False)['DAYS_ENTRY_PAYMENT'].mean().rename(columns = {'DAYS_ENTRY_PAYMENT': 'install_DAYS_ENTRY_PAYMENT_mean'})
install_stats_by_prev = install_stats_by_prev.merge(install_DAYS_ENTRY_PAYMENT_mean, on = 'SK_ID_PREV', how = 'left')
gc.enable()
del install_DAYS_ENTRY_PAYMENT_mean
gc.collect()


# In[ ]:


install_AMT_INSTALMENT_mean = install.groupby('SK_ID_PREV', as_index=False)['AMT_INSTALMENT'].mean().rename(columns = {'AMT_INSTALMENT': 'install_AMT_INSTALMENT_mean'})
install_stats_by_prev = install_stats_by_prev.merge(install_AMT_INSTALMENT_mean, on = 'SK_ID_PREV', how = 'left')
gc.enable()
del install_AMT_INSTALMENT_mean
gc.collect()


# In[ ]:


install_AMT_PAYMENT_mean = install.groupby('SK_ID_PREV', as_index=False)['AMT_PAYMENT'].mean().rename(columns = {'AMT_PAYMENT': 'install_AMT_PAYMENT_mean'})
install_stats_by_prev = install_stats_by_prev.merge(install_AMT_PAYMENT_mean, on = 'SK_ID_PREV', how = 'left')
gc.enable()
del install_AMT_PAYMENT_mean
gc.collect()


# In[ ]:


# capture the mean, max, and min for DAYS_DIFF
install_DAYS_DIFF_mean = install.groupby('SK_ID_PREV', as_index=False)['DAYS_DIFF'].mean().rename(columns = {'DAYS_DIFF': 'install_DAYS_DIFF_mean'})
install_DAYS_DIFF_max = install.groupby('SK_ID_PREV', as_index=False)['DAYS_DIFF'].max().rename(columns = {'DAYS_DIFF': 'install_DAYS_DIFF_max'})
install_DAYS_DIFF_min = install.groupby('SK_ID_PREV', as_index=False)['DAYS_DIFF'].min().rename(columns = {'DAYS_DIFF': 'install_DAYS_DIFF_min'})
install_stats_by_prev = install_stats_by_prev.merge(install_DAYS_DIFF_mean, on = 'SK_ID_PREV', how = 'left')
install_stats_by_prev = install_stats_by_prev.merge(install_DAYS_DIFF_max, on = 'SK_ID_PREV', how = 'left')
install_stats_by_prev = install_stats_by_prev.merge(install_DAYS_DIFF_min, on = 'SK_ID_PREV', how = 'left')
gc.enable()
del install_DAYS_DIFF_mean, install_DAYS_DIFF_max, install_DAYS_DIFF_min
gc.collect()


# In[ ]:


# capture the mean, max, and min for AMT_DIFF
install_AMT_DIFF_mean = install.groupby('SK_ID_PREV', as_index=False)['AMT_DIFF'].mean().rename(columns = {'AMT_DIFF': 'install_AMT_DIFF_mean'})
install_AMT_DIFF_max = install.groupby('SK_ID_PREV', as_index=False)['AMT_DIFF'].max().rename(columns = {'AMT_DIFF': 'install_AMT_DIFF_max'})
install_AMT_DIFF_min = install.groupby('SK_ID_PREV', as_index=False)['AMT_DIFF'].min().rename(columns = {'AMT_DIFF': 'install_AMT_DIFF_min'})
install_stats_by_prev = install_stats_by_prev.merge(install_AMT_DIFF_mean, on = 'SK_ID_PREV', how = 'left')
install_stats_by_prev = install_stats_by_prev.merge(install_AMT_DIFF_max, on = 'SK_ID_PREV', how = 'left')
install_stats_by_prev = install_stats_by_prev.merge(install_AMT_DIFF_min, on = 'SK_ID_PREV', how = 'left')
gc.enable()
del install_AMT_DIFF_mean, install_AMT_DIFF_max, install_AMT_DIFF_min
gc.collect()


# In[ ]:


install_stats_by_prev.head()


# Now we have all of this data by previous ID, but we need to aggregate this on SK_ID_CURR.  We will repeat what we did above for the credit data.  For most cases, I will use the average to combine.

# In[ ]:


install_NUM_INSTALMENT_VERSION_count_mean = install_stats_by_prev.groupby('SK_ID_CURR', as_index=False)['install_NUM_INSTALMENT_VERSION_count'].mean().rename(columns = {'install_NUM_INSTALMENT_VERSION_count': 'install_NUM_INSTALMENT_VERSION_count_mean'})
dataset = dataset.merge(install_NUM_INSTALMENT_VERSION_count_mean, on = 'SK_ID_CURR', how = 'left')
gc.enable()
del install_NUM_INSTALMENT_VERSION_count_mean
gc.collect()


# In[ ]:


install_NUM_INSTALMENT_VERSION_max_max = install_stats_by_prev.groupby('SK_ID_CURR', as_index=False)['install_NUM_INSTALMENT_VERSION_max'].max().rename(columns = {'install_NUM_INSTALMENT_VERSION_max': 'install_NUM_INSTALMENT_VERSION_max_max'})
dataset = dataset.merge(install_NUM_INSTALMENT_VERSION_max_max, on = 'SK_ID_CURR', how = 'left')
gc.enable()
del install_NUM_INSTALMENT_VERSION_max_max
gc.collect()


# In[ ]:


install_DAYS_INSTALMENT_mean_mean = install_stats_by_prev.groupby('SK_ID_CURR', as_index=False)['install_DAYS_INSTALMENT_mean'].mean().rename(columns = {'install_DAYS_INSTALMENT_mean': 'install_DAYS_INSTALMENT_mean_mean'})
dataset = dataset.merge(install_DAYS_INSTALMENT_mean_mean, on = 'SK_ID_CURR', how = 'left')
gc.enable()
del install_DAYS_INSTALMENT_mean_mean
gc.collect()


# In[ ]:


install_DAYS_ENTRY_PAYMENT_mean_mean = install_stats_by_prev.groupby('SK_ID_CURR', as_index=False)['install_DAYS_ENTRY_PAYMENT_mean'].mean().rename(columns = {'install_DAYS_ENTRY_PAYMENT_mean': 'install_DAYS_ENTRY_PAYMENT_mean_mean'})
dataset = dataset.merge(install_DAYS_ENTRY_PAYMENT_mean_mean, on = 'SK_ID_CURR', how = 'left')
gc.enable()
del install_DAYS_ENTRY_PAYMENT_mean_mean
gc.collect()


# In[ ]:


install_AMT_INSTALMENT_mean_mean = install_stats_by_prev.groupby('SK_ID_CURR', as_index=False)['install_AMT_INSTALMENT_mean'].mean().rename(columns = {'install_AMT_INSTALMENT_mean': 'install_AMT_INSTALMENT_mean_mean'})
dataset = dataset.merge(install_AMT_INSTALMENT_mean_mean, on = 'SK_ID_CURR', how = 'left')
gc.enable()
del install_AMT_INSTALMENT_mean_mean
gc.collect()


# In[ ]:


install_AMT_PAYMENT_mean_mean = install_stats_by_prev.groupby('SK_ID_CURR', as_index=False)['install_AMT_PAYMENT_mean'].mean().rename(columns = {'install_AMT_PAYMENT_mean': 'install_AMT_PAYMENT_mean_mean'})
dataset = dataset.merge(install_AMT_PAYMENT_mean_mean, on = 'SK_ID_CURR', how = 'left')
gc.enable()
del install_AMT_PAYMENT_mean_mean
gc.collect()


# In[ ]:


install_DAYS_DIFF_mean_mean = install_stats_by_prev.groupby('SK_ID_CURR', as_index=False)['install_DAYS_DIFF_mean'].mean().rename(columns = {'install_DAYS_DIFF_mean': 'install_DAYS_DIFF_mean_mean'})
dataset = dataset.merge(install_DAYS_DIFF_mean_mean, on = 'SK_ID_CURR', how = 'left')
gc.enable()
del install_DAYS_DIFF_mean_mean
gc.collect()


# In[ ]:


install_DAYS_DIFF_max_mean = install_stats_by_prev.groupby('SK_ID_CURR', as_index=False)['install_DAYS_DIFF_max'].mean().rename(columns = {'install_DAYS_DIFF_max': 'install_DAYS_DIFF_max_mean'})
dataset = dataset.merge(install_DAYS_DIFF_max_mean, on = 'SK_ID_CURR', how = 'left')
gc.enable()
del install_DAYS_DIFF_max_mean
gc.collect()


# In[ ]:


install_DAYS_DIFF_min_mean = install_stats_by_prev.groupby('SK_ID_CURR', as_index=False)['install_DAYS_DIFF_min'].mean().rename(columns = {'install_DAYS_DIFF_min': 'install_DAYS_DIFF_min_mean'})
dataset = dataset.merge(install_DAYS_DIFF_min_mean, on = 'SK_ID_CURR', how = 'left')
gc.enable()
del install_DAYS_DIFF_min_mean
gc.collect()


# In[ ]:


install_AMT_DIFF_mean_mean = install_stats_by_prev.groupby('SK_ID_CURR', as_index=False)['install_AMT_DIFF_mean'].mean().rename(columns = {'install_AMT_DIFF_mean': 'install_AMT_DIFF_mean_mean'})
dataset = dataset.merge(install_AMT_DIFF_mean_mean, on = 'SK_ID_CURR', how = 'left')
gc.enable()
del install_AMT_DIFF_mean_mean
gc.collect()


# In[ ]:


install_AMT_DIFF_max_mean = install_stats_by_prev.groupby('SK_ID_CURR', as_index=False)['install_AMT_DIFF_max'].mean().rename(columns = {'install_AMT_DIFF_max': 'install_AMT_DIFF_max_mean'})
dataset = dataset.merge(install_AMT_DIFF_max_mean, on = 'SK_ID_CURR', how = 'left')
gc.enable()
del install_AMT_DIFF_max_mean
gc.collect()


# In[ ]:


install_AMT_DIFF_min_mean = install_stats_by_prev.groupby('SK_ID_CURR', as_index=False)['install_AMT_DIFF_min'].mean().rename(columns = {'install_AMT_DIFF_min': 'install_AMT_DIFF_min_mean'})
dataset = dataset.merge(install_AMT_DIFF_min_mean, on = 'SK_ID_CURR', how = 'left')
gc.enable()
del install_AMT_DIFF_min_mean
gc.collect()


# In[ ]:


print('dataset data shape: ', dataset.shape)
dataset.head()


# In[ ]:


# let's free up some memory by deleting some of the dataframes we are done with
gc.enable()
del install, install_stats_by_prev
gc.collect()


# In[ ]:


# the next file we will investigate is POS_CASH_balance
cash = pd.read_csv('../input/POS_CASH_balance.csv')
cash.head()


# In[ ]:


cash_stats_by_prev = cash[['SK_ID_PREV', 'SK_ID_CURR']]


# In[ ]:


cash_MONTHS_BALANCE_count = cash.groupby('SK_ID_PREV', as_index=False)['MONTHS_BALANCE'].count().rename(columns = {'MONTHS_BALANCE': 'cash_MONTHS_BALANCE_count'})
cash_MONTHS_BALANCE_mean = cash.groupby('SK_ID_PREV', as_index=False)['MONTHS_BALANCE'].mean().rename(columns = {'MONTHS_BALANCE': 'cash_MONTHS_BALANCE_mean'})
cash_stats_by_prev = cash_stats_by_prev.merge(cash_MONTHS_BALANCE_count, on = 'SK_ID_PREV', how = 'left')
cash_stats_by_prev = cash_stats_by_prev.merge(cash_MONTHS_BALANCE_mean, on = 'SK_ID_PREV', how = 'left')
gc.enable()
del cash_MONTHS_BALANCE_count, cash_MONTHS_BALANCE_mean
gc.collect()


# In[ ]:


cash_CNT_INSTALMENT_mean = cash.groupby('SK_ID_PREV', as_index=False)['CNT_INSTALMENT'].mean().rename(columns = {'CNT_INSTALMENT': 'cash_CNT_INSTALMENT_mean'})
cash_stats_by_prev = cash_stats_by_prev.merge(cash_CNT_INSTALMENT_mean, on = 'SK_ID_PREV', how = 'left')
gc.enable()
del cash_CNT_INSTALMENT_mean
gc.collect()


# In[ ]:


cash_CNT_INSTALMENT_FUTURE_mean = cash.groupby('SK_ID_PREV', as_index=False)['CNT_INSTALMENT_FUTURE'].mean().rename(columns = {'CNT_INSTALMENT_FUTURE': 'cash_CNT_INSTALMENT_FUTURE_mean'})
cash_stats_by_prev = cash_stats_by_prev.merge(cash_CNT_INSTALMENT_FUTURE_mean, on = 'SK_ID_PREV', how = 'left')
gc.enable()
del cash_CNT_INSTALMENT_FUTURE_mean
gc.collect()


# In[ ]:


cash_SK_DPD_mean = cash.groupby('SK_ID_PREV', as_index=False)['SK_DPD'].mean().rename(columns = {'SK_DPD': 'cash_SK_DPD_mean'})
cash_stats_by_prev = cash_stats_by_prev.merge(cash_SK_DPD_mean, on = 'SK_ID_PREV', how = 'left')
gc.enable()
del cash_SK_DPD_mean
gc.collect()


# In[ ]:


cash_SK_DPD_DEF_mean = cash.groupby('SK_ID_PREV', as_index=False)['SK_DPD_DEF'].mean().rename(columns = {'SK_DPD_DEF': 'cash_SK_DPD_DEF_mean'})
cash_stats_by_prev = cash_stats_by_prev.merge(cash_SK_DPD_DEF_mean, on = 'SK_ID_PREV', how = 'left')
gc.enable()
del cash_SK_DPD_DEF_mean
gc.collect()


# In[ ]:


# now let's deal with our one categorical variable, NAME_CONTRACT_STATUS, in this cash file
cash_cats = pd.get_dummies(cash.select_dtypes('object'))
cash_cats['SK_ID_PREV'] = cash['SK_ID_PREV']
cash_cats.head()


# In[ ]:


cash_cats_grouped = cash_cats.groupby('SK_ID_PREV').agg('sum')
cash_cats_grouped.head()


# In[ ]:


cash_stats_by_prev = cash_stats_by_prev.merge(cash_cats_grouped, on = 'SK_ID_PREV', how = 'left')
gc.enable()
del cash_cats_grouped, cash_cats
gc.collect()


# In[ ]:


cash_stats_by_prev.head()


# In[ ]:


cash_MONTHS_BALANCE_count_mean = cash_stats_by_prev.groupby('SK_ID_CURR', as_index=False)['cash_MONTHS_BALANCE_count'].mean().rename(columns = {'cash_MONTHS_BALANCE_count': 'cash_MONTHS_BALANCE_count_mean'})
dataset = dataset.merge(cash_MONTHS_BALANCE_count_mean, on = 'SK_ID_CURR', how = 'left')
gc.enable()
del cash_MONTHS_BALANCE_count_mean
gc.collect()


# In[ ]:


cash_MONTHS_BALANCE_mean_mean = cash_stats_by_prev.groupby('SK_ID_CURR', as_index=False)['cash_MONTHS_BALANCE_mean'].mean().rename(columns = {'cash_MONTHS_BALANCE_mean': 'cash_MONTHS_BALANCE_mean_mean'})
dataset = dataset.merge(cash_MONTHS_BALANCE_mean_mean, on = 'SK_ID_CURR', how = 'left')
gc.enable()
del cash_MONTHS_BALANCE_mean_mean
gc.collect()


# In[ ]:


cash_CNT_INSTALMENT_mean_mean = cash_stats_by_prev.groupby('SK_ID_CURR', as_index=False)['cash_CNT_INSTALMENT_mean'].mean().rename(columns = {'cash_CNT_INSTALMENT_mean': 'cash_CNT_INSTALMENT_mean_mean'})
dataset = dataset.merge(cash_CNT_INSTALMENT_mean_mean, on = 'SK_ID_CURR', how = 'left')
gc.enable()
del cash_CNT_INSTALMENT_mean_mean
gc.collect()


# In[ ]:


cash_CNT_INSTALMENT_FUTURE_mean_mean = cash_stats_by_prev.groupby('SK_ID_CURR', as_index=False)['cash_CNT_INSTALMENT_FUTURE_mean'].mean().rename(columns = {'cash_CNT_INSTALMENT_FUTURE_mean': 'cash_CNT_INSTALMENT_FUTURE_mean_mean'})
dataset = dataset.merge(cash_CNT_INSTALMENT_FUTURE_mean_mean, on = 'SK_ID_CURR', how = 'left')
gc.enable()
del cash_CNT_INSTALMENT_FUTURE_mean_mean
gc.collect()


# In[ ]:


cash_SK_DPD_mean_mean = cash_stats_by_prev.groupby('SK_ID_CURR', as_index=False)['cash_SK_DPD_mean'].mean().rename(columns = {'cash_SK_DPD_mean': 'cash_SK_DPD_mean_mean'})
dataset = dataset.merge(cash_SK_DPD_mean_mean, on = 'SK_ID_CURR', how = 'left')
gc.enable()
del cash_SK_DPD_mean_mean
gc.collect()


# In[ ]:


cash_SK_DPD_DEF_mean_mean = cash_stats_by_prev.groupby('SK_ID_CURR', as_index=False)['cash_SK_DPD_DEF_mean'].mean().rename(columns = {'cash_SK_DPD_DEF_mean': 'cash_SK_DPD_DEF_mean_mean'})
dataset = dataset.merge(cash_SK_DPD_DEF_mean_mean, on = 'SK_ID_CURR', how = 'left')
gc.enable()
del cash_SK_DPD_DEF_mean_mean
gc.collect()


# In[ ]:


cash_NAME_CONTRACT_STATUS_Active_mean = cash_stats_by_prev.groupby('SK_ID_CURR', as_index=False)['NAME_CONTRACT_STATUS_Active'].mean().rename(columns = {'NAME_CONTRACT_STATUS_Active': 'cash_NAME_CONTRACT_STATUS_Active_mean'})
dataset = dataset.merge(cash_NAME_CONTRACT_STATUS_Active_mean, on = 'SK_ID_CURR', how = 'left')
gc.enable()
del cash_NAME_CONTRACT_STATUS_Active_mean
gc.collect()


# In[ ]:


cash_NAME_CONTRACT_STATUS_Amortized_mean = cash_stats_by_prev.groupby('SK_ID_CURR', as_index=False)['NAME_CONTRACT_STATUS_Amortized debt'].mean().rename(columns = {'NAME_CONTRACT_STATUS_Amortized debt': 'cash_NAME_CONTRACT_STATUS_Amortized_mean'})
dataset = dataset.merge(cash_NAME_CONTRACT_STATUS_Amortized_mean, on = 'SK_ID_CURR', how = 'left')
gc.enable()
del cash_NAME_CONTRACT_STATUS_Amortized_mean
gc.collect()


# In[ ]:


cash_NAME_CONTRACT_STATUS_Approved_mean = cash_stats_by_prev.groupby('SK_ID_CURR', as_index=False)['NAME_CONTRACT_STATUS_Approved'].mean().rename(columns = {'NAME_CONTRACT_STATUS_Approved': 'cash_NAME_CONTRACT_STATUS_Approved_mean'})
dataset = dataset.merge(cash_NAME_CONTRACT_STATUS_Approved_mean, on = 'SK_ID_CURR', how = 'left')
gc.enable()
del cash_NAME_CONTRACT_STATUS_Approved_mean
gc.collect()


# In[ ]:


cash_NAME_CONTRACT_STATUS_Canceled_mean = cash_stats_by_prev.groupby('SK_ID_CURR', as_index=False)['NAME_CONTRACT_STATUS_Canceled'].mean().rename(columns = {'NAME_CONTRACT_STATUS_Canceled': 'cash_NAME_CONTRACT_STATUS_Canceled_mean'})
dataset = dataset.merge(cash_NAME_CONTRACT_STATUS_Canceled_mean, on = 'SK_ID_CURR', how = 'left')
gc.enable()
del cash_NAME_CONTRACT_STATUS_Canceled_mean
gc.collect()


# In[ ]:


cash_NAME_CONTRACT_STATUS_Completed_mean = cash_stats_by_prev.groupby('SK_ID_CURR', as_index=False)['NAME_CONTRACT_STATUS_Completed'].mean().rename(columns = {'NAME_CONTRACT_STATUS_Completed': 'cash_NAME_CONTRACT_STATUS_Completed_mean'})
dataset = dataset.merge(cash_NAME_CONTRACT_STATUS_Completed_mean, on = 'SK_ID_CURR', how = 'left')
gc.enable()
del cash_NAME_CONTRACT_STATUS_Completed_mean
gc.collect()


# In[ ]:


cash_NAME_CONTRACT_STATUS_Demand_mean = cash_stats_by_prev.groupby('SK_ID_CURR', as_index=False)['NAME_CONTRACT_STATUS_Demand'].mean().rename(columns = {'NAME_CONTRACT_STATUS_Demand': 'cash_NAME_CONTRACT_STATUS_Demand_mean'})
dataset = dataset.merge(cash_NAME_CONTRACT_STATUS_Demand_mean, on = 'SK_ID_CURR', how = 'left')
gc.enable()
del cash_NAME_CONTRACT_STATUS_Demand_mean
gc.collect()


# In[ ]:


cash_NAME_CONTRACT_STATUS_Returned_mean = cash_stats_by_prev.groupby('SK_ID_CURR', as_index=False)['NAME_CONTRACT_STATUS_Returned to the store'].mean().rename(columns = {'NAME_CONTRACT_STATUS_Returned to the store': 'cash_NAME_CONTRACT_STATUS_Returned_mean'})
dataset = dataset.merge(cash_NAME_CONTRACT_STATUS_Returned_mean, on = 'SK_ID_CURR', how = 'left')
gc.enable()
del cash_NAME_CONTRACT_STATUS_Returned_mean
gc.collect()


# In[ ]:


cash_NAME_CONTRACT_STATUS_Signed_mean = cash_stats_by_prev.groupby('SK_ID_CURR', as_index=False)['NAME_CONTRACT_STATUS_Signed'].mean().rename(columns = {'NAME_CONTRACT_STATUS_Signed': 'cash_NAME_CONTRACT_STATUS_Signed_mean'})
dataset = dataset.merge(cash_NAME_CONTRACT_STATUS_Signed_mean, on = 'SK_ID_CURR', how = 'left')
gc.enable()
del cash_NAME_CONTRACT_STATUS_Signed_mean
gc.collect()


# In[ ]:


cash_NAME_CONTRACT_STATUS_XNA_mean = cash_stats_by_prev.groupby('SK_ID_CURR', as_index=False)['NAME_CONTRACT_STATUS_XNA'].mean().rename(columns = {'NAME_CONTRACT_STATUS_XNA': 'cash_NAME_CONTRACT_STATUS_XNA_mean'})
dataset = dataset.merge(cash_NAME_CONTRACT_STATUS_XNA_mean, on = 'SK_ID_CURR', how = 'left')
gc.enable()
del cash_NAME_CONTRACT_STATUS_XNA_mean
gc.collect()


# In[ ]:


# let's free up some memory by deleting some of the dataframes we are done with
gc.enable()
del cash, cash_stats_by_prev
gc.collect()


# In[ ]:


print('dataset data shape: ', dataset.shape)
dataset.head()


# # Feature Selection
# In this next section, I will work on reducing the number of variables.  I will do this by getting rid of collinear variables, variables with too many missing values, and feature importance.  After completing these steps, we will have a dataset better suited for modeling.

# In[ ]:


# let's review our dataset data types
dataset.dtypes.value_counts()


# In[ ]:


# it looks like we have a couple of objects still in our data?
dataset.describe(include=[np.object])


# In[ ]:


# our outlier variables appear to still be in the format of True or False, so we need to fix this before continuing.
dataset['bureau_DAYS_CREDIT_ENDDATE_max_outlier'] = dataset['bureau_DAYS_CREDIT_ENDDATE_max_outlier'].map({False:0, True:1})
dataset['bureau_DAYS_ENDDATE_FACT_mean_outlier'] = dataset['bureau_DAYS_ENDDATE_FACT_mean_outlier'].map({False:0, True:1})


# In[ ]:


# check again
dataset['bureau_DAYS_CREDIT_ENDDATE_max_outlier'].describe()


# In[ ]:


dataset['bureau_DAYS_ENDDATE_FACT_mean_outlier'].describe()


# ## Start with collinear variables

# In[ ]:


# because the dataset file is so large, let's use a subsample of the data to evaluate the collinear variables
y_temp = dataset[['TARGET']]
X_temp = dataset.drop(['TARGET'], axis=1)
X_big, X_small, y_big, y_small = train_test_split(X_temp, y_temp, test_size=0.2, random_state=1)


# In[ ]:


# let's first make the correlation matrix
corr = X_small.drop(['SK_ID_CURR'], axis=1)
corr_matrix = corr.corr().abs()


# In[ ]:


upper_corr = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
upper_corr.head()


# In[ ]:


# let's drop any columns with correlations above 0.9
drop_cols = [column for column in upper_corr.columns if any(upper_corr[column] > 0.9)]
print('Columns to remove: ', len(drop_cols))


# In[ ]:


# now we can drop these columns from the full dataset file
dataset = dataset.drop(columns = drop_cols)
print('dataset data shape: ', dataset.shape)


# In[ ]:


# delete the dataframes we don't need anymore
gc.enable()
del X_temp, X_big, X_small, y_temp, y_big, y_small, corr, corr_matrix, upper_corr, drop_cols
gc.collect()


# ## Next look at missing values

# In[ ]:


# missing values (in percent)
dataset_missing = (dataset.isnull().sum() / len(dataset)).sort_values(ascending = False)
dataset_missing.head(10)


# In[ ]:


# let's remove columns with more than 75% missing data
dataset_missing = dataset_missing.index[dataset_missing > 0.75]
print('Columns with more than 75% missing values: ', len(dataset_missing))


# In[ ]:


# let's drop these columns
dataset = dataset.drop(columns = dataset_missing)
print('dataset data shape: ', dataset.shape)


# Next we will look at feature importance.  But before we do so, we will need to split our data back into test and train.

# In[ ]:


# separate training and testing data for modeling
train = dataset[:train_len]
x_test = dataset[train_len:]
train_ids = train['SK_ID_CURR']
test_ids = x_test['SK_ID_CURR']
train.drop(columns=['SK_ID_CURR'], axis = 1, inplace=True)
x_test.drop(columns=['TARGET', 'SK_ID_CURR'], axis = 1, inplace=True)


# In[ ]:


# separate training data
train['TARGET'] = train['TARGET'].astype(int)
y_train = train['TARGET']
x_train = train.drop(columns=['TARGET'], axis = 1)


# In[ ]:


print('x_train data shape: ', x_train.shape)
print('y_train data shape: ', y_train.shape)
print('x_test data shape: ', x_test.shape)


# ## Feature Importance
# To evaluate feature importance, I will use the LightGBM model.  I will run the model twice to capture the feature importances, and then average the results.

# In[ ]:


# create a dataframe of all zeroes to hold feature importance calculations
feature_imp = np.zeros(x_train.shape[1])


# In[ ]:


# create the model to use
# for the parameters, objective is binary (as this is either default or no default that we are predicting),
# boosting type is gradient-based one-side sampling (larger gradients contribute more to information gain so this keeps those 
# with larger gradients and only randomly drops those with smaller), class weight is balanced
# (automatically adjust the weights to be inversely proportional to the frequencies)

model = lgb.LGBMClassifier(objective='binary', boosting_type='goss', n_estimators=10000, class_weight='balanced')


# In[ ]:


# we will fit the model twice and record the feature importances each time
# note that we will use auc (area under the curve) for evaluation, as on this is what our model will be judged

for i in range(2):
    train_x1, train_x2, train_y1, train_y2 = train_test_split(x_train, y_train, test_size = 0.25, random_state = i)
    model.fit(train_x1, train_y1, early_stopping_rounds=100, eval_set = [(train_x2, train_y2)], eval_metric = 'auc', verbose = 200)
    feature_imp += model.feature_importances_


# In[ ]:


# review features with most importance
feature_imp = feature_imp / 2
feature_imp = pd.DataFrame({'feature': list(x_train.columns), 'importance': feature_imp}).sort_values('importance', ascending = False)
feature_imp.head(10)


# In[ ]:


# review features with zero importance
zero_imp = list(feature_imp[feature_imp['importance'] == 0.0]['feature'])
print('count of features with 0 importance: ', len(zero_imp))
feature_imp.tail(10)


# In[ ]:


# let's drop the features with zero importance
x_train = x_train.drop(columns = zero_imp)
x_test = x_test.drop(columns = zero_imp)


# In[ ]:


print('x_train data shape: ', x_train.shape)
print('x_test data shape: ', x_test.shape)


# # Modeling
# Let's begin the modeling section now.  We will use LightGBM with cross validation.

# In[ ]:


# dataframe to hold predictions
test_predictions = np.zeros(x_test.shape[0])
# dataframe for out of fold validation predictions
out_of_fold = np.zeros(x_train.shape[0])
# lists for validation and training scores
valid_scores = []
train_scores = []


# In[ ]:


k_fold = KFold(n_splits = 5, shuffle = False, random_state = 50)


# In[ ]:


x_train = np.array(x_train)
x_test = np.array(x_test)


# In[ ]:


# iterate through each of the five folds
for train_indices, valid_indices in k_fold.split(x_train):
    train_features, train_labels = x_train[train_indices], y_train[train_indices]
    valid_features, valid_labels = x_train[valid_indices], y_train[valid_indices]
    
    # create the model, similar to the one used above for feature importances
    model = lgb.LGBMClassifier(n_estimators=10000, objective = 'binary', boosting_type='goss',class_weight = 'balanced', 
                               learning_rate = 0.05, reg_alpha = 0.1, reg_lambda = 0.1, n_jobs = -1, random_state = 50)
    
    # train the model
    model.fit(train_features, train_labels, eval_metric = 'auc',
              eval_set = [(valid_features, valid_labels), (train_features, train_labels)],
              eval_names = ['valid', 'train'], early_stopping_rounds = 100, verbose = 200)
    
    # record the best iteration
    best_iteration = model.best_iteration_
    
    # test predictions
    test_predictions += model.predict_proba(x_test, num_iteration = best_iteration)[:, 1] / k_fold.n_splits
    
    # out of fold predictions
    out_of_fold[valid_indices] = model.predict_proba(valid_features, num_iteration = best_iteration)[:, 1]
    
    # record scores
    valid_score = model.best_score_['valid']['auc']
    train_score = model.best_score_['train']['auc']
    valid_scores.append(valid_score)
    train_scores.append(train_score)
    
    # Clean up memory
    gc.enable()
    del model, train_features, valid_features
    gc.collect()


# In[ ]:


# scores
valid_auc = roc_auc_score(y_train, out_of_fold)

valid_scores.append(valid_auc)
train_scores.append(np.mean(train_scores))

fold_names = list(range(5))
fold_names.append('overall')

metrics = pd.DataFrame({'fold': fold_names, 'train': train_scores, 'valid': valid_scores}) 


# In[ ]:


metrics


# In[ ]:


# make submission file
submission = pd.DataFrame({'SK_ID_CURR': test_ids, 'TARGET': test_predictions})
submission.to_csv('submission.csv', index = False)

