#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('ggplot')

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold
import lightgbm as lgb

import warnings
warnings.filterwarnings('ignore')


# In[ ]:


train = pd.read_csv('../input/train.csv', index_col='Id')
test = pd.read_csv('../input/test.csv', index_col='Id')


# In[ ]:


train.head()


# In[ ]:


test.head()


# ## Exploratory Analysis

# I like to check for missing values in a data set before I start the exploratory analysis. A large number of missing values in a particular feature can sometimes lead to erroneous conclusions when looking at summary statistics. 

# In[ ]:


missing = train.isnull().sum().sort_values(ascending=False)
missing= (missing[missing > 0] / train.shape[0])
ax = missing.round(3).plot.bar();
ax.set_title('% Missing values')
vals = ax.get_yticks()
ax.set_yticklabels(['{:,.0%}'.format(x) for x in vals]);


# Fortunately, only five of the features have any missing values in the training data. Of these only three have a large number of missing values. These are:
# * 'rez_esc' - year behind in schooling
# * 'v18q1' - owns a tablet
# * 'v2a1' - monthly rent payments
# 
# Because there are so many missing points in these features, we'd probably want to encode them somehow rather than attempt to impute them. The remaining two:
# * 'meaneduc' - average years of education for adults
# * 'SQBmeaned' - square of the mean years of education of adults
# 
# only have a few points missing, so we might try to impute them using the mean or median for example. 

# Let's move on to exploring some of the features. The data set has an id for each person as well as for each household. We can get the number of people by household by grouping by the household id ('idhogar') and getting a count. Here I do so and then plot a histogram:

# In[ ]:


plt.figure(figsize=(10,5));
train.groupby('idhogar')['idhogar'].count().hist(bins=10);
plt.title('Distribution of household size');
plt.xlabel('Household size');


# Let's look at the first feature 'v2a1' or monthly rent payment. How is monthly rent distributed among the four poverty levels? Due to the presence of a possible outlier in in monthly rent payment, the box plots get squished vertically. I've plotted them with and without the outlier. 

# In[ ]:


fig, (ax1, ax2) = plt.subplots(1, 2);
sns.boxplot(x='Target', y='v2a1', data=train, ax=ax1);
sns.boxplot(x='Target', y='v2a1', data=train[train.v2a1 < 2000000], ax=ax2);
ax1.set_title('With outlier');
ax2.set_title('Without outlier');
ax1.set_xlabel('Poverty level');
ax2.set_xlabel('Poverty level');
ax1.set_ylabel('Monthyl rent payment');
ax2.set_ylabel('');
fig.suptitle('Distribution of monthly rent payment by poverty level', x=1, y=1, fontsize=16);
fig.subplots_adjust(left=0.1, right=2, wspace=0.3);


# It is not entirely clear from the boxplots, but it appears that the median monthly rent payment increases with poverty level. This makes sense as a higher numerical value for the poverty level actually indicates a lower level of poverty. Let's verify this by looking at the medians be poverty level:

# In[ ]:


train.groupby('Target')['v2a1'].median()


# Actually, poverty levels 2 and 3 have the same median monthly rent, at least in this particular sample of data.

# The next variable in the data set, 'rooms', is the total number of rooms in the individual's house. Let's take a look at how rooms are distributed by poverty level. Since this is individual level data, not household level, we'll first need to group by the household id 'idhogar', otherwise, we will be double-counting. 

# In[ ]:


train_by_hhid = train.groupby('idhogar')
rm_by_id = train_by_hhid['Target', 'rooms'].first()

plt.figure(figsize=(10, 5));
sns.countplot(x='rooms', hue='Target', data=rm_by_id);
plt.title('Distribution of rooms in house by poverty level');
plt.xlabel('Number of rooms');


# Here we see that for poverty level 4 (non-vulnerable) the most common number of rooms is 5. There are also a significant number of non-vulnerable houses with 7-11 rooms. For poverty levels 1, 2, and 3 (moderate and vulnerable) the most common number of rooms is 4. In the homes with only 1 or 2 rooms, there are a significant number of households in each poverty level. This indicates that the number of rooms alone may not be great at discerning poverty levels in the lower extreme. 

# The next two variables - 'v14a' and 'refrig' - indicate whether the individual lives in a house with a bathroom or a refrigerator respectively. Let's take a look at the distribution of these across poverty types. 

# In[ ]:


bathroom_by_id = train_by_hhid['Target', 'v14a', 'refrig'].first()

fig, (ax1, ax2) = plt.subplots(1, 2)
sns.countplot(x='v14a', hue='Target', data=bathroom_by_id, ax=ax1);
sns.countplot(x='refrig', hue='Target', data=bathroom_by_id, ax=ax2);
ax1.set_xticklabels(['No', 'Yes']);
ax2.set_xticklabels(['No', 'Yes']);
ax1.set_xlabel('Has bathroom');
ax2.set_ylabel('Has Refrigerator');
fig.subplots_adjust(left=0.1, right=2)
fig.suptitle('Distribution of Bathrooms and Refrigerators by Poverty Level', x=1, y=1, fontsize=16);


# In[ ]:


fig, (ax1, ax2) = plt.subplots(1, 2);
sns.boxplot(x='Target', y='meaneduc', data=train, ax=ax1);
sns.violinplot(x='Target', y='meaneduc', data=train, ax=ax2);

ax1.set_xlabel('Poverty level');
ax2.set_xlabel('Poverty level');
ax1.set_ylabel('Mean years education');
ax2.set_ylabel('');
fig.suptitle('Distribution of mean years education by poverty level', x=1, y=1, fontsize=16);
fig.subplots_adjust(left=0.1, right=2, wspace=0.3);


# ## Feature Engineering

# In[ ]:


#outlier in test set which rez_esc is 99.0
test.loc[test['rez_esc'] == 99.0 , 'rez_esc'] = 5


# In[ ]:


# correct entries that were not delabeled as per discussion thread
relabel_cols = ['edjefe', 'edjefa', 'dependency']

train[relabel_cols] = train[relabel_cols].replace({'yes':1, 'no':1}).astype(float)
test[relabel_cols] = test[relabel_cols].replace({'yes':1, 'no':1}).astype(float)


# In[ ]:


# set monthly rent payment to 0 where hh owns home
train.loc[train.tipovivi1 == 1, 'v2a1'] = 0
test.loc[test.tipovivi1 == 1, 'v2a1'] = 0


# In[ ]:


# dictionary of columns and aggregation method
agg_dict = {'escolari':np.sum, 'rez_esc':np.sum, 'age':np.sum, 'estadocivil1':np.sum}

# group by household and apply aggregtion methods
train_by_hh = train.groupby('idhogar').agg(agg_dict)
test_by_hh = test.groupby('idhogar').agg(agg_dict)

# join household level data with individual level data
train = train.join(train_by_hh, on='idhogar', rsuffix='_hh')
test = test.join(test_by_hh, on='idhogar', rsuffix='_hh')


# In[ ]:


# per capita monthly rent
train['rent_per_cap'] = train['v2a1'] / train['tamhog']
test['rent_per_cap'] = test['v2a1'] / test['tamhog']

# per capital tablets
train['tab_per_cap'] = train['v18q'] / train['tamhog']
test['tab_per_cap'] = test['v18q'] / test['tamhog']

# male-female ratio of hh
train['mf_rat'] = train['r4h3'] / train['r4m3']
test['mf_rat'] = test['r4h3'] / test['r4m3']

train['walls_roof_bad'] = train['epared1'] + train['eviv1']
test['walls_roof_bad'] = test['epared1'] + test['eviv1']

# percent of hh under 12 years old
train['child_perc'] = ( train['r4h1'] + train['r4m1'] ) / train['r4t3']
test['child_perc'] = ( test['r4h1'] + test['r4m1'] ) / test['r4t3']

#share of children under 19 that are 12 or under
train['young_perc'] = train['r4t1'] / train['hogar_nin']
test['young_perc'] = test['r4t1'] / test['hogar_nin']

#number of children per adult
train['child_per_adult'] = train['hogar_nin'] / train['hogar_adul']
test['child_per_adult'] = test['hogar_nin'] / test['hogar_adul']

# number of 65+ as percent of total
train['older_perc'] = train['hogar_mayor'] / train['tamviv']
test['older_perc'] = test['hogar_mayor'] / test['tamviv']

# difference between number of poeple living in hh and hh members
train['tamdiff'] = train['tamhog'] - train['tamviv']
test['tamdiff'] = test['tamhog'] - test['tamviv']

## hh has computer and/or television
train['comp_tv'] = train['computer'] + train['television']
test['comp_tv'] = test['computer'] + test['television']


# For now, let's just replace NaNs with -1. 

# In[ ]:


# replace NaNs in train and test data with -1
train.fillna(-1, inplace=True)
test.fillna(-1, inplace=True)

train.replace([np.inf, -np.inf], -1, inplace=True)
test.replace([np.inf, -np.inf], -1, inplace=True)


# In[ ]:


# Create the feature and target arrays for sklearn
keep_cols = [col for col in train.columns if col[:3] != 'SQB']
keep_cols = [col for col in keep_cols if col not in ['idhogar', 'agesq', 'Target']]

X = train.loc[:, keep_cols].values
y = train.Target.values

X_test = test.loc[:, keep_cols].values


# Let's wrap up by running a quick GBM model to get the feature importances:

# In[ ]:


gbm = GradientBoostingClassifier(n_estimators=100).fit(X,y)


# In[ ]:


fi = pd.DataFrame({
    'importance':gbm.feature_importances_.round(5)
}, index=train.loc[:, keep_cols].columns)

fi.sort_values('importance', ascending=False, inplace=True)

fi.iloc[1:30, ].plot.bar(legend=None, figsize=(17,5));
plt.title('Feature Importance');


# ## Modeling

# In[ ]:


skf = StratifiedKFold(n_splits=3)
skf.split(X, y)


# In[ ]:


logmod = LogisticRegression(class_weight='balanced')

param_grid = { 'C': [0.5, 1] }

log_grid = GridSearchCV(logmod, cv=skf, param_grid=param_grid, scoring='f1_macro')
log_grid.fit(X, y);


# In[ ]:


print(log_grid.best_params_)
print(log_grid.best_score_)


# In[ ]:


params = {
    'learning_rate': [0.01, 0.1],
    'max_depth': [3, 4, 5, 6],
    'n_estimators': [500, 1000],
    'colsample_bytree':[0.5, 1],
}


# In[ ]:


lgb_mod = lgb.LGBMClassifier(class_weight='balanced')

lgbm_grid = GridSearchCV(estimator=lgb_mod, cv=skf, param_grid=params, scoring='f1_macro', verbose=1, n_jobs=2)
lgbm_grid.fit(X, y);


# In[ ]:


print('Best lightgbm parameters: {}'.format(lgbm_grid.best_params_))
print('Best tuning score: {}'.format(lgbm_grid.best_score_))


# In[ ]:


preds_log = log_grid.predict(X_test)
preds_lgbm = lgbm_grid.predict(X_test)


# In[ ]:


# plot distribution of predictions from models
plt.subplot(2, 2, 1)
sns.countplot(y, color='red')
plt.title('Distribution of Training Labels')
plt.subplot(2, 2, 2)
sns.countplot(preds_log, color='red')
plt.title('Distribution of Logistic Predictions')
plt.subplot(2, 2, 3)
sns.countplot(preds_lgbm, color='red')
plt.title('Distribution of LightGBM Predictions')
plt.subplots_adjust(top=2, right=2)


# In[ ]:


# create final submission dataframe and write to csv
sub_log = pd.DataFrame({'Id':test.index, 'Target':preds_log})
sub_lgbm = pd.DataFrame({'Id':test.index, 'Target':preds_lgbm})

#sub_log.to_csv('sub_log1.csv', index=False)
sub_lgbm.to_csv('sub_lgbm1.csv', index=False)

