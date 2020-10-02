#!/usr/bin/env python
# coding: utf-8

# # IEEE Fraud Detection Kaggle Competition
# ## Exploratory Data Analysis
# 
# The objective of this notebook is to get an initial sense of the data and investigate the following properties:
#      - What kind of data do we have at hand?
#      - Get a sense of the target and its distribution
#      - Missing data
#      - Correlation between target and features
#      - Differences between train and test

# In[ ]:


get_ipython().run_cell_magic('bash', '', '\nls -l')


# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', 500)


# In[ ]:


train_trans = pd.read_csv('../input/train_transaction.csv')
test_trans = pd.read_csv('../input/test_transaction.csv')


# In[ ]:


SEED = 5000
train_trans.sample(20, random_state=SEED)


# Initial observations:
# - from competition: "The TransactionDT feature is a timedelta from a given reference datetime (not an actual timestamp)." These could be seconds since transaction?
# - card features are categorical (from data tab) and related to physical(/virtual?) card used to make online purchase
# - Is TransactionAmt the same currency throughout? Could this be normalized?
# - addr1 is area? addr2 is country?. Worth plotting their distributions
# - P_emaildomain and R_emaildomain for purchasing email address and recipient email address? Could be Amazon/eBay sellers?
# - D features have a lot of missing values
# - M4 has values of M1, M2, M3, which are also features as well as some mystery M0
# - V features have a lot of missing values throughout in blocks. Seem to be mainly integer valued

# In[ ]:


train_trans.shape


# In[ ]:


len(train_trans['TransactionID'].unique())


# Target distribution. Significant class imbalance in training data. Most cases are not fraud. We saw this previously in Home Credit Risk competition. Some sort of downsampling/upsampling may be the way to go here.

# In[ ]:


fig = plt.figure(figsize=(10,7))
sns.countplot(train_trans['isFraud'])


# Calculating stats for the transaction data

# In[ ]:


train_trans_stats = train_trans.describe(include='all')


# In[ ]:


train_trans_stats.loc['max', 'TransactionDT']


# In[ ]:


train_trans_stats.loc['min', 'TransactionDT']


# Maybe these are seconds?

# In[ ]:


train_trans_stats.loc['max', 'TransactionDT'] / (60*60*24) / 30


# In[ ]:


train_trans_stats.loc['min', 'TransactionDT'] / (60*60*24)


# The earliest cut off could be about 6 months ago with the latest cut off being 24 hours ago

# In[ ]:


train_trans_stats.loc['na'] = train_trans.shape[0] - train_trans_stats.loc['count']


# Looking at distributions of card features:

# In[ ]:


fig, axes = plt.subplots(nrows=3, ncols=2, figsize = (15,20))

for ax, feature in zip(axes.flatten(), ['card' + str(x) for x in range(1,7)]):
    ax.bar(train_trans[feature].value_counts().index, train_trans[feature].value_counts().values)
    ax.set(title=feature.upper())


# A lot of visa cards and debit cards. Perhaps most fraud is committed using credit cards?

# In[ ]:


fig, axes = plt.subplots(nrows=1, ncols=2, figsize = (15,7), sharey=True)

for ax, card_type in zip(axes.flatten(), ['credit', 'debit']):
    ax.bar(train_trans[train_trans['card6'] == card_type]['isFraud'].value_counts().index,
           train_trans[train_trans['card6'] == card_type]['isFraud'].value_counts().values/\
           train_trans[train_trans['card6'] == card_type]['isFraud'].value_counts().sum())
    ax.set(title=card_type.upper())


# Interesting! Looks like some more fraudulent transactions with credit cards for sure. Let's look at a similar breakdown between Visa and Mastercard. 

# Definitely more fraud with credit cards alright. 

# In[ ]:


fig, axes = plt.subplots(nrows=1, ncols=2, figsize = (15,7), sharey=True)

for ax, card_type in zip(axes.flatten(), ['visa', 'mastercard']):
    ax.bar(train_trans[train_trans['card4'] == card_type]['isFraud'].value_counts().index,
           train_trans[train_trans['card4'] == card_type]['isFraud'].value_counts().values/\
           train_trans[train_trans['card4'] == card_type]['isFraud'].value_counts().sum())
    ax.set(title=card_type.upper())


# About the same across the board here.

# Looking at distributions of categorical variables

# In[ ]:


fig, axes = plt.subplots(nrows=2, ncols=3, figsize = (15,7))

for ax, feature in zip(axes.flatten(), ['ProductCD', 'addr1', 'addr2', 'dist1','dist2']):
    ax.bar(train_trans[feature].value_counts().index, train_trans[feature].value_counts().values)
    ax.set(title=feature.upper())


# ProductCD could be some sort of grouping for types of products purchased.
# 
# Interesting that almost all of the addr2 values lie at 87 - maybe this really is a country code from a pretty homogenous dataset. Doesn't look like addr1 is an area code - no 264 code in the States anyway.

# In[ ]:


p_emails = pd.DataFrame(data={'email_domains':train_trans['P_emaildomain'].value_counts().index,
                      'email_counts':train_trans['P_emaildomain'].value_counts().values})

r_emails = pd.DataFrame(data={'email_domains':train_trans['R_emaildomain'].value_counts().index,
                      'email_counts':train_trans['R_emaildomain'].value_counts().values})


# In[ ]:


fig = plt.figure(figsize=(15,10))

sns.set(style="whitegrid")

ax = sns.barplot(x='email_counts', y='email_domains', data=p_emails)

# Add a legend and informative axis label
ax.legend(ncol=2, loc="lower right", frameon=True)
ax.set(xlim=(0, 250000), ylabel="",
       xlabel="'P' email domains (purchaser?)")
sns.despine(left=True, bottom=True)


# In[ ]:


fig = plt.figure(figsize=(15,10))

sns.set(style="whitegrid")

ax = sns.barplot(x='email_counts', y='email_domains', data=r_emails)

# Add a legend and informative axis label
ax.legend(ncol=2, loc="lower right", frameon=True)
ax.set(xlim=(0, 250000), ylabel="",
       xlabel="'R' email domains (recipient?)")
sns.despine(left=True, bottom=True)


# Both domains dominated by gmail accounts. Let's see if theres any association between the domains and the label.

# In[ ]:


fig, axes = plt.subplots(nrows=30, ncols=2, figsize = (15,120), sharey=True)

for ax, email_domain in zip(axes.flatten(), p_emails['email_domains']):
    ax.bar(train_trans[train_trans['P_emaildomain'] == email_domain]['isFraud'].value_counts().index,
           train_trans[train_trans['P_emaildomain'] == email_domain]['isFraud'].value_counts().values/\
           train_trans[train_trans['P_emaildomain'] == email_domain]['isFraud'].value_counts().sum())
    ax.set(title=email_domain.upper())


# protonmail.com will be important for our model as well as mail.com, outlook.es, and aim.com to lesser extents. There aren't many of these domains in the P_emaildomain column though, let's check their counts in test and double check their counts in train

# In[ ]:


useful_p_domains = ['protonmail.com', 'mail.com', 'outlook.es', 'aim.com']


# In[ ]:


p_train_emails = p_emails[p_emails['email_domains'].isin(useful_p_domains)]
p_train_emails['email_counts %'] = (p_train_emails['email_counts']*100)/len(train_trans)
p_train_emails


# In[ ]:


p_test_emails = pd.DataFrame(data={'email_domains':test_trans['P_emaildomain'].value_counts().index,
                      'email_counts':test_trans['P_emaildomain'].value_counts().values})

p_test_emails = p_test_emails[p_test_emails['email_domains'].isin(useful_p_domains)]
p_test_emails['email_counts %'] = (p_test_emails['email_counts']*100)/len(test_trans)
p_test_emails


# Similar counts between the two datasets. Happy days!

# In[ ]:


fig, axes = plt.subplots(nrows=30, ncols=2, figsize = (15,120), sharey=True)

for ax, email_domain in zip(axes.flatten(), r_emails['email_domains']):
    ax.bar(train_trans[train_trans['R_emaildomain'] == email_domain]['isFraud'].value_counts().index,
           train_trans[train_trans['R_emaildomain'] == email_domain]['isFraud'].value_counts().values/\
           train_trans[train_trans['R_emaildomain'] == email_domain]['isFraud'].value_counts().sum())
    ax.set(title=email_domain.upper())


# In[ ]:


useful_r_domains = ['protonmail.com', 'mail.com', 'outlook.es', 'outlook.com', 'netzero.net']
r_train_emails = r_emails[r_emails['email_domains'].isin(useful_r_domains)]
r_train_emails['email_counts %'] = (r_train_emails['email_counts']*100)/len(train_trans)
r_train_emails


# In[ ]:


r_test_emails = pd.DataFrame(data={'email_domains':test_trans['R_emaildomain'].value_counts().index,
                      'email_counts':test_trans['R_emaildomain'].value_counts().values})

r_test_emails = r_test_emails[r_test_emails['email_domains'].isin(useful_r_domains)]
r_test_emails['email_counts %'] = (r_test_emails['email_counts']*100)/len(test_trans)
r_test_emails


# Same again here. Good to see.

# Let's check the distribution of transaction amounts

# In[ ]:


plt.figure(figsize=(15,9))
sns.distplot(train_trans['TransactionAmt'])


# In[ ]:


plt.figure(figsize=(15,9))
sns.boxenplot(train_trans['TransactionAmt'])


# Mostly transactions around 0 but looks like some extremely expensive ones too around 30000. Are these extreme transactions straight up fraud?

# In[ ]:


plt.figure(figsize=(15,9))
sns.countplot(train_trans[train_trans['TransactionAmt'] > 30000]['isFraud'])


# Nope. Could be missing values. Let's look at them:

# In[ ]:


train_trans[train_trans['TransactionAmt'] > 30000]


# Difficult to say what's going on here? Let's see if there's any similar rows in test.

# In[ ]:


plt.figure(figsize=(15,9))
sns.distplot(test_trans['TransactionAmt'])


# In[ ]:


plt.figure(figsize=(15,9))
sns.boxenplot(test_trans['TransactionAmt'])


# In[ ]:


train_trans['TransactionAmt'].describe()


# In[ ]:


test_trans['TransactionAmt'].describe()


# Nope. If anything, train is narrower around 0 than test is with the exception of the 30000 value outliers. Let's remove these and re-plot the train distribution

# In[ ]:


plt.figure(figsize=(15,9))
sns.distplot(train_trans[train_trans['TransactionAmt'] < 30000]['TransactionAmt'])


# In[ ]:


plt.figure(figsize=(15,9))
sns.boxenplot(train_trans[train_trans['TransactionAmt'] < 30000]['TransactionAmt'])


# That makes a lot more sense. It might be worthwhile to obtain a z-score for each value in this feature so we can easily see the variance between points

# Looking at transactions above the 75th percentile

# In[ ]:


train_trans[train_trans['TransactionAmt'] > 125]['isFraud'].value_counts()


# In[ ]:


fig, axes = plt.subplots(nrows=1, ncols=2, figsize = (15,7), sharey=True)

axes = axes.flatten()

axes[0].bar(train_trans[train_trans['TransactionAmt'] > 125]['isFraud'].value_counts().index,
            train_trans[train_trans['TransactionAmt'] > 125]['isFraud'].value_counts().values/\
            train_trans[train_trans['TransactionAmt'] > 125]['isFraud'].value_counts().sum())
axes[0].set(title='Transaction amounts > 75th percentile'.upper(),
            ylabel = 'Normalized count',
            xlabel = 'isFraud')

axes[1].bar(train_trans[train_trans['TransactionAmt'] < 125]['isFraud'].value_counts().index,
            train_trans[train_trans['TransactionAmt'] < 125]['isFraud'].value_counts().values/\
            train_trans[train_trans['TransactionAmt'] < 125]['isFraud'].value_counts().sum())
axes[1].set(title='Transaction amounts < 75th percentile'.upper(),
            ylabel = 'Normalized count',
            xlabel = 'isFraud')


# Slightly more fraudulent transactions above 125 - makes sense. If you're going to commit fraud, go big?

# Looking at missing values among features:

# In[ ]:


fig = plt.figure(figsize=(15,5))
ax = sns.barplot(x=train_trans_stats.columns, y=train_trans_stats.loc['na'])


# Quite a lot of missing values spread throughout dataset. A lot of features, that are presumably similar - particularly towards the right side of plot are consistently missing most of their values so this likely isn't an error in recording. The values are probably missing for a reason.

# In[ ]:


fig = plt.figure(figsize=(15,10))
ax = sns.barplot(x=train_trans_stats.loc[:, (train_trans_stats.loc['na'] > 0) & (train_trans_stats.loc['na'] < 1500)].columns,
                 y=train_trans_stats.loc[:, (train_trans_stats.loc['na'] > 0) & (train_trans_stats.loc['na'] < 1500)].loc['na'])


# These features with only a few missing values (comparitively) are also missing equal numbers of values so this probably isn't recording error.

# In[ ]:


'No. of featues without NaNs: {}'.format(len(train_trans_stats.loc[:, train_trans_stats.loc['na'] == 0].columns))


# Digging into features that are missing most of their values

# In[ ]:


fig = plt.figure(figsize=(15,10))
ax = sns.barplot(x=train_trans_stats.loc[:, train_trans_stats.loc['na'] > 450000].columns,
                 y=train_trans_stats.loc[:, train_trans_stats.loc['na'] > 450000].loc['na'])


# In[ ]:


train_trans_stats.loc[:, train_trans_stats.loc['na'] > 450000].columns.values


# Let's look into some of these "V" features

# In[ ]:


v_features = train_trans_stats.loc[:, train_trans_stats.loc['na'] > 450000].columns.values[9:].tolist()


# In[ ]:


len(v_features)


# Looking at first 30:

# In[ ]:


fig = plt.figure(figsize=(15,10))
ax = sns.barplot(x=v_features[:30],
                 y=train_trans_stats.loc[:, v_features[:30]].loc['na'])


# In[ ]:


train_trans['V138'].dtype


# In[ ]:


plt.figure(figsize=(15,10))
ax = sns.violinplot(data=train_trans.loc[:, v_features])


# Looks like even among the features that have the exact same number of missing values, we have pretty different distributions - but it's a bit difficult to tell from this plot due to differences in scale 

# In[ ]:


train_trans.loc[:, v_features].head(10)


# Looks like these features have simultaneous missing values - it will be interesting to see how important these anonymised continuous features are to any models we train. Another interesting point to note is that each of these features seems to have integer values despite being decimal value types in the data.

# Let's plot all of their distributions

# In[ ]:


fig, axes = plt.subplots(nrows=36, ncols=4, figsize = (15,120))

for ax, feature in zip(axes.flatten(), v_features):
    ax.bar(train_trans[feature].value_counts().index, train_trans[feature].value_counts().values)
    ax.set(title=feature.upper())


# Interesting that all the "V" features have values primarily at 0 or 1 have a positive skew. Again note that these are integers. It looks like they may be counts of values whose existence may possibly be determined by a categorical variable or variables elsewhere in the data. Let's see if this is similar for the "D" features that are missing a lot of values:

# In[ ]:


d_features = train_trans_stats.loc[:, train_trans_stats.loc['na'] > 450000].columns.values[2:8].tolist()


# In[ ]:


d_features


# In[ ]:


import matplotlib as mpl
import importlib
importlib.reload(mpl); importlib.reload(plt); importlib.reload(sns)


# In[ ]:


fig, axes = plt.subplots(nrows=2, ncols=3, figsize = (15,10))

for ax, feature in zip(axes.flatten(), d_features):
    ax.bar(train_trans[feature].value_counts().index, train_trans[feature].value_counts().values)
    ax.set(title=feature.upper())


# Looks like each of the D features follow a similar pattern except for D9 - let's remove this feature and re-plot

# In[ ]:


fig, axes = plt.subplots(nrows=2, ncols=3, figsize = (15,10))

for ax, feature in zip(axes.flatten(), [feature for feature in d_features if feature not in 'D9']):
    ax.bar(train_trans[feature].value_counts().index, train_trans[feature].value_counts().values)
    ax.set(title=feature.upper())


# Yep - looks like a similar pattern here with even more of a concentration of values at 0. Let's see if we're still dealing with integers:

# In[ ]:


train_trans_stats[d_features]


# Each of the features is an int feature except for D9, which looks to be some sort of ratio or likelihood, and D8.

# In[ ]:


fig = plt.figure(figsize=(15,10))
ax = sns.barplot(x=train_trans_stats.columns[:60], y=train_trans_stats.loc['na'][:60])


# These features among the first 60 may be missing values that can be imputed since the number of values missing don't seem to be systematic. These features are among:

# Looking at the features with most missing values here:

# In[ ]:


fig, axes = plt.subplots(nrows=2, ncols=4, figsize = (15,7))

for ax, feature in zip(axes.flatten(),
                       train_trans.iloc[:, :60].loc[:, train_trans_stats.loc['na'][:60] > 500000].columns):
    
    ax.bar(train_trans[feature].value_counts().index, train_trans[feature].value_counts().values)
    ax.set(title=feature.upper())


# Mostly the "D" features again. How many of these are there?

# In[ ]:


len([column for column in train_trans.columns.tolist() if column.startswith('D')])


# In[ ]:


fig, axes = plt.subplots(nrows=4, ncols=4, figsize = (15,14))

for ax, feature in zip(axes.flatten(),
                       [column for column in train_trans.columns.tolist() if column.startswith('D')]):
    
    ax.bar(train_trans[feature].value_counts().index, train_trans[feature].value_counts().values)
    ax.set(title=feature.upper())


# Despite a large number of missing values, most of these features (except for D9) seem to be positively skewed with the majority of their values around 0. Are most of them ints?

# In[ ]:


train_trans_stats.loc[:, [column for column in train_trans.columns.tolist() if column.startswith('D')]]


# Yep looks like most of them are

# Quickly looking at the most positive and negative correlations with the target

# In[ ]:


correlations = train_trans.corr()['isFraud'].sort_values()


# In[ ]:


neg_corrs = correlations.head(10)
pos_corrs = correlations.tail(10)


# In[ ]:


corrs = pos_corrs.append(neg_corrs)


# In[ ]:


corrs


# In[ ]:


train_trans[corrs.index].corr()


# In[ ]:


# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(train_trans[corrs.index].corr(),  cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})


# Looks like too many simultaneous missing values between some of our features to calculate a correlation coefficient. Perhaps these features are dependent upon each other or another feature and cannot coexist.

# In[ ]:




