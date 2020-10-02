#!/usr/bin/env python
# coding: utf-8

# <h1 style="text-align:center; color:DarkBlue">Fraud Detection</h1>
# 
# <img src="https://upload.wikimedia.org/wikipedia/en/thumb/2/21/IEEE_logo.svg/500px-IEEE_logo.svg.png" style="display:block;margin-left:25%;margin-right:auto;width:50%;"/>
# 
# <div style="margin-left: 10px">
# <a style="cursor:pointer">1. Intorduction</a><br>
# <a style="cursor:pointer">2. IEEE Fraud Detection</a><br>
# <a style="cursor:pointer">3. Import Libraries</a><br>
# <a style="cursor:pointer">4. Loading Files</a><br>
# <a style="cursor:pointer">5. Reduce Memory Size</a><br>
# <a style="cursor:pointer">6. Exploring Data</a><br>
#     <a style="cursor:pointer">&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; <span>&#8226;</span>&emsp;Visualization</a><br>
#     <a style="cursor:pointer">&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;<span>&#8226;</span>&emsp;Delve into TransactionDT</a><br>
#     <a style="cursor:pointer">&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;<span>&#8226;</span>&emsp;Check Device Info</a><br>
#     <a style="cursor:pointer">&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;<span>&#8226;</span>&emsp;Check Email Address</a><br>
# <a style="cursor:pointer">7. Feature Engineering</a><br>
# <a style="cursor:pointer">8. Prepare Data for Modeling</a><br>
#     <a style="cursor:pointer">&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;<span>&#8226;</span>&emsp;Fill Nan Values</a><br>
#     <a style="cursor:pointer">&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;<span>&#8226;</span>&emsp;Label Encoding</a><br>
# <a style="cursor:pointer">9. Create Test Set and Train Set</a><br>
# <a style="cursor:pointer">10. Finding Best Model</a>
# </div>
# 
# <br>
# 
# Thanks to this kernels (Refrences) : 
# * [feature-engineering-lightgbm-corrected](https://www.kaggle.com/davidcairuz/feature-engineering-lightgbm-corrected")
# * [eda-and-models](https://www.kaggle.com/artgor/eda-and-models)
# * [feature-engineering-lightgbm-corrected](https://www.kaggle.com/davidcairuz/feature-engineering-lightgbm-corrected)
# * [reducing-memory-size-for-ieee](https://www.kaggle.com/mjbahmani/reducing-memory-size-for-ieee)
# * [day-and-time-powerful-predictive-feature](https://www.kaggle.com/fchmiel/day-and-time-powerful-predictive-feature)
# 
# <br>
# 
# ## Intorduction
# In law, `fraud` is intentional deception to secure unfair or unlawful gain, or to deprive a victim of a legal right. Fraud can violate civil law (i.e., a fraud victim may sue the fraud perpetrator to avoid the fraud or recover monetary compensation), a criminal law (i.e., a fraud perpetrator may be prosecuted and imprisoned by governmental authorities), or it may cause no loss of money, property or legal right but still be an element of another civil or criminal wrong.The purpose of fraud may be monetary gain or other benefits, for example by obtaining a passport, travel document, or driver's license, or mortgage fraud, where the perpetrator may attempt to qualify for a mortgage by way of false statements. [Fraud](https://en.wikipedia.org/wiki/Fraud)
# <br>
# 
# `Fraud detection` is a set of activities undertaken to prevent money or property from being obtained through false pretenses. Fraud detection is applied to many industries such as banking or insurance. In banking, fraud may include forging checks or using stolen credit cards. Other forms of fraud may involve exaggerating losses or causing an accident with the sole intent for the payout.<br> [Data analysis techniques for fraud detection](https://en.wikipedia.org/wiki/Data_analysistechniques_for_fraud_detection)

# ## IEEE Fraud Detection
# <div style="text-align: justify">In this competition, we will benchmark `machine learning models` on a challenging large-scale dataset. The data comes from Vesta's real-world e-commerce transactions and contains a wide range of features from device type to product features. We also have the opportunity to create new features to improve your results. [Vesta Coopration](https://trustvesta.com/)</div>
# <br>
# 
# <img src="http://news.mit.edu/sites/mit.edu.newsoffice/files/styles/news_article_image_top_slideshow/public/images/2018/MIT-Fraud-Detection-PRESS_0.jpg?itok=laiU-5nR" style="display:block;margin-left:25%;margin-right:auto;width:50%;"/>

# We have to use classification method on this dataset and find out which of these instances are seem to be fraud; actually, we should find the probability of being fraud for each of these instances.
# <br><br>
# As you can see the dataset, it is too big and working on that may take a lot of time, therefore we should reduce the size of data either by using PCA(Principle Component Analysis) or by downcasting integer and float columns.
# <br>
# <br>
# The submission is measured with calculating  `AUC - ROC Curve`
# <br><br>
# <b>What is AUC - ROC Curve?</b><br>
# AUC - ROC curve is a performance measurement for classification problem at various thresholds settings. ROC is a probability curve and AUC represents degree or measure of separability. It tells how much model is capable of distinguishing between classes. Higher the AUC, better the model is at predicting 0s as 0s and 1s as 1s. By analogy, Higher the AUC, better the model is at distinguishing between patients with disease and no disease.
# The ROC curve is plotted with TPR against the FPR where TPR is on y-axis and FPR is on the x-axis.<br>[Understanding AUC - ROC Curve](https://towardsdatascience.com/understanding-auc-roc-curve-68b2303cc9c5)
# <br>
# 
# <img src="https://miro.medium.com/max/722/1*pk05QGzoWhCgRiiFbz-oKQ.png" style="display:block;margin-left:25%;margin-right:auto;width:40%;"/>

# ## Import Libraries

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import gc
from itertools import cycle, islice
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold


# ## Loading Files
# 
# First we should check what files we have in input directory.
# 
# We have four files two of which are train and the other two are test files.

# In[ ]:


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# Then we should load csv datasets in pandas' dataframes.
# 
# `TransactionID` is the index of all files, so we use index_col to use this column when loading files.

# In[ ]:


train_transaction = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_transaction.csv', index_col='TransactionID')
test_transaction = pd.read_csv('/kaggle/input/ieee-fraud-detection/test_transaction.csv', index_col='TransactionID')
train_identity = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_identity.csv', index_col='TransactionID')
test_identity = pd.read_csv('/kaggle/input/ieee-fraud-detection/test_identity.csv', index_col='TransactionID')
sample_submission = pd.read_csv('/kaggle/input/ieee-fraud-detection/sample_submission.csv', index_col='TransactionID')


# Now we should `join` train and test dataset.
# 
# We use merge method.

# In[ ]:


train = train_transaction.merge(train_identity, how='left', left_index=True, right_index=True)
test = test_transaction.merge(test_identity, how='left', left_index=True, right_index=True)


# ## Reduce Memory Size
# 
# As mentioned above, this dataset is feeding on memory, so we should use anything to reduce memory usage.
# 
# One we to dimnish memory usage is calling `garbage collector` just after we don't need a dataframe.
# 
# To dimnish `memory usage`, we delete previous dataframes and call garbage collector to collet this unrefrence data.

# In[ ]:


del train_transaction, train_identity
del test_transaction, test_identity
gc.collect()


# Another way which is told earlier is to `downcast` float and integer columns.
# 
# This is a function that downcast the float columns

# In[ ]:


def downcast_df_float_columns(df):
    list_of_columns = list(df.select_dtypes(include=["float64"]).columns)
        
    if len(list_of_columns)>=1:
        max_string_length = max([len(col) for col in list_of_columns])
        print("downcasting float for:", list_of_columns, "\n")
        
        for col in list_of_columns:
            df[col] = pd.to_numeric(df[col], downcast="float")
    else:
        print("no columns to downcast")
    gc.collect()
    print("done")


# This is a function that downcast the integer columns

# In[ ]:


def downcast_df_int_columns(df):
    list_of_columns = list(df.select_dtypes(include=["int32", "int64"]).columns)
        
    if len(list_of_columns)>=1:
        max_string_length = max([len(col) for col in list_of_columns])
        print("downcasting integers for:", list_of_columns, "\n")
        
        for col in list_of_columns:
            df[col] = pd.to_numeric(df[col], downcast="integer")
    else:
        print("no columns to downcast")
    gc.collect()
    print("done")


# ## Exploring Data

# In[ ]:


train.head()


# Train dataset has 399 float, 3 int and 31 object(category) columns.

# In[ ]:


train.info()


# In[ ]:


train.describe()


# In[ ]:


plt.bar(['train', 'test'], [len(train), len(test)], width=0.2, color='b')
plt.title("Number of train set and test set instances ")
plt.show()


# There are more columns with at least one null value in train set than in test set

# In[ ]:


plt.bar(['train', 'test'], [train.isnull().any().sum(), test.isnull().any().sum()], width=0.2, color='g')
plt.title("Number of column with null values")
plt.show()


# Find missing values in train and test set
# 
# These are columns with the most `null instances` in both train set and test set

# In[ ]:


train_missing_values = train.isnull().sum().sort_values(ascending=False) / len(train)
test_missing_values = test.isnull().sum().sort_values(ascending=False) / len(test)

fig, axes = plt.subplots(2, 1, figsize=(12, 8))
sns.barplot(list(train_missing_values.keys()[:10]), train_missing_values[:10], ax=axes[0])
sns.barplot(list(test_missing_values.keys()[:10]), test_missing_values[:10], ax=axes[1])
plt.show()


# Show the percentage of fraud and not-fraud instances
# 
# We can see up to 97% of instances are not fruad and only 3% of data are labeled fraud; as a result, we ought to take care of our model not being `overfitted`

# In[ ]:


def show_values_on_bars(axs):
    def _show_on_single_plot(ax):        
        for p in ax.patches:
            _x = p.get_x() + p.get_width() / 2
            _y = p.get_y() + p.get_height()
            value = '{:.2f}'.format(p.get_height())
            ax.text(_x, _y, value, ha="center") 

    if isinstance(axs, np.ndarray):
        for idx, ax in np.ndenumerate(axs):
            _show_on_single_plot(ax)
    else:
        _show_on_single_plot(axs)

plt.figure(figsize=(5, 4))
ax = sns.barplot(["fraud", "not fraud"],
            [len(train[train.isFraud == 1])/len(train),
             len(train[train.isFraud == 0])/len(train)])
show_values_on_bars(ax)
plt.show()


# ### Delve into TransactionDT
# 
# As we can see in **[this](https://www.kaggle.com/c/ieee-fraud-detection/discussion/100071#latest-577632)** discussion, probably TransactionDT column is in seconds.
# 
# We can add day's hour to out dataset. **[Day and Time - powerful predictive feature?)](https://www.kaggle.com/fchmiel/day-and-time-powerful-predictive-feature)**

# In[ ]:


train["hour"] = np.floor(train["TransactionDT"] / 3600) % 24
test["hour"] = np.floor(train["TransactionDT"] / 3600) % 24


# We can `visualize` to see if there is any relationship between day's hour and fraud
# 
# It's showing that in which hours, the rate of fraud is `greater`

# In[ ]:


plt.plot(train.groupby('hour').mean()['isFraud'], color='r')
ax = plt.gca()
ax2 = ax.twinx()
_ = ax2.hist(train['hour'], alpha=0.3, bins=24)
ax.set_xlabel('Encoded hour')
ax.set_ylabel('Fraction of fraudulent transactions')

ax2.set_ylabel('Number of transactions')
plt.show()


# ### Check Device Info

# Check which type of devices has been used

# In[ ]:


train["DeviceType"].value_counts(dropna=False).plot.bar()
plt.show()


# We can see Windows, iOS and MacOS are the most popular operating systems

# In[ ]:


plt.figure(figsize=(8, 8))
sns.barplot(train["DeviceInfo"].value_counts(dropna=False)[:15], 
            train["DeviceInfo"].value_counts(dropna=False).keys()[:15])
plt.show()


# ### Check Email Address

# In[ ]:


my_colors = list(islice(cycle(['b', 'r', 'g', 'y', 'k']), None, len(train.P_emaildomain.value_counts())))
train.P_emaildomain.value_counts().plot.bar(figsize=(20, 10), color=my_colors)
plt.show()


# We can see that more than 40% of instnces which have `P_emaildomain` equal to `protonmail.com` are Fraud and this shows that we should check if P_emaildomain or R_emaildomain is equal to protonmail.com or not

# In[ ]:


plt.figure(figsize=(6, 6))
plt.pie([np.sum(train[(train['P_emaildomain'] == 'protonmail.com')].isFraud.values),
                                 len(train[(train['P_emaildomain'] == 'protonmail.com')].isFraud.values) - 
                                 np.sum(train[(train['P_emaildomain'] == 'protonmail.com')].isFraud.values)],
        labels=['isFraud', 'notFraud'], autopct='%1.1f%%')
plt.show()


# In[ ]:


train['is_proton_mail'] = (train['P_emaildomain'] == 'protonmail.com') | (train['R_emaildomain']  == 'protonmail.com')
test['is_proton_mail'] = (test['P_emaildomain'] == 'protonmail.com') | (test['R_emaildomain']  == 'protonmail.com')


# Changing label of emails with their address from P_emaildomain and R_emaildomain columns

# In[ ]:


emails = {'gmail': 'google', 'att.net': 'att', 'twc.com': 'spectrum', 'scranton.edu': 'other', 'optonline.net': 'other',
          'hotmail.co.uk': 'microsoft', 'comcast.net': 'other', 'yahoo.com.mx': 'yahoo', 'yahoo.fr': 'yahoo',
          'yahoo.es': 'yahoo', 'charter.net': 'spectrum', 'live.com': 'microsoft', 'aim.com': 'aol', 'hotmail.de': 'microsoft',
          'centurylink.net': 'centurylink', 'gmail.com': 'google', 'me.com': 'apple', 'earthlink.net': 'other', 
          'gmx.de': 'other', 'web.de': 'other', 'cfl.rr.com': 'other', 'hotmail.com': 'microsoft', 'protonmail.com': 'other',
          'hotmail.fr': 'microsoft', 'windstream.net': 'other', 'outlook.es': 'microsoft', 'yahoo.co.jp': 'yahoo',
          'yahoo.de': 'yahoo', 'servicios-ta.com': 'other', 'netzero.net': 'other', 'suddenlink.net': 'other',
          'roadrunner.com': 'other', 'sc.rr.com': 'other', 'live.fr': 'microsoft', 'verizon.net': 'yahoo',
          'msn.com': 'microsoft', 'q.com': 'centurylink', 'prodigy.net.mx': 'att', 'frontier.com': 'yahoo',
          'anonymous.com': 'other', 'rocketmail.com': 'yahoo', 'sbcglobal.net': 'att', 'frontiernet.net': 'yahoo',
          'ymail.com': 'yahoo', 'outlook.com': 'microsoft', 'mail.com': 'other', 'bellsouth.net': 'other',
          'embarqmail.com': 'centurylink', 'cableone.net': 'other', 'hotmail.es': 'microsoft', 'mac.com': 'apple',
          'yahoo.co.uk': 'yahoo', 'netzero.com': 'other', 'yahoo.com': 'yahoo', 'live.com.mx': 'microsoft', 'ptd.net': 'other',
          'cox.net': 'other', 'aol.com': 'aol', 'juno.com': 'other', 'icloud.com': 'apple'}
us_emails = ['gmail', 'net', 'edu']
for c in ['P_emaildomain', 'R_emaildomain']:
    train[c + '_bin'] = train[c].map(emails)
    test[c + '_bin'] = test[c].map(emails)
    
    train[c + '_suffix'] = train[c].map(lambda x: str(x).split('.')[-1])
    test[c + '_suffix'] = test[c].map(lambda x: str(x).split('.')[-1])
    
    train[c + '_suffix'] = train[c + '_suffix'].map(lambda x: x if str(x) not in us_emails else 'us')
    test[c + '_suffix'] = test[c + '_suffix'].map(lambda x: x if str(x) not in us_emails else 'us')

print('done')


# ### Change Browser Label
# 
# We can check if browser is the lastest version of browser or not

# In[ ]:


a = np.zeros(train.shape[0])
train["lastest_browser"] = a
a = np.zeros(test.shape[0])
test["lastest_browser"] = a
def setbrowser(df):
    df.loc[df["id_31"]=="samsung browser 7.0",'lastest_browser']=1
    df.loc[df["id_31"]=="opera 53.0",'lastest_browser']=1
    df.loc[df["id_31"]=="mobile safari 10.0",'lastest_browser']=1
    df.loc[df["id_31"]=="google search application 49.0",'lastest_browser']=1
    df.loc[df["id_31"]=="firefox 60.0",'lastest_browser']=1
    df.loc[df["id_31"]=="edge 17.0",'lastest_browser']=1
    df.loc[df["id_31"]=="chrome 69.0",'lastest_browser']=1
    df.loc[df["id_31"]=="chrome 67.0 for android",'lastest_browser']=1
    df.loc[df["id_31"]=="chrome 63.0 for android",'lastest_browser']=1
    df.loc[df["id_31"]=="chrome 63.0 for ios",'lastest_browser']=1
    df.loc[df["id_31"]=="chrome 64.0",'lastest_browser']=1
    df.loc[df["id_31"]=="chrome 64.0 for android",'lastest_browser']=1
    df.loc[df["id_31"]=="chrome 64.0 for ios",'lastest_browser']=1
    df.loc[df["id_31"]=="chrome 65.0",'lastest_browser']=1
    df.loc[df["id_31"]=="chrome 65.0 for android",'lastest_browser']=1
    df.loc[df["id_31"]=="chrome 65.0 for ios",'lastest_browser']=1
    df.loc[df["id_31"]=="chrome 66.0",'lastest_browser']=1
    df.loc[df["id_31"]=="chrome 66.0 for android",'lastest_browser']=1
    df.loc[df["id_31"]=="chrome 66.0 for ios",'lastest_browser']=1
    return df
train=setbrowser(train)
test=setbrowser(test)


# We can see 10.7% of transactions which are done by users with lastest browser are fraud

# In[ ]:


plt.figure(figsize=(6, 6))
plt.pie([np.sum(train[(train['lastest_browser'] == True)].isFraud.values),
                                 len(train[(train['lastest_browser'] == True)].isFraud.values) - 
                                 np.sum(train[(train['lastest_browser'] == True)].isFraud.values)],
        labels=['isFraud', 'notFraud'], autopct='%1.1f%%', colors=['y', 'g'])
plt.show()


# ## Feature Engineering
# 
# First we are going to drop columns with more than 80 precent of null values in both train and test set

# In[ ]:


train_missing_values = [str(x) for x in train_missing_values[train_missing_values > 0.80].keys()]
test_missing_values = [str(x) for x in test_missing_values[test_missing_values > 0.80].keys()]

dropped_columns = train_missing_values + test_missing_values


# Then we should drop columns that have more than 90 precent of a same value.
# 
# We must notice that the label column (isFraud) also has more than 90% of the same value, therefore we have to remove it from dropped columns.

# In[ ]:


dropped_columns = dropped_columns + [col for col in train.columns if train[col].value_counts(dropna=False, normalize=True).values[0] > 0.9]
dropped_columns = dropped_columns + [col for col in test.columns if test[col].value_counts(dropna=False, normalize=True).values[0] > 0.9]
dropped_columns.remove('isFraud')

train.drop(dropped_columns, axis=1, inplace=True)
test.drop(dropped_columns, axis=1, inplace=True)

len(dropped_columns)


# Adding some feature to make prediction better

# In[ ]:


train['TransactionAmt_to_mean_card1'] = train['TransactionAmt'] / train.groupby(['card1'])['TransactionAmt'].transform('mean')
train['TransactionAmt_to_mean_card4'] = train['TransactionAmt'] / train.groupby(['card4'])['TransactionAmt'].transform('mean')
train['TransactionAmt_to_std_card1'] = train['TransactionAmt'] / train.groupby(['card1'])['TransactionAmt'].transform('std')
train['TransactionAmt_to_std_card4'] = train['TransactionAmt'] / train.groupby(['card4'])['TransactionAmt'].transform('std')

test['TransactionAmt_to_mean_card1'] = test['TransactionAmt'] / test.groupby(['card1'])['TransactionAmt'].transform('mean')
test['TransactionAmt_to_mean_card4'] = test['TransactionAmt'] / test.groupby(['card4'])['TransactionAmt'].transform('mean')
test['TransactionAmt_to_std_card1'] = test['TransactionAmt'] / test.groupby(['card1'])['TransactionAmt'].transform('std')
test['TransactionAmt_to_std_card4'] = test['TransactionAmt'] / test.groupby(['card4'])['TransactionAmt'].transform('std')

train['id_02_to_mean_card1'] = train['id_02'] / train.groupby(['card1'])['id_02'].transform('mean')
train['id_02_to_mean_card4'] = train['id_02'] / train.groupby(['card4'])['id_02'].transform('mean')
train['id_02_to_std_card1'] = train['id_02'] / train.groupby(['card1'])['id_02'].transform('std')
train['id_02_to_std_card4'] = train['id_02'] / train.groupby(['card4'])['id_02'].transform('std')

test['id_02_to_mean_card1'] = test['id_02'] / test.groupby(['card1'])['id_02'].transform('mean')
test['id_02_to_mean_card4'] = test['id_02'] / test.groupby(['card4'])['id_02'].transform('mean')
test['id_02_to_std_card1'] = test['id_02'] / test.groupby(['card1'])['id_02'].transform('std')
test['id_02_to_std_card4'] = test['id_02'] / test.groupby(['card4'])['id_02'].transform('std')

train['D15_to_mean_card1'] = train['D15'] / train.groupby(['card1'])['D15'].transform('mean')
train['D15_to_mean_card4'] = train['D15'] / train.groupby(['card4'])['D15'].transform('mean')
train['D15_to_std_card1'] = train['D15'] / train.groupby(['card1'])['D15'].transform('std')
train['D15_to_std_card4'] = train['D15'] / train.groupby(['card4'])['D15'].transform('std')

test['D15_to_mean_card1'] = test['D15'] / test.groupby(['card1'])['D15'].transform('mean')
test['D15_to_mean_card4'] = test['D15'] / test.groupby(['card4'])['D15'].transform('mean')
test['D15_to_std_card1'] = test['D15'] / test.groupby(['card1'])['D15'].transform('std')
test['D15_to_std_card4'] = test['D15'] / test.groupby(['card4'])['D15'].transform('std')

train['D15_to_mean_addr1'] = train['D15'] / train.groupby(['addr1'])['D15'].transform('mean')
train['D15_to_mean_card4'] = train['D15'] / train.groupby(['card4'])['D15'].transform('mean')
train['D15_to_std_addr1'] = train['D15'] / train.groupby(['addr1'])['D15'].transform('std')
train['D15_to_std_card4'] = train['D15'] / train.groupby(['card4'])['D15'].transform('std')

test['D15_to_mean_addr1'] = test['D15'] / test.groupby(['addr1'])['D15'].transform('mean')
test['D15_to_mean_card4'] = test['D15'] / test.groupby(['card4'])['D15'].transform('mean')
test['D15_to_std_addr1'] = test['D15'] / test.groupby(['addr1'])['D15'].transform('std')
test['D15_to_std_card4'] = test['D15'] / test.groupby(['card4'])['D15'].transform('std')


# In[ ]:


train['uid'] = train['card1'].astype(str)+'_'+train['card2'].astype(str)
test['uid'] = test['card1'].astype(str)+'_'+test['card2'].astype(str)

train['uid2'] = train['uid'].astype(str)+'_'+train['card3'].astype(str)+'_'+train['card5'].astype(str)
test['uid2'] = test['uid'].astype(str)+'_'+test['card3'].astype(str)+'_'+test['card5'].astype(str)

train['uid3'] = train['uid2'].astype(str)+'_'+train['addr1'].astype(str)+'_'+train['addr2'].astype(str)
test['uid3'] = test['uid2'].astype(str)+'_'+test['addr1'].astype(str)+'_'+test['addr2'].astype(str)

train['TransactionAmt_check'] = np.where(train['TransactionAmt'].isin(test['TransactionAmt']), 1, 0)
test['TransactionAmt_check']  = np.where(test['TransactionAmt'].isin(train['TransactionAmt']), 1, 0)

train['TransactionAmt'] = np.log1p(train['TransactionAmt'])
test['TransactionAmt'] = np.log1p(test['TransactionAmt'])    


# In[ ]:


for feature in ['id_36']:
    train[feature + '_count_full'] = train[feature].map(pd.concat([train[feature], test[feature]], ignore_index=True).value_counts(dropna=False))
    test[feature + '_count_full'] = test[feature].map(pd.concat([train[feature], test[feature]], ignore_index=True).value_counts(dropna=False))
        
for feature in ['id_01', 'id_31', 'id_35', 'id_36']:
    train[feature + '_count_dist'] = train[feature].map(train[feature].value_counts(dropna=False))
    test[feature + '_count_dist'] = test[feature].map(test[feature].value_counts(dropna=False))


# In[ ]:


for col in ['card1']: 
    valid_card = pd.concat([train[[col]], test[[col]]])
    valid_card = valid_card[col].value_counts()
    valid_card = valid_card[valid_card>2]
    valid_card = list(valid_card.index)

    train[col] = np.where(train[col].isin(test[col]), train[col], np.nan)
    test[col]  = np.where(test[col].isin(train[col]), test[col], np.nan)

    train[col] = np.where(train[col].isin(valid_card), train[col], np.nan)
    test[col]  = np.where(test[col].isin(valid_card), test[col], np.nan)


# ## Prepare Data for Modeling
# 
# ### Fill Nan Values
# 
# First we should fill null values for both categorical and numerical values

# In[ ]:


numerical_columns = list(test.select_dtypes(exclude=['object']).columns)

train[numerical_columns] = train[numerical_columns].fillna(train[numerical_columns].median())
test[numerical_columns] = test[numerical_columns].fillna(train[numerical_columns].median())
print("filling numerical columns null values done")


# Now, we find out categorical columns

# In[ ]:


categorical_columns = list(filter(lambda x: x not in numerical_columns, list(test.columns)))
categorical_columns[:5]


# then, fill missing values in categorical columns

# In[ ]:


train[categorical_columns] = train[categorical_columns].fillna(train[categorical_columns].mode())
test[categorical_columns] = test[categorical_columns].fillna(train[categorical_columns].mode())
print("filling numerical columns null values done")


# ### Label Encoding

# Encode categorical columns

# In[ ]:


from sklearn.preprocessing import LabelEncoder

for col in categorical_columns:
    le = LabelEncoder()
    le.fit(list(train[col].astype(str).values) + list(test[col].astype(str).values))
    train[col] = le.transform(list(train[col].astype(str).values))
    test[col] = le.transform(list(test[col].astype(str).values))


# ## Create Test Set and Train Set

# Because of using cross validation, we use this step later when we want to find best model.
# 
# Now, we have to remove isFraud column from daraframe

# In[ ]:


labels = train["isFraud"]
train.drop(["isFraud"], axis=1, inplace=True)


# In[ ]:


X_train, y_train = train, labels
del train, labels
gc.collect()


# ## Finding Best Model
# 
# use sample_submission file for final submission

# In[ ]:


lgb_submission=sample_submission.copy()
lgb_submission['isFraud'] = 0


# We use cross validation with 5 folds

# In[ ]:


n_fold = 5
folds = KFold(n_fold)


# Using LGBM for finding best model

# In[ ]:


for fold_n, (train_index, valid_index) in enumerate(folds.split(X_train)):
    print(fold_n)
    
    X_train_, X_valid = X_train.iloc[train_index], X_train.iloc[valid_index]
    y_train_, y_valid = y_train.iloc[train_index], y_train.iloc[valid_index]
    dtrain = lgb.Dataset(X_train, label=y_train)
    dvalid = lgb.Dataset(X_valid, label=y_valid)
    
    lgbclf = lgb.LGBMClassifier(
            num_leaves= 512,
            n_estimators=512,
            max_depth=9,
            learning_rate=0.064,
            subsample=0.85,
            colsample_bytree=0.85,
            boosting_type= "gbdt",
            reg_alpha=0.3,
            reg_lamdba=0.243
    )
    
    X_train_, X_valid = X_train.iloc[train_index], X_train.iloc[valid_index]
    y_train_, y_valid = y_train.iloc[train_index], y_train.iloc[valid_index]
    lgbclf.fit(X_train_,y_train_)
    
    del X_train_,y_train_
    print('finish train')
    pred=lgbclf.predict_proba(test)[:,1]
    val=lgbclf.predict_proba(X_valid)[:,1]
    print('finish pred')
    del lgbclf, X_valid
    print('ROC accuracy: {}'.format(roc_auc_score(y_valid, val)))
    del val,y_valid
    lgb_submission['isFraud'] = lgb_submission['isFraud']+ pred/n_fold
    del pred
    gc.collect()


# Create submission file with name `prediction.csv`

# In[ ]:


lgb_submission.insert(0, "TransactionID", np.arange(3663549, 3663549 + 506691))
lgb_submission.to_csv('prediction.csv', index=False)

