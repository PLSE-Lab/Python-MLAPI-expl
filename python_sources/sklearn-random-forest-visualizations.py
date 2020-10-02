#!/usr/bin/env python
# coding: utf-8

# # Random Forest Regression Fraud Classification
# 
# In this entry for the IEEE CIS Fraud Detection competition on Kaggle (https://www.kaggle.com/c/ieee-fraud-detection), where a set of credit card transactions must be analyzed to predict a likelihood of fraud, I will be using Scikit-learn to build a Random Forest Regression model for predicting fraud from credit card transaction data.
# 
# While I didn't finish in time to build my model for submission, I still learned a lot about properly visualizing and manipulating data to fit a regression problem.

# In[ ]:


# Data processing libraries
import numpy as np
import pandas as pd

# Plotting libraries for data exploration
import matplotlib.pyplot as plt
import seaborn as sns

# Misc libraries
import math

# List all files in working directory:
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# ### Dataset Retrieval
# 
# We now load in each CSV file for the sets of Transaction and Identification tables for the train and test sets. As a first step, we will put them in ascending order according to the TransactionID and display the dataframes' formats.

# In[ ]:


# Allows us to see every column in the [dataset].head() calls, keep this commented
# out unless you want to see the name of every variable (takes a while to run)
#pd.set_option('display.max_columns', None)

train_trans = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_transaction.csv')
train_trans = train_trans.set_index('TransactionID').sort_index()
print("Training transaction set has %d rows and %d columns" % train_trans.shape)
train_trans.head()


# In[ ]:


train_id = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_identity.csv')
train_id = train_id.set_index('TransactionID').sort_index()
print("Training identification set has %d rows and %d columns" % train_id.shape)
train_id.head()


# In[ ]:


test_trans = pd.read_csv('/kaggle/input/ieee-fraud-detection/test_transaction.csv')
test_trans = test_trans.set_index('TransactionID').sort_index()
print("Test transaction set has %d rows and %d columns" % test_trans.shape)
test_trans.head()


# In[ ]:


test_id = pd.read_csv('/kaggle/input/ieee-fraud-detection/test_identity.csv')
test_id = test_id.set_index('TransactionID').sort_index()
print("Test identification set has %d rows and %d columns" % test_id.shape)
test_id.head()


# In[ ]:


### Preparing submission file
submission = pd.read_csv('/kaggle/input/ieee-fraud-detection/sample_submission.csv')
print("Submission file has %d rows and %d columns" % submission.shape)
submission.head()


# ### Significance of the Features
# 
# As per the discussion post (https://www.kaggle.com/c/ieee-fraud-detection/discussion/101203#latest-624125), the data fields in each dataset have the following summaries as to their significance:
# #### Transaction Tables
# * TransactionDT - timedelta from an unknown reference date (ex. 3 days after an unknown origin)
# * TransactionAMT - payment amount in USD
# * ProductCD - code of the product in the transaction
# * card(1-6) - card info including type, bank, country, etc.
# * addr(1,2) - address
# * dist(1,2) - distance
# * P_emaildomain, R_emaildomain - Purchaser and Recipient email domains
# * C(1-14) - counts of various factors, their meaning is intentionally obfuscated
# * D(1-15) - timedeltas between certain events, such as previous purchases
# * M(1-9) - match, such as name on card and address, etc
# * V(1-339) - engineered features added by Vesta
# 
# #### Identification Tables
# * DeviceType - mobile or desktop
# * DeviceInfo - category of computing device used
# * id_(01-38) - a series of variables with their meanings masked, however it is clear that id_33 is device resolution, id_31 is browser used, and id_30 is the OS used. Other id fields include information on IP, ISP, and other networking information
# 
# The transactions in each table are linked via a matching TransactionID, and are rated for being fraud or not in the transaction table's isFraud field (0 for False, 1 for True)
# 
# The modifications I make to the stock dataset include:
# * Adding the sum of N/A values as a feature for each transaction
# * Splitting id_33 into numerical horizontal and vertical resolution integers
# * Extracting the email service provider as an additional feature from the email domain features

# ### Data Linking
# 
# Now, the transaction and identity tables will be linked together into a singular pandas Dataframe and modified as necessary. I will also be extensively exploring the distributions of each data field for insights to guide my approach in tuning models.
# 
# It is clear from the start that only about 1 in 4 of all the transactions in both the train and test sets have identification data to match them. If a matching ID isn't found in the identification table, we'll simply default all the extra fields for that transaction as 'NaN' and create a new variable 'isIdentified' in the Transaction table that equates to 1 (True) if a matching record was found for that transaction, 0 (False) otherwise.

# In[ ]:


def checkMatches(transaction, identification):
    isIdentified = []
    trans_ids = transaction.index.array
    id_ids = identification.index.array
    # Since we guarantee order of the transaction ids, we don't need to check the
    # entirety of id_ids for each id and can just check the current index instead
    length = len(id_ids)
    i = 0
    for id in trans_ids:
        if i >= length or id != id_ids[i]:
            isIdentified.append(0) # False - no match found
        else:
            i += 1
            isIdentified.append(1) # True - match found
    transaction["isIdentified"] = isIdentified
    return transaction


# In[ ]:


checkMatches(train_trans,train_id)
train_trans.head()


# In[ ]:


checkMatches(test_trans,test_id)
test_trans.head()


# Now to join the transaction and identification datasets:

# In[ ]:


train_combined = train_trans.join(train_id)
print("Combined training set has %d rows and %d columns" % train_combined.shape)
train_combined.head()


# In[ ]:


test_combined = test_trans.join(test_id)
print("Combined test set has %d rows and %d columns" % test_combined.shape)
test_combined.head()


# ## Data Exploration
# 
# Here, we extensively explore distributions of the features and their relations to each other, in order to inform the model-building process. To consolidate the code, helper functions are defined here for each type of data processed, numerical or categorical.

# In[ ]:


numGraphs = 3 # If multiple variables are plotted, this is how many graphs to plot per row
subWidth = 12 # Width to allocate for each row of subplots
subHeight = 6 # Height to allocate for each subplot
font = 8 # Font size for subplots

def plotNum(setType,dataset,fields):
    # Plots KDE plots of numerical data, to show distribution of frequencies
    # setType is a string, either "train" or "test"
    sns.set(style='darkgrid')
    n = len(fields) # Number of variables to plot
    if n == 1:
        field = fields[0]
        data = dataset[field].dropna()
        plt.xticks(rotation=90)
        sns.kdeplot(data).set_title("%s set %s" % (setType, field))
        #print("Minimum %s value: %d" % (field, data.min()))
        #print("Maximum %s value: %d" % (field, data.max()))
        #print("Average %s value: %d" % (field, (data.sum()/len(data))))
        #print("Median %s value: %d" % (field, data.median()))
    else:
        size = (subWidth, subHeight * math.ceil(n/numGraphs)) # Allot 4 in of height per row
        if n > numGraphs: 
            fig, axes = plt.subplots(math.ceil(n/numGraphs), numGraphs, figsize=size)
        else:
            fig, axes = plt.subplots(1, n, figsize=size)
        for i in range(n):
            field = fields[i]
            data = dataset[field].dropna()
            if n > numGraphs:
                sns.kdeplot(data,ax=axes[i//numGraphs,i % numGraphs]).set_title("%s set %s" % (setType, field))
            else:
                sns.kdeplot(data,ax=axes[i]).set_title("%s set %s" % (setType, field))
            #print("Minimum %s value: %d" % (field, data.min()))
            #print("Maximum %s value: %d" % (field, data.max()))
            #print("Average %s value: %d" % (field, (data.sum()/len(data))))
            #print("Median %s value: %d" % (field, data.median()))
        for ax in axes.flatten():
            for tick in ax.get_xticklabels():
                tick.set_rotation(90)
            ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.show()
    return None

def plotCat(setType,dataset,fields):
    maxCat = 10 # Number of categories to reduce plots to showing
    sns.set(style='darkgrid')
    n = len(fields) # Number of variables to plot
    if n == 1:
        field = fields[0]
        data = dataset[field].dropna()
        plt.xticks(rotation=90)
        sns.countplot(data, order = data.value_counts().iloc[:maxCat].index).set_title("%s set %s" % (setType, field))
    else:
        size = (subWidth, subHeight * math.ceil(n/numGraphs))
        if n > numGraphs:
            fig, axes = plt.subplots(math.ceil(n/numGraphs), numGraphs, figsize=size)
        else:
            fig, axes = plt.subplots(1, n, figsize=size)
        for i in range(n):
            field = fields[i]
            data = dataset[field].dropna()
            if n > numGraphs:
                sns.countplot(data, order = data.value_counts().iloc[:maxCat].index,
                    ax=axes[i//numGraphs,i % numGraphs]).set_title("%s set %s" % (setType, field))
            else:
                sns.countplot(data, order = data.value_counts().iloc[:maxCat].index,
                    ax=axes[i]).set_title("%s set %s" % (setType, field))
        for ax in axes.flatten():
            for tick in ax.get_xticklabels():
                tick.set_rotation(90)
    plt.show()
    return None


# Also, it may be of significance to the model to take into account the presence of missing values in certain fields. In the code snippets below, I will be analyzing the frequency of null values on a per-column basis, afterwards replacing all null values with 0:

# In[ ]:


nulls = train_combined.isnull().sum()
#nulls = nulls.sort_values()
nulls.plot(kind='barh', figsize=(10,60), fontsize=10, 
           title = "Number of Training Set Null Values")


# In[ ]:


nulls = test_combined.isnull().sum()
print(nulls)
#nulls = nulls.sort_values()
nulls.plot(kind='barh', figsize=(10,60), fontsize=10, 
           title = "Number of Test Set Null Values")


# Some peculiarities to observe from the distribution of null values:
# * It seems that the first few features through card1 are always defined, but the other card# values may rarely be missing values.
# * On the training set, V279-V321 are virtually always defined, but a few of these variables are more frequently undefined on the test set (thus, it would be dangerous to assume engineered features that are never missing on the test set are never going to be missing on real data).
# * There are several clear "groups" of features with similar probabilities of being missing values.

# ### TransactionAmt

# In[ ]:


plotNum('train',train_combined,['TransactionAmt'])


# In[ ]:


plotNum('test',test_combined,['TransactionAmt'])


# The datasets seem to be abnormally weighted in favor of small transactions, with some bias as well in favor of the maximum transaction amounts. While the maximum transaction amount is far higher in the training set, the mean and median are virtually identical in both the train and test sets, and so the training set likely is indicative of the test set.

# ### Product CD

# In[ ]:


plotCat('train',train_combined,['ProductCD'])


# In[ ]:


plotCat('test',test_combined,['ProductCD'])


# Each dataset has similar distributions of ProductCD, heavily imbalanced in favor of 'W'.

# ### Card 1-6
# 
# Card fields 1-3 and 5 are numerical data, whereas card4 and card6 are categorical.

# In[ ]:


catFields = []
numFields = []
for i in [4,6]:
    catFields.append('card%d' % i)
for i in [1,2,3,5]:
    numFields.append('card%d' % i)


# In[ ]:


plotCat('train',train_combined,catFields)


# In[ ]:


plotCat('test',test_combined,catFields)


# In[ ]:


plotNum('train',train_combined,numFields)


# In[ ]:


plotNum('test',test_combined,numFields)


# ### Address 1 and 2
# 
# Both fields are numerical, addr2 is often undefined (Likely due to it only being used if an address is too specific to only be defined by one number)

# In[ ]:


catFields = ['addr1','addr2']


# In[ ]:


plotNum('train',train_combined,catFields)


# In[ ]:


plotNum('test',test_combined,catFields)


# ### Distance 1 and 2
# 
# Both fields are numerical

# In[ ]:


numFields = ['dist1','dist2']


# In[ ]:


plotNum('train',train_combined,numFields)


# In[ ]:


plotNum('test',test_combined,numFields)


# ### Email Addresses (and Providers)
# 
# 'P_emaildomain' is a categorical purchaser email address, while 'R_emaildomain' is a categorical recipient email address. The 'P_emailprovider' and 'R_emailprovider' are my own features of the email service extracted from the raw address (everything after the @ in the strings).

# In[ ]:


# Extracting email services from a list of addresses:
def emailServices(emails):
    services = []
    top = 10 # How many of the most frequent providers to display before lumping into "other"
    for email in emails:
        if type(email) != str:
            service = float('nan')
        else:
            splitIndex = email.find('@')
            service = email[splitIndex + 1:]
        services.append(service)
    return services


# In[ ]:


# Adding each set of servicers to the datasets:
train_combined['P_emailprovider'] = emailServices(train_combined['P_emaildomain'])
train_combined['R_emailprovider'] = emailServices(train_combined['R_emaildomain'])
test_combined['P_emailprovider'] = emailServices(test_combined['P_emaildomain'])
test_combined['R_emailprovider'] = emailServices(test_combined['R_emaildomain'])


# To get a better idea of what formats email addresses are often used, I'll instead put countplots of only the 10 most frequent providers here:

# In[ ]:


catFields = ['P_emailprovider','R_emailprovider']


# In[ ]:


plotCat('train',train_combined,catFields)


# In[ ]:


plotCat('test',test_combined,catFields)


# ### Count 1-14 Variables
# 
# These are all numeric data with obfuscated meaning

# In[ ]:


numFields = []
for i in range(1,15):
    numFields.append('C%d' % i)


# In[ ]:


plotNum('train',train_combined,numFields)


# In[ ]:


plotNum('test',test_combined,numFields)


# ### Timedelta (D1-15) Variables
# These are all numeric variables that measure some time delays between unknown events of importance (ex. possibly measuring time between purchases)

# In[ ]:


numFields = []
for i in range(1,16):
    numFields.append('D%d' % i)


# In[ ]:


plotNum('train',train_combined,numFields)


# In[ ]:


plotNum('test',test_combined,numFields)


# ### Match 1-9
# These are categorical variables indicating a match (or lack thereof) being found for some unknown conditions

# In[ ]:


catFields = []
for i in range(1,10):
    catFields.append('M%d' % i)


# In[ ]:


plotCat('train',train_combined,catFields)


# In[ ]:


plotCat('test',test_combined,catFields)


# ### Device Features
# 
# DeviceType and DeviceInfo are always strings if defined, and so should be treated as categorical information. DeviceInfo can be a very wide array of values, so I will display only the top 10.

# In[ ]:


catFields = ['DeviceType','DeviceInfo']


# In[ ]:


plotCat('train',train_combined,catFields)


# In[ ]:


plotCat('test',test_combined,catFields)


# ### ID 01-38
# 
# These are obfuscated values that are of different types, we should first examine what variables are categorical or numeric. Also, note that id numbers 1-9 are listed as 01-09 in the datasets.

# In[ ]:


ids = []
numFields = []
catFields = []
for i in range(1,10):
    ids.append('id_0%d' % i)
for i in range(10,39):
    ids.append('id_%d' % i)
for id in ids:
    idType = train_combined[id].dtype
    if idType == 'float64':
        numFields.append(id)
    else:
        catFields.append(id)
    print(id, train_combined[id].dtype)


# Now we will display the numeric fields for each set, followed by the categorical fields.

# In[ ]:


plotNum('train',train_combined,numFields)


# In[ ]:


plotNum('test',test_combined,numFields)


# In[ ]:


plotCat('train',train_combined,catFields)


# In[ ]:


plotCat('test',test_combined,catFields)


# As stated in the intro, it would likely be more useful to split the resolution field (id_33) into numeric fields of width and height (I will also include a measure of screen area by multiplying the two).

# In[ ]:


def resolutionParams(res):
    # Retrieves width, height, and area from a list of strings
    widths = []
    heights = []
    areas = []
    for r in res:
        if type(r) != str:
            # Catch missing values
            width = float('nan')
            height = float('nan')
            area = float('nan')
        else:
            w, h = r.split('x')
            width = int(w)
            height = int(h)
            area = width * height
        widths.append(width)
        heights.append(height)
        areas.append(area)
    return widths, heights, areas


# In[ ]:


widths, heights, areas = resolutionParams(train_combined["id_33"])
train_combined["width"] = widths
train_combined["height"] = heights
train_combined["area"] = areas

widths, heights, areas = resolutionParams(test_combined["id_33"])
test_combined["width"] = widths
test_combined["height"] = heights
test_combined["area"] = areas


# In[ ]:


numFields = ["width", "height", "area"]


# In[ ]:


plotNum('train',train_combined,numFields)


# In[ ]:


plotNum('test',test_combined,numFields)


# ### Model Training
# 
# Now for generating a final model, we will import the appropriate Scikit-learn libraries and create train test splits for validation, before generating a final prediction file.

# In[ ]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


# First, we will need to label encode the categorical columns to be understood by the regression algorithm. Credit goes to Yoong Kang Lim for the sections in label encoding the categorical columns: https://www.kaggle.com/yoongkang/beginner-s-random-forest-example

# In[ ]:


catColumns = []
for col in train_combined.columns:
    if train_combined[col].dtype == 'object':
        catColumns.append(col)

cats = {}
for col in catColumns:
    train_combined[col].fillna('missing',inplace=True)
    test_combined[col].fillna('missing',inplace=True)
    
    train_combined[col] = train_combined[col].astype('category')
    train_combined[col].cat.add_categories('unknown',inplace=True)
    cats[col] = train_combined[col].cat.categories


# In[ ]:


for k, v in cats.items():
    test_combined[k][~test_combined[k].isin(v)] = 'unknown'


# In[ ]:


from pandas.api.types import CategoricalDtype

for k, v in cats.items():
    new_dtype = CategoricalDtype(categories=v, ordered=True)
    test_combined[k] = test_combined[k].astype(new_dtype)


# In[ ]:


for col in catColumns:
    train_combined[col] = train_combined[col].cat.codes
    test_combined[col] = test_combined[col].cat.codes


# In[ ]:


train_combined.fillna(-999,inplace=True)
test_combined.fillna(-999,inplace=True)


# In[ ]:


y = train_combined.pop("isFraud") # Separates results from parameters
x_train, x_test, y_train, y_test = train_test_split(train_combined, y, train_size = .2)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[ ]:


model = RandomForestRegressor(
    n_estimators=400, max_features=0.3,
    min_samples_leaf=20, n_jobs=-1, verbose=1)
model.fit(x_train, y_train)


# In[ ]:


pred_test = model.predict(x_test)


# In[ ]:


roc_auc_score(y_test, pred_test)


# ### Final Submission

# In[ ]:


model = RandomForestRegressor(
    n_estimators=400, max_features=0.3,
    min_samples_leaf=20, n_jobs=-1, verbose=1)
model.fit(train_combined,y)


# In[ ]:


pred = model.predict(test_combined)


# In[ ]:


submission["isFraud"] = pred
submission.to_csv('submission.csv', index=False)

