#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import sys
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno
get_ipython().run_line_magic('matplotlib', 'inline')

# disable the warning about settingwithcopy warning:
pd.set_option('chained_assignment',None)


# In[ ]:


working_directory_path = "/kaggle/input/ieee-fraud-detection/"
os.chdir(working_directory_path)


# In[ ]:


def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


# In[ ]:


train_identity = pd.read_csv("train_identity.csv")
train_transaction = pd.read_csv("train_transaction.csv")

test_identity = pd.read_csv("test_identity.csv")
test_transaction = pd.read_csv("test_transaction.csv")

train_identity = reduce_mem_usage(train_identity)
train_transaction = reduce_mem_usage(train_transaction)
test_identity = reduce_mem_usage(test_identity)
test_transaction = reduce_mem_usage(test_transaction)


# ## Objective:
# 
# **Predict fraud based on transaction information**
# 

# In[ ]:


# Since the number of columns are too large, we can expand it using pd.set_option()
pd.set_option('display.max_columns', None)  


# ## Joining the transaction data:
# 
# ### Transaction Table:
# 
# Transaction table contain TransactionID, which we can use to join with the identity table.

# In[ ]:


train_transaction['TransactionID'].value_counts().sort_values(ascending = False)


# The transaction ID for transaction table is also unique for each observation. Therefore, we have 1 to 1 join from identity table to transaction table:

# In[ ]:


train_full = pd.merge(train_identity, train_transaction, on = 'TransactionID')


# However, it comes to my attention that the number of rows are different for each table, despite having unique TransactionID:

# In[ ]:


print('Number of row in transaction:', len(train_transaction))
print('Number of row in identity:', len(train_identity))

# remove train_transaction from memory
# del train_transaction


# This suggests that there are transactions that don't have identity. A quick research on the [discussion thread](https://www.kaggle.com/c/ieee-fraud-detection/discussion/101203#latest-605862) reveals that Vesta was unable to collect all identity information due to technical difficulty. Therefore, we will need to face two options:
# 
# 1. Using identity + transaction to make predictions. This option results in fewer observations but more complete (more features).
# 
# 2. Using only transaction
# 
# 3. Using transaction but add identity when avaiable
# 
# For now, we only explore the identity + transaction joined table to do EDA and build model. 

# # Data Quality Inspection
# 
# There are couple common issues that we need to watch out for:
# 
# 1. Attributes Formatting (data types)
# 
# 2. Missing Data
# 
# 3. Replacement or Drop
# 
# 4. Response Variable
# 
# First, let's transform our data into the types that we expected:
# 
# ## 1. Attributes formatting (data types)

# In[ ]:


train_full.info(verbose=True, null_counts=True)


# Most of the column names have been masked for privacy protection. Without accurate description of the fields meaning, it would be difficult to determine the type of data. Fortunately, Vesta have provided us with high-level summary of data.
# 
# Let's recall the avaiable groups of information that were provided for us:
# 
# 1. Identity Table:
# 
#     * id_01 - id_38: contains network connection information
#     
#     * DeviceType and DeviceInfo
#     
# 2. Transactional Table:
# 
#     * card1 - card6: card information
#     
#     * addr: address
#     
#     * dist: distance
# 
#     * P_ and (R__) emaildomain: purchaser and recipient email domain
#     
#     * C1-C14: counting
#     
#     * D1-D15: timedelta, such as days between previous transaction, etc.
#     
#     * M1-M9: match, such as names on card and address, etc.
#     
#     * V1-V339: Vesta engineered features
#     
#     * ProductCD: product code, the product for each transaction
#     
#     * TransactionDT: timedelta from a given reference datetime
#     
#     * TransactionAMT: transaction payment amount in USD    
#     
# ## Missing Data
#     
# Let's take a look at the missing data for the **categorical variables** first:

# In[ ]:


train_full_cat = train_full.filter(regex='id|card|ProductCD|addr|email|M|DeviceType|DeviceInfo')


# In[ ]:


plt.figure(figsize=(18,9))
sns.heatmap(train_full_cat.isnull(), cbar= False)


# **Observation**: We can see that our data has a lot of missing values. White color presents missing values.
# 
# 1. Most M columns missing almost if not all data
# 
# 2. Id_07, 08 and id_21-27 missing most data
# 
# 3. Id_01, id_12, card1, card2 contains mostly non-null. Perhaps, these columns contain unique ID information, and therefore, cannot be null. Let's double check the number of missing in these columns: 

# In[ ]:


train_full_cat[['id_01','id_12','card1','card2']].info(null_counts=True)


# Yes, they are indeed complete, except for card 2. If I were to guess, card1 could be first name and card2 could be last name.
# 
# Now let's check out missing data for **numerical variables**:

# In[ ]:


train_full_num = train_full.filter(regex='isFraud|TransactionDT|TransactionAmt|dist|C|D')
plt.figure(figsize=(18,9))
sns.heatmap(train_full_num.isnull(), cbar= False)


# **Observation**: 
#     
#     1. Basic information about transaction such as ID, DT, amount and type of product is complete 
#     
#     2. Dist1 and dist2 is very sparse.
#     
#     3. C columns are complete
#     
#     4. Most D columns are sparse except D1
#     
# Lastly, we want to check for data completeness of **Vesta's engineered features**:

# In[ ]:


train_full_Vesta = train_full.filter(regex='V')
plt.figure(figsize=(18,9))
sns.heatmap(train_full_Vesta.isnull(), cbar= False)


# **Observation**: Ahh, she looks like a work of art. The repeated missing patterns in the V columns suggest that many V columns are related and perhaps trying to describe certain characteristics of a transaction. For example, columns V322-V399 have identical missing locations.
# 
# Let's verify our intuition with correlation heatmap measures nullity correlation: how strongly the presence or absence of one variable affects the presence of another:

# In[ ]:


msno.dendrogram(train_full_Vesta)


# **Interpretation**: The dendrogram uses a hierarchical clustering algorithm to bin variables against one another by their nullity correlation. Each cluster of leaves explain how one variable might always be empty when another is empty, or filled when another variable is filled. This dendogram suggests that the position of missing/fill values are correlated. Perhaps similar columns were derived from the same feature or combinations of features.

# ## 3. Replacement or drop the missings
# The idea of imputation is both *"seductive and dangerous"* in the words of R.J.A Little. 
# 
# I truly believe that there is no best way to deal with missing, especially when having to deal with partial information. Knowing which columns could be imputed or dropped may alter the result of the final predictions by a non-trivial amount. The fact that certain value is missing could have been due to specific variation in the feature (missing not at random). This is one of the process that could have been much more useful if we were given the meaning of each columns. But when life gives you lemon, you turns it into sweet, sweet meachine learning input juice. 
# 
# ### Understand that Train and Test data were splitted by time
# 
# This is a graceful finding from https://www.kaggle.com/robikscube/ieee-fraud-detection-first-look-and-eda.
# 
# * The `TransactionDT` feature is defined as time delta from a chosen datetime. This gives us information about the relative time and the countinuity of each transaction. Ploting both test and train `TransactionDT` on the graph suggests that train and test dataset were splited by time, with a gap in between.

# In[ ]:


plt.hist(train_transaction['TransactionDT'], label='train')
plt.hist(test_transaction['TransactionDT'], label='test')
plt.legend()
plt.title('Distribution of TransactionDT')


# Some people suggest that if `TransactionDT` is measured in seconds, then the combined time period between test and train dataset could total to approximately 1 year, and the gap can account for ~ 1 month. 
# 
# Lynn@Vesta commented in one of the discussion post:
# 
# *"We define reported chargeback on card, user account, associated email address and transactions directly linked to these attributes as fraud transaction (isFraud=1); If none of above is found after 120 days, then we define as legit (isFraud=0)"*
# 

# ## 4. Response/ Target Variable

# In[ ]:


plt.figure(figsize=(12,6))
g = sns.countplot(x = 'isFraud', data = train_full)
g.set_title("Fraud Distribution", fontsize = 17)
g.set_xlabel("Is Fraud?", fontsize = 15)
g.set_ylabel("Count", fontsize = 15)
plt.legend(title='Fraud', labels=['No', 'Yes'])

for p in g.patches:
    height = p.get_height()
    g.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.2f}%'.format(height/len(train_full) * 100),
            ha="center", fontsize=15) 


# **Observation**:The fraud percentage is quite high: 7.85% for the complete observations (identity + transaction). We can see there is a class imbalance problem, where occurence of one class is significantly higher than another. This will lead to much a higher false negative - tendency of picking "not fraud". We can mitigate this issue by using two common methods:
# 
# 1. Cost function based approaches
# 
# 2. Sampling based approaches

# # Explore Categorical Features
# **Categorical Features:**
# 
# **1. Transactional Table:**
#     
#     ProductCD
# 
#     card1 - card6
# 
#     addr1, addr2
# 
#     Pemaildomain Remaildomain
# 
#     M1 - M9
#     
#     
# **2. Identity Table**
# 
#     DeviceType
# 
#     DeviceInfo
#     
#     id12 - id38
# 
# Let's take a quick look at these categorical features:

# In[ ]:


train_full_cat.head()


# ## Examine Product Code

# In[ ]:


plt.figure(figsize=(12,6))

total = len(train_full_cat)

plt.subplot(121)
g = sns.countplot(x = 'ProductCD', data = train_full_cat)
g.set_title('ProductCD Distribution', fontsize = 15)
g.set_xlabel("Product Code", fontsize=15)
g.set_ylabel("Count", fontsize=15)
for p in g.patches:
    height = p.get_height()
    g.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.2f}%'.format(height/total*100),
            ha="center", fontsize=14) 

plt.subplot(122)
g1 = sns.countplot(x='ProductCD', hue='isFraud', data=train_full)
g1.set_title('ProductCD by Fraud', fontsize = 15)
g1.set_xlabel("Product Code", fontsize=15)
g1.set_ylabel("Count", fontsize=15)
plt.legend(title='Fraud', loc='best', labels=['No', 'Yes'])


# **Observations**: C is the most frequent product category. Product C also have the highest count of fraud. We can obtain the proportion of fraud for each product category:

# In[ ]:


train_full[train_full['isFraud'] == 1]['ProductCD'].value_counts(normalize = True)


# In[ ]:


# grouped table
train_full.groupby('ProductCD')['isFraud'].value_counts(normalize = True)


# In[ ]:


# visualization of table
plt.figure(figsize=(12,12))
a = train_full.groupby('ProductCD')['isFraud'].value_counts(normalize = True).unstack().plot.bar(stacked = True)
a.set_title('Rate of Fraud by Product Category', fontsize = 15)
plt.xticks(rotation='horizontal')


# **Conclusion**: Product C takes up 67.5% of fraud cases for transactions that have identity. And also have highest rate of fraud: 12%, more than double any other class of product.
# 
# **Question**: Why product C? Is there any additional information that help us better understand product C high fraud rate?
# 
# We have 2 numerical variables that we can compare between groups of products:
# 
# TransactionDT: timedelta from a given reference datetime
# 
# TransactionAmt: transaction payment amount in USD

# In[ ]:


plt.figure(figsize=(12,10))
sns.boxplot(x = 'ProductCD', y = 'TransactionAmt', hue = 'isFraud', data = train_full)


# **Observation**: Product C are items with low dollar value.

# In[ ]:


plt.figure(figsize=(12,10))
sns.boxplot(x = 'ProductCD', y = 'TransactionDT', hue = 'isFraud', data = train_full)


# **Observation**: All products have same min and max timedelta range. 
# 
# **Conclusion**:The plot suggests little to no difference in timedelta accross all groups.
# 
# ## Examine Card 1,2,3,5
# 
# The card 1,2,3, and 5 was represented as numerical values, temping us to plot the histogram. However, we need to remember that card columns were classified as categorical variables. Meaning it's likely that these numerical variables were coded for categorical variables.

# In[ ]:


train_full_cat.describe().loc[:,'card1':'card5']


# In[ ]:


train_full_cat.loc[:,'card1':'card5'].nunique()


# Card 1 contains 8499 unique values, suggesting card 1 may have been ID of the card. Card 2,3 and 5 have less unique values, so perhaps they could be expiration date, or combinations that generate card identity? Since we don't know how these information was scrammbled, we might pickup patterns generated by encryption algorithm instead of data. No further analysis should be done unless more infomation is given.
# 
# Same goes for the addr1 and addr2.

# ## Examine Card 4 and 6
# 
# ### Card 4: Card Network

# In[ ]:


plt.figure(figsize=(12,6))

total = len(train_full_cat)

plt.subplot(121)
g = sns.countplot(x = 'card4', data = train_full_cat)
g.set_title('Card Network Distribution', fontsize = 15)
g.set_xlabel("Card Issuers", fontsize=15)
g.set_ylabel("Count", fontsize=15)
for p in g.patches:
    height = p.get_height()
    g.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.2f}%'.format(height/total*100),
            ha="center", fontsize=14) 

plt.subplot(122)
g1 = sns.countplot(x='card4', hue='isFraud', data=train_full)
g1.set_title('Card Network by Fraud', fontsize = 15)
g1.set_xlabel("Card Issuers", fontsize=15)
g1.set_ylabel("Count", fontsize=15)
plt.legend(title='Fraud', loc='best', labels=['No', 'Yes'])


# **Observation:** Visa card accounts for the highest instances of fraud, but this also because visa is the most popular card type. Again, we can only conclude after comparing the fraud propotion for each card type:

# In[ ]:


train_full[train_full['isFraud'] == 1]['card4'].value_counts(normalize = True)


# In[ ]:


# grouped table
train_full.groupby('card4')['isFraud'].value_counts(normalize = True)


# In[ ]:


# visualization of table
plt.figure(figsize=(12,12))
b = train_full.groupby('card4')['isFraud'].value_counts(normalize = True).unstack().plot.bar(stacked = True)
b.set_title('Rate of Fraud by Card Network', fontsize = 15)
plt.xticks(rotation='horizontal')


# **Conclusion**: Visa accounts for 61% of all fraud occurences. However, when normalized by total number of each type, Visa have fraud rate of only 8%, lower than Mastercard and same as Discovercard. Only American Express have significantly lower fraud rate compare to others.

# ### Card 6: Card Type
# 
# Similarly, we can use the same method of data analysis on this variable:

# In[ ]:


plt.figure(figsize=(12,6))

total = len(train_full_cat)

plt.subplot(121)
g = sns.countplot(x = 'card6', data = train_full)
g.set_title('Card Type Distribution', fontsize = 15)
g.set_xlabel("Card Type", fontsize=15)
g.set_ylabel("Count", fontsize=15)
for p in g.patches:
    height = p.get_height()
    g.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.2f}%'.format(height/total*100),
            ha="center", fontsize=14) 

plt.subplot(122)
g1 = sns.countplot(x='card6', hue='isFraud', data=train_full)
g1.set_title('Card Type by Fraud', fontsize = 15)
g1.set_xlabel("Card Type", fontsize=15)
g1.set_ylabel("Count", fontsize=15)
plt.legend(title='Fraud', loc='best', labels=['No', 'Yes'])


# **Observation:** The number of card type are fairly simiar, and so does the fraud cases. 

# In[ ]:


# grouped table
train_full.groupby('card6')['isFraud'].value_counts(normalize = True)


# In[ ]:


# visualization of table
plt.figure(figsize=(12,12))
b = train_full.groupby('card6')['isFraud'].value_counts(normalize = True).unstack().plot.bar(stacked = True)
b.set_title('Rate of Fraud by Card Type', fontsize = 15)
plt.xticks(rotation='horizontal')


# **Observation**: Not much difference in fraud rate between credit card and debit card
# 
# ## Examine Email Domain
# 
# ### 1. Purchaser Email

# In[ ]:


plt.figure(figsize=(12,6))

g = sns.countplot(x = 'P_emaildomain', data = train_full)
g.set_title('Purchaser Email Domain Distribution', fontsize = 15)
g.set_xlabel("Email Domain", fontsize=15)
g.set_ylabel("Count", fontsize=15)
plt.xticks(rotation='vertical')


# **Observation**: I see alot of domains came from the same distributors such as hotmail.com, hotmail.fr, yahoo.com, yahoo.fr, yahoo.de, etc. We can group these domains together under the parent distributors.
# 
# **Action:** Create P_parent_emaildomain field that remove the part after '.' 

# In[ ]:


train_full["P_parent_emaildomain"] = train_full["P_emaildomain"].str.split('.', expand = True)[[0]]


# In[ ]:


plt.figure(figsize=(12,6))

g = sns.countplot(x = 'P_parent_emaildomain', data = train_full)
g.set_title('Purchaser Email Domain Distribution', fontsize = 15)
g.set_xlabel("Email Domain", fontsize=15)
g.set_ylabel("Count", fontsize=15)
plt.xticks(rotation= "vertical")


# Fewer email domains result in cleaner x tickers. Let's add the fraud rate like in the previous graphs, but this time we add the rate line on top of this graph:

# In[ ]:


P_emaildomain_fraud_rate = train_full.groupby('P_parent_emaildomain')['isFraud'].value_counts(normalize = True).unstack().fillna(0)[1]

plt.figure(figsize=(12,6))

g = sns.countplot(x = 'P_parent_emaildomain', data = train_full, order = P_emaildomain_fraud_rate.index)
g.set_title('Purchaser Email Domain Distribution', fontsize = 15)
g.set_xlabel("Email Domain", fontsize=15)
g.set_ylabel("Count", fontsize=15)
plt.xticks(rotation= "vertical")

r = g.twinx()
r = sns.pointplot(x = P_emaildomain_fraud_rate.index, y = P_emaildomain_fraud_rate, color = 'blue')
r.set_ylabel("Fraud Rate", fontsize = 16, color = "blue")


# Protonmail returns an exemely high fraud rate. Almost 80% of transactions from purchaser using protonmail.com were label fraud. Let's double check this result:

# In[ ]:


protonmail_fraud = len(train_full[(train_full['P_parent_emaildomain'] == "protonmail") & (train_full['isFraud'] == 1)])
protonmail_non_fraud = len(train_full[(train_full['P_parent_emaildomain'] == "protonmail") & (train_full['isFraud'] == 0)])

protonmail_fraud_rate = protonmail_fraud/ (protonmail_fraud + protonmail_non_fraud)
print("Number of protonmail fraud transactions:", protonmail_fraud)
print("Number of protonmail non-fraud transactions:", protonmail_non_fraud)
print("Protonmail fraud rate:", protonmail_fraud_rate)


# ### 2. Recipient Email
# 
# Similarly, we can perform the similar analysis on Recepient email domains

# In[ ]:


train_full["R_parent_emaildomain"] = train_full["R_emaildomain"].str.split('.', expand = True)[[0]]
train_full["R_parent_emaildomain"].fillna("NA", inplace=True)

R_emaildomain_fraud_rate = train_full.groupby('R_parent_emaildomain')['isFraud'].value_counts(normalize = True).unstack().fillna(0)[1]

plt.figure(figsize=(12,6))

g = sns.countplot(x = 'R_parent_emaildomain', data = train_full, order = R_emaildomain_fraud_rate.index)
g.set_title('Recipient Email Domain Distribution', fontsize = 15)
g.set_xlabel("Email Domain", fontsize=15)
g.set_ylabel("Count", fontsize=15)
plt.xticks(rotation= "vertical")

r = g.twinx()
r = sns.pointplot(x = R_emaildomain_fraud_rate.index, y = R_emaildomain_fraud_rate, color = "blue")
r.set_ylabel("Fraud Rate", fontsize = 16, color = "blue")


# I enjoy this format of visualizing, so I should creat a function that help me explore the categorical format with regard to fraud rate:

# In[ ]:


def visualize_cat_cariable(variable, df=train_full):
    train_full[variable].fillna("NA", inplace=True)
    variable_fraud_rate = df.groupby(variable)['isFraud'].value_counts(normalize = True).unstack().fillna(0)[1]
    
    plt.figure(figsize=(12,6))

    g = sns.countplot(x = variable, data = df, order = variable_fraud_rate.index)
    g.set_title('{} Count'.format(variable), fontsize = 15)
    g.set_xlabel("{}".format(variable), fontsize=15)
    g.set_ylabel("Count", fontsize=15)
    plt.xticks(rotation= "vertical")

    r = g.twinx()
    r = sns.pointplot(x = variable_fraud_rate.index, y = variable_fraud_rate, color = "blue")
    r.set_ylabel("Fraud Rate", fontsize = 16, color = "blue")


# ## Examine M1-M9
# 
# The transaction data that has comple identity returns mostly NaN except for M4. Let's check it out:

# In[ ]:


train_full_cat.loc[:,'M1':'M9'].apply(pd.value_counts)


# In[ ]:


visualize_cat_cariable('M4')


# **Observartion**: Not much variation in fraud rate between M0, M1, and M2 in M4
# 
# We have gone through all categorical variables in the Transaction Table, now we check out the remaining categorical variables in the Identity Table.
# 
# ## Examine DeviceType

# In[ ]:


visualize_cat_cariable('DeviceType')


# **Observation**: Fraud rate is higher for mobile device compared to desktop

# ## Examine DeviceInfo

# In[ ]:


train_full['DeviceInfo'].value_counts()


# Since we have way too many devices, it makes more sense to select a few devices that has non-trivial count. Let's select categories that have more than 500 counts:

# In[ ]:


devicelist = train_full.groupby('DeviceInfo').filter(lambda x: len(x) >500)['DeviceInfo'].unique()


# In[ ]:


visualize_cat_cariable('DeviceInfo', df = train_full[train_full['DeviceInfo'].isin(devicelist)])


# **Observation**: We can see the fraud rate is higher for certain devices
# 
# ## Examine id12 - id38
# 
# We may generate all the graphs for id12 to id38. Depend on your preference, some graphs may be more informative than the other. The graphs below are selected based on:
# 
# 1. If the graph contains non-masked information (or categories have self-expalainatory meaning)
#     * For example: 'Found' and 'NotFound' are two categories that by themselves, don't provide us with any helpful information in understanding their relationships with target variable. Perhaps our learner can pickup on the differences, but it's outside of our domain to understand these variables semantically. 
# 
# 2. If the graph contains not too many categories so that the xtickers can be plotted legibly
# 
# You can plots them all out and select for yourself. Here are some of my picks:
# 
# ### IP Proxy

# In[ ]:


# id_list = train_full.loc[:1, 'id_12':'id_38'].columns

# for i in id_list:
#     print (visualize_cat_variable(i))


# In[ ]:


visualize_cat_cariable('id_23')


# **Obervation**: The first notable id plot is the IP status. It is interesting to see the anonymous IP_Proxy would have a higher fraud rate. If someone were to commit a fraudulent transaction, it makes sense that the person would want to protect his/her identity.
# 
# ### Operating Systems

# In[ ]:


visualize_cat_cariable('id_30')


# We can aggregate the operating system into a few major OSs. 

# In[ ]:


train_full['major_os'] = train_full["id_30"].str.split(' ', expand = True)[[0]]

visualize_cat_cariable('major_os')


# **Observation**: The fraud rate across multiple well-known OSs seem fairly similar. "Other" operating systems have a much higher fraud rate.
# 
# However, it's strange that we see more IOS devices compared to Android, given that Android is the most popular mobile system. If I were to work for Vista, I would ask how the system collects more IOS instances. Could it be that Vista have given us an filtered dataset? Specific market segment? Systematic error or deficiency in collecting Android info?
# 
# ### Browsers

# In[ ]:


visualize_cat_cariable('id_31')


# Same as previous plot, we need to reduce the number of categories using aggregation:

# In[ ]:


train_full['browser'] = train_full["id_31"].str.split(' ', expand = True)[[0]]

visualize_cat_cariable('browser')


# We have a few browers that have absurdly high fraud rate. This is likely to due the scarcity of those browsers. We can fix this by apply a minimum-instance-filter. Let's say 0.1 percent of data rows is our cut-off, then each category must have at least 144 instances to be included in our plot:

# In[ ]:


browser_list = train_full.groupby('browser').filter(lambda x: len(x) > 144)['browser'].unique()
visualize_cat_cariable('browser',  df = train_full[train_full['browser'].isin(browser_list)])


# **Observation**: Opera and android browser have relatively high fraud rate

# # Explore Numerical Features
# 
# I anticipate that most variables we will encounter would not follow a normal distribution. Therefore, for each variable, we will explore:
# 
# 1. Distribution
# 
# 2. Log of distribution
# 
# 3. Distribution by target variable
# 
# 4. Log of distribution by target variable
# 
# 5. Boxplot comparison between fraud and non-fraud
# 
# ## Examine Transaction Amount

# In[ ]:


def visualize_num_variable(variable, df=train_full):
    plt.figure(figsize=(12,18))
    plt.suptitle('Distribution of: {}'.format(variable), fontsize=22)

    plt.subplot(321)
    sns.distplot(df[variable], kde= False)
    plt.title('{} Distribution'.format(variable), fontsize = 15)

    plt.subplot(322)
    sns.distplot(np.log10(df[variable]), kde= False)
    plt.title('Log-transformed Distribution', fontsize = 15)


    plt.subplot(323)
    sns.distplot(df[df['isFraud'] == 0][variable], color = 'skyblue', kde= False, label = 'Not Fraud')
    sns.distplot(df[df['isFraud'] == 1][variable], color = 'red', kde= False , label = 'Fraud')
    plt.title('Fraud vs Non-Fraud Distribution', fontsize = 15)
    plt.legend()

    plt.subplot(324)
    sns.distplot(np.log10(df[df['isFraud'] == 0][variable]), color = 'skyblue', kde= False, label = 'Not Fraud')
    sns.distplot(np.log10(df[df['isFraud'] == 1][variable]), color = 'red', kde= False , label = 'Fraud')
    plt.title('Log-transformed Distribution', fontsize = 15)
    plt.legend()
    
    plt.subplot(313)
    sns.boxplot(x = 'isFraud', y = variable, data = df)
    plt.title('Transaction Amount by Fraud', fontsize = 15,  weight='bold')


# In[ ]:


visualize_num_variable('TransactionAmt')


# **Observation**: 
#     1. TransactionAmt has right-skewed distribution: most transactions are small (less than $200)
#     2. There is little difference between distribution and average amount for fraud and non-fraud

# ## Examine Transaction DT

# In[ ]:


visualize_num_variable('TransactionDT')


# **Observation**: There is a large number of non-fraud transactions generated at a certain period . This discrepancy also causing the difference in our boxplot.
# 
# **Possible Improvement**: I should try undersampling the period of non-fraud so that we have less imbalance issue for that particular period.

# ## Examine Distance 2
# 
# Dist1 contains no values. For dist2, we also running into two problems:
# 
# 1. Missing values:
# 
#     Solution: keeping only the non-null rows in dist2.
# 
# 2. Zero values:
# 
#     Zero values cause log transform to return infinity values
# 
#     Solution: add small amount to 0s to avoid infinity
#     
# 3. Negative values
#     
#     The logarithm is only defined for positive numbers. I could perhaps take the log(x+n), where n is the offset values that make the min negative value > 0. However, for such data 0 has a meaning (equality!) that should be respected. Unless I know the meaning of the data, I cannot make arbitrary transformation.
#     
#     Solution: no solution, omit the log-transformation graphs
# 
# Let's update our graphing function with this implementation
# 

# In[ ]:


def visualize_num_variable(variable, df=train_full.copy()):
    # check for homogeneity:
    if len(df[variable].unique()) <= 1:
        print('{} is a homogeneous set'.format(variable))
        return
    
    # check for NAs and Zeros
    if df[variable].isnull().values.any():
        df = train_full.dropna(subset=[variable])

    if df[variable].min() < 0:
        plt.figure(figsize=(12,12))
        plt.suptitle('Distribution of: {}'.format(variable), fontsize=22)
    
        plt.subplot(221)
        sns.distplot(df[variable], kde= False)
        plt.title('{} Distribution'.format(variable), fontsize = 15)
        
        plt.subplot(222)
        sns.distplot(df[df['isFraud'] == 0][variable], color = 'skyblue', kde= False, label = 'Not Fraud')
        sns.distplot(df[df['isFraud'] == 1][variable], color = 'red', kde= False , label = 'Fraud')
        plt.title('Fraud vs Non-Fraud Distribution', fontsize = 15)
        plt.legend()
        
        plt.subplot(212)
        sns.boxplot(x = 'isFraud', y = variable, data = df)
        plt.title('{} by Fraud'.format(variable), fontsize = 15,  weight='bold')
        
    else:
        smallest_value = df[df[variable] != 0][variable].min()
        if df[variable].min() == 0:
            df[variable].replace(0, smallest_value/10, inplace=True)       

        plt.figure(figsize=(12,18))
        plt.text(x=0.5, y=0.5,
                 s="Zeros have been replaced with {} to avoid log infinity".format(smallest_value/10),
                 fontsize=12,horizontalalignment='center')

        plt.suptitle('Distribution of: {}'.format(variable), fontsize=22)

        plt.subplot(321)
        sns.distplot(df[variable], kde= False)
        plt.title('{} Distribution'.format(variable), fontsize = 15)

        plt.subplot(322)
        sns.distplot(np.log10(df[variable]), kde= False)
        plt.title('Log-transformed Distribution', fontsize = 15)


        plt.subplot(323)
        sns.distplot(df[df['isFraud'] == 0][variable], color = 'skyblue', kde= False, label = 'Not Fraud')
        sns.distplot(df[df['isFraud'] == 1][variable], color = 'red', kde= False , label = 'Fraud')
        plt.title('Fraud vs Non-Fraud Distribution', fontsize = 15)
        plt.legend()

        plt.subplot(324)
        sns.distplot(np.log10(df[df['isFraud'] == 0][variable]), color = 'skyblue', kde= False, label = 'Not Fraud')
        sns.distplot(np.log10(df[df['isFraud'] == 1][variable]), color = 'red', kde= False , label = 'Fraud')
        plt.title('Log-transformed Distribution', fontsize = 15)
        plt.legend()

        plt.subplot(313)
        sns.boxplot(x = 'isFraud', y = variable, data = df)
        plt.title('{} by Fraud'.format(variable), fontsize = 15,  weight='bold')


# In[ ]:


visualize_num_variable('dist2')


# **Observation**: Dist2 does not seem to varies between fraud and not-fraud
# 
# ## Examine C features
# 
# Same way of handling a large number of variables, I only choose the notable plots that reflect a large degree of variation. Trying to keep this kernel concised is one of my goals. In this case, I only consider C3 to have some significant patterns:

# In[ ]:


# id_list = train_full.loc[:1, 'C1':'C14'].columns

# for i in id_list:
#     print (visualize_num_variable(i))


# In[ ]:


visualize_num_variable('C3')


# **Observation**: Higher values of C3 associated with no-fraud.
# 
# C5 and C9 are homogeneous columns.
# 
# ## Examine D features

# In[ ]:


# id_list = train_full.loc[:1, 'D1':'D15'].columns

# for i in id_list:
#     print (visualize_num_variable(i))


# In[ ]:


visualize_num_variable('D2')
visualize_num_variable('D8')
visualize_num_variable('D9')


# ## Conclusion for EDA:
# 
# 1. Target variable has class imbalance problem where instance of fraud is much lower than non-fraud
# 
# 2. Multiple columns contain too many missing values
# 
# 3. Several columns are homogeneous, therefore, prodvide no useful information in predicting the target variable (this may not be the case for transaction table since we are using a joined table)
# 
# 4. There is period of time where instances of non-fraud far exceed the usual proportion of non-fraud to fraud 
# 
# 5. Basic understand of variables can help us do simple feature engineering
# 
# We will deal with each problem with the purpose of improving the prediction accuracy. But first, let's try a default XGBoost model provided by Vesta. We can use this model as a baseline to compare the improvement (or reduction) of each engineered feature, change, and alteration that  we made along the way.
# 
# ## Brainstorm
# Before treating this problem like a black box of ensemble learning, it's worthwhile to take our hands off the keyboards and think about the problem of fraud detection in a more "open-box" way. There are a lot of intersting questions worth investigating before diving into the madness of hyperparameters tuning. Insights that could lead to trivial and sometimes important questions. Questions that take us on a journey of curiosity and fulfilment. 
# 
# For the data scientists whose minds love to wander. This section is dedicated for silly and serious questions alike.
# 
# **Scenario** :A Vesta executive storms in the office and excitedly tells everyone that an exciting project has fallen in their laps. It's the fraud detection problem. And he ask his people for some ideas of where to start, which features should be useful in prediciting fraud. He knows it is strange to ask the scientists before attempting any EDA or modeling. After all, they haven't seen a lick of relevant data. But he saids it would be great practive to dip the toe into the water before diving in without any direction. So let's start with the few things that were provided to us: transaction amount, time, card infor, identity, etc... Which information would give us a good start at cracking this problem?
# 
# Let's define clearly what is a fraud transaction first. "Fraud detection is a set of activities undertaken to prevent money or property from being obtained through false pretenses" [Source](https://searchsecurity.techtarget.com/definition/fraud-detection). Most common type of frauds are forging checks or using stolen credit cards. If a person got of hold of your card info, what should he/she do with it? After browsing on Reddit, I found some crude scenarios:
# 
# 1. If you drop your card, it's likely the person who found it by chance and commit a fraud would spend it on consumable and essential products like grocery and gas. The perp will likely go somewhere nearby and spend a larger amount than usual before the card get locked. So perhaps we should look at user's purchase history so that any activities or purchases that deviate from normal buying habit would stand out. But we don't have identifiable data, so we can't go on this route.
# 
# 2. If your information get hacked by careless purchases on some shady websites/gas stations, it's likely that your information will be sold to someone else who use your information for making fraud transaction. This person will make an online purchase and ship it to a distributor, who sells the good for cash and share the profit with the frauder. In this situation, the good is shipped to some far-away place from the user's home address. So the further the distance, the more a transaction looks like a fraud? No, of course not. People sends gifts all the times. But perhaps gifting 3 expensive laptops is slightly more suspicious than gifting a box of chocolate.
#     
#     **Feature Engineering:** Combine transaction amount, type of good, and distance together.
#     
# 3. Fraud commited by someone close to you (family member: spouse, siblings, etc). It's rare, but it could happen.
# 
# 4. Prefered tools for committing fraud. We have learned previously in the EDA that Protonmail has exceptionally high fraud rate >95%. A quick google reveal that Proton is a email service that provide free, anonymous, end-to-end encryption email accounts. Quote from Proton website: "ProtonMail is incorporated in Switzerland and all our servers are located in Switzerland. This means all user data is protected by strict Swiss privacy laws". Meaning fraud perpetrator not only protected by the full extend of the privacy law, but also doing it at no cost. Similarly, we have other tools that also have abnormally high fraud rate such as:
# 
#     * Browser: Comodo IceDragon, Mozilla/Firefox?? (not firefox, but perhaps is Comodo IceDragon but recognized as another version of Firefox?)
#     
#     * Operating system: "other" category has fraud rate of 60%.
#     
#     * Phone (or browser?): Lanix Ilium
#     
#     **Feature Engineering:** New features that emphasize the importance of these tools
#     
# 5. Time of operation. Just like any other jobs, frauders operate at routinely hours that perhaps different from the real users. It is strange, at least to me, to make purchase decision to buy an iphone at 3 in the morning. Again, without historical data, this approach is dead in the egg.
# 
# 
# # Baseline Model
# 
# Modeling section is being explore in another private notebook since only 1 GPU instance is allow in Kaggle Kernel...

# 
