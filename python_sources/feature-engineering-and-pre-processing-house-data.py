#!/usr/bin/env python
# coding: utf-8

# # Feature Engineering and Pre-Processing for House Price Data
# #### By Nick Brooks
# 
# ### Content: 
# - 
# 

# In[ ]:


# Generael
import numpy as np
import pandas as pd
import os

# Visualization
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[ ]:


# Read
train_df = pd.read_csv("/kaggle/input/train.csv", index_col='Id')
test_df = pd.read_csv("/kaggle/input/test.csv", index_col='Id')

# Combine Train and Test for unified processing
combine = [train_df, test_df]


# In[ ]:


pd.options.display.max_rows = 65
dtype_df = train_df.dtypes.reset_index()
dtype_df.columns = ["Count", "Column Type"]
dtype_df.groupby("Column Type").aggregate('count').sort_values(by=["Count"], ascending=False).plot(kind="bar")
plt.xticks(rotation=35)
plt.ylabel('Count')
plt.title("Variable Dtypes")
plt.show()


# In[ ]:


object_info = pd.DataFrame()
num_info = pd.DataFrame()

for x in combine[0].columns:
    # Missing Values Dataframe
    if combine[0][x].isnull().any() == True:
        object_info = object_info.append({'Column': x,'dtype': combine[0][x].dtypes,
        'Count': combine[0][x].count().astype(int),
        'Missing %':(combine[0][x].isnull().sum()/combine[0].shape[0])*100,
        'Unique':len(combine[0][x].unique())},ignore_index=True)
    # Custom Descriptive Statistics Table
    if combine[0][x].dtype != "object" :
        num_info = num_info.append({'Column': x, 'dtype': combine[0][x].dtypes, 'Count':
        combine[0][x].count().astype(int), 'Missing %':(combine[0][x].isnull().sum()/combine[0].shape[0])*100,
        'Unique': len(combine[0][x].unique()), 'Stdev':combine[0][x].std(),
        'Mean':combine[0][x].mean(), 'Stdev':combine[0][x].std(),
        'Variance':combine[0][x].var()},ignore_index=True)
object_info.sort_values(by=["Missing %"], ascending=False, inplace=True)


# In[ ]:


object_info


# In[ ]:


num_info


# ## Dependent Variable
# 
# - Sales Price Matplotlib Histogram
# - Log Sales Price Seaborn Histogram
# 
# Fairly normally distributed, with a right tail.

# In[ ]:


# Histogram with Matplotlib
plt.hist(train_df.SalePrice, normed=True, bins=30)
plt.xlabel('Sales Price')
plt.title('Sales Price Histogram')
plt.ylabel('Frequency');


# In[ ]:


# Log Histogram with Seaborn
plt.figure(figsize=(12,8))
sns.distplot(np.log2(train_df.SalePrice), bins=30, kde=False, rug=True, color="g")
plt.xlabel('Log Sale Price', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.title("Log Sale Price Histogram",fontsize=30)
plt.ylim(-25, 300)
plt.show()


# ## Missing Values

# In[ ]:


# Helpers
# Proportion Missing Table:
settypes= combine[0].dtypes.reset_index()
def test_train_mis(test, train):
    missing_test = test.isnull().sum(axis=0).reset_index()
    missing_test.columns = ['column_name', 'test_missing_count']
    missing_test['test_missing_ratio'] = (missing_test['test_missing_count'] / test_df.shape[0])*100
    missing_train = train.isnull().sum(axis=0).reset_index()
    missing_train.columns = ['column_name', 'train_missing_count']
    missing_train['train_missing_ratio'] = (missing_train['train_missing_count'] / train_df.shape[0])*100
    missing = pd.merge(missing_train, missing_test,
                       on='column_name', how='outer',indicator=True,)
    missing = pd.merge(missing,settypes, left_on='column_name', right_on='index',how='inner')
    missing = missing.loc[(missing['train_missing_ratio']>0) | (missing['test_missing_ratio']>0)]    .sort_values(by=["train_missing_ratio"], ascending=False)
    missing['Diff'] = missing.train_missing_ratio - missing.test_missing_ratio
    return missing

def missing_plot(train_df):
    missing_df = train_df.isnull().sum(axis=0).reset_index()
    missing_df.columns = ['column_name', 'missing_count']
    missing_df = missing_df.loc[missing_df['missing_count']>0]
    missing_df = missing_df.sort_values(by='missing_count')

    ind = np.arange(missing_df.shape[0])
    width = 0.9
    fig, ax = plt.subplots(figsize=(12,18))
    rects = ax.barh(ind, missing_df.missing_count.values, color='red')
    ax.set_yticks(ind)
    ax.set_yticklabels(missing_df.column_name.values, rotation='horizontal')
    ax.set_xlabel("Count of missing values")
    ax.set_title("Number of missing values in each column")
    plt.show()


# ## Visualize

# Next, I wish to study the difference in missing values between the train and test set.

# In[ ]:


missing = pd.DataFrame(test_train_mis(combine[0],combine[1]))
missing


# In[ ]:


missing.head()


# In[ ]:


f, (ax1, ax2, ax3) = plt.subplots(1,3,figsize=(15,8), sharey=True)
ax1 = sns.barplot(y ="column_name", x ="train_missing_ratio", data=missing,ax=ax1)
ax1.set_xlabel('Percent of Data Missing')
ax1.set_title('Train Set Percent Missing')
ax1.set_ylabel('Independent Variables')

ax2 = sns.barplot(y ="column_name", x ="Diff", data=missing,ax=ax2)
ax2.set_xlabel('Test/Train Missing Difference')
ax2.set_title('Test/Train Missing Difference by Variable')
ax2.set_ylabel('')

ax3 = sns.barplot(y ="column_name", x ="test_missing_ratio", data=missing,ax=ax3)
ax3.set_xlabel('Percent of Data Missing')
ax3.set_title('Test Set Percent Missing')
ax3.set_ylabel('')

f.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
plt.savefig("test-train missing values and diff plot.png")


# In[ ]:


# Delete Columns with High Missing
for i in list(range(len(combine))):
    combine[i].drop(missing.column_name[(missing.train_missing_ratio > 45) |
                                        (missing.test_missing_ratio > 45)],
                    axis=1, inplace=True)


# ## Imputation

# In[ ]:


for i in list(range(len(combine))):
    combine[i].loc[:, combine[i].dtypes == float] = combine[i].loc[:, combine[i].dtypes == float].fillna(combine[i].mean())
    combine[i].loc[:, combine[i].dtypes == object] = combine[i].loc[:, combine[i].dtypes == object].fillna(combine[i].mode().iloc[0])
    combine[i].loc[:, combine[i].dtypes == int] =combine[i].loc[:, combine[i].dtypes == int].fillna(combine[i].median())

print("Train Missing Values? -> {}".format(combine[0].isnull().values.any()))
print("Test Missing Values? -> {}".format(combine[1].isnull().values.any()))


# In[ ]:


"""
RETIRED CODE

 for dataset in combine:
     print(np.count_nonzero(dataset.isnull().values.ravel()))

# Dummy Variables
for i in list(range(len(combine))):
    combine[i] = pd.get_dummies(combine[i],
        columns=combine[i].select_dtypes(include=['object']).columns)
    
diffcols = combine[0].columns.difference(combine[1].columns)
print(diffcols)
SalePrice = combine[0].SalePrice.copy()
combine[0] = combine[0].drop(diffcols, axis=1)

pd.options.display.max_rows = 65
dtype_df = combine[0].dtypes.reset_index()
dtype_df.columns = ["Count", "Column Type"]
dtype_df.groupby("Column Type").aggregate('count').reset_index()

combine[0].columns.equals(combine[1].columns)
"""


# In[ ]:


# Object Variables
combine[0].select_dtypes(include=['object']).columns


# In[ ]:


combine[0].shape, combine[1].shape


# In[ ]:


combine[0].to_csv("house_train.csv",header=True,index=True)
combine[1].to_csv("house_test.csv",header=True,index=True)

