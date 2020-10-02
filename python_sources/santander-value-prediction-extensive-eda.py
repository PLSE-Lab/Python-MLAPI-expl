#!/usr/bin/env python
# coding: utf-8

# 
# <img src="https://upload.wikimedia.org/wikipedia/commons/b/b8/Banco_Santander_Logotipo.svg" width="800"></img>
# 
# <h1><center><font size="6">Santander Value Prediction Extensive EDA</font></center></h1>
# 
# 
# 
# # <a id='0'>Content</a>
# 
# - <a href='#1'>Introduction</a>  
# - <a href='#2'>Load packages</a>  
# - <a href='#3'>Read the data</a>  
# - <a href='#4'>Check the data</a>
#     - <a href='#41'>Glimpse the data</a>  
#     - <a href='#42'>Check missing data</a>  
#     - <a href='#43'>Check data sparsity</a>
# - <a href='#5'>Data exploration</a>
#     - <a href='#51'>Features type</a>
#     - <a href='#52'>Data sparsity per column type</a>
#     - <a href='#521'>Constant columns</a>  
#     - <a href='#53'>Target variable</a>  
#     - <a href='#54'>Distribution of non-zeros per row</a>  
#     - <a href='#55'>Distribution of non-zeros per column</a>  
#     - <a href='#56'>Float features</a>  
#     - <a href='#57'>Integer features</a>      
#     - <a href='#58'>Highly correlated features</a>  
# - <a href='#6'>Model</a>    
# - <a href='#7'>Submission</a>
# - <a href='#8'>References</a>

# # <a id="1">Introduction</a>  
# 
# Santander Group aims to go a step beyond recognizing that there is a need to provide a customer a financial service and intends to **determine the amount or value of the customer's transaction**. This means anticipating customer needs in a more concrete, but also simple and personal way. With so many choices for financial services, this need is greater now than ever before.
# 
# In this competition, Santander Group is asking Kagglers to help them identify the value of transactions for each potential customer. This is a first step that Santander needs to nail in order to personalize their services at scale.
# 
# **Note**: This Kernel was made before Giba published his findings about the leak, which changed dramatically this competition.
# 
# **Late note**: Congratulation to Giba for winning 1st place in this Competition!
# 
# <a href="#0"><font size="1">Go to top</font></a>

# # <a id="2">Load packages</a>

# In[ ]:


import pandas as pd 
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from math import pi
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

import warnings
warnings.filterwarnings("ignore")

# Print all rows and columns
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
IS_LOCAL = False

import os

if(IS_LOCAL):
    PATH="../input/santander-value-prediction-challenge"
else:
    PATH="../input"
print(os.listdir(PATH))


# # <a id="3">Read the data</a>

# In[ ]:


train_df = pd.read_csv(PATH+"/train.csv")
test_df = pd.read_csv(PATH+"/test.csv")


# <a href="#0"><font size="1">Go to top</font></a>
# 
# # <a id="4">Check the data</a>

# In[ ]:


print("Santander Value Prediction Challenge train -  rows:",train_df.shape[0]," columns:", train_df.shape[1])


# There are **4459** data rows and **4993** columns.

# In[ ]:


print("Santander Value Prediction Challenge test -  rows:",test_df.shape[0]," columns:", test_df.shape[1])


# The schema dataset contains **49342** rows - and **4992** columns (target column missing).
# 
# 
# There are few observations that we can already make:
# * The column number exceeds the rows number for the train data.  
# * The test data is containing almost 10 times more data than the train data.  
# 
# 
# 
# <a href="#0"><font size="1">Go to top</font></a>

# ## <a id="41">Glimpse the data</a>
# 
# We start by looking to the data features (first 5 rows).

# In[ ]:


train_df.head()


# In[ ]:


test_df.head()


# The columns in the train data are as following: 
# 
# * **ID**: we will have to check if the ID is in anyway connected with the column names; these are hexa numbers with 9 digits.
# * **target**: this is the *target* variable and has numeric (real) values;  
# * 4991 columns with names anonymized - there are hexa large numbers with 9 digits.  Most of the columns have 0 values, the dataset is sparse. The columns types seems to be integers and reals.
# 
# 
# Test data has the same columns, without **target**.
# 

# ## <a id="42">Check missing data</a>  
# 
# Let's check the missing data for train set.

# In[ ]:


def check_nulls(df):
    nulls = df.isnull().sum(axis=0).reset_index()
    nulls.columns = ['column', 'missing']
    nulls = nulls[nulls['missing']>0]
    nulls = nulls.sort_values(by='missing')
    return nulls    


check_nulls(train_df)


# There are no missing data in the train set.  
# 
# Let's check the missing data for test set.

# In[ ]:


check_nulls(test_df)


# There are no missing data in the test set either.
# 
# 
# 
# ## <a id="43">Check data sparsity</a>  
# 
# Let's check the data sparsity for train set.

# In[ ]:


def check_sparsity(df):
    non_zeros = (df.ne(0).sum(axis=1)).sum()
    total = df.shape[1]*df.shape[0]
    zeros = total - non_zeros
    sparsity = round(zeros / total * 100,2)
    density = round(non_zeros / total * 100,2)

    print(" Total:",total,"\n Zeros:", zeros, "\n Sparsity [%]: ", sparsity, "\n Density [%]: ", density)
    return density

d1 = check_sparsity(train_df)


# Let's check the data sparsity for test set.

# In[ ]:


d2 = check_sparsity(test_df)


# One important observation is that the data sparsity is slightly larger for the test set than for the train set (density is more than double for the train set). We will look into more details about the data distribution in the following section.
# 
# 
# <a href="#0"><font size="1">Go to top</font></a>

# # <a id="5">Data exploration</a>
# 
# ##  <a id="51">Features type</a>
# 
# 
# Let's check the features type in the data.

# In[ ]:


dtype_df = train_df.dtypes.reset_index()
dtype_df.columns = ["Count", "Column Type"]
dtype_df.groupby("Column Type").aggregate('count').reset_index()


# There are 3147 integer features, 1845 float values and one non-numeric value (the ID field).  
# 
# Let's save the **metadata** for the columns. For each feature we set the metadata for **role**, if we will use the feature - **keep** and the **dtype**. 

# In[ ]:


data = []
for feature in train_df.columns:
    # Defining the role
    if feature == 'target':
        use = 'target'
    elif feature == 'ID':
        use = 'id'
    else:
        use = 'input'
         
        
    # Initialize keep to True for all variables except for `ID`
    keep = True
    if feature == 'ID':
        keep = False
    
    # Defining the data type 
    dtype = train_df[feature].dtype
    
    
    
    # Creating a Dict that contains all the metadata for the variable
    feature_dictionary = {
        'varname': feature,
        'use': use,
        'keep': keep,
        'dtype': dtype,
    }
    data.append(feature_dictionary)
    
# Create the metadata
metadata = pd.DataFrame(data, columns=['varname', 'use', 'keep', 'dtype'])
metadata.set_index('varname', inplace=True)

# Sample the metadata
metadata.head(10)


# Let's check that we have the expected distribution of the dtype in the metadata (as identified before).

# In[ ]:


pd.DataFrame({'count' : metadata.groupby(['dtype'])['dtype'].size()}).reset_index()


# <a href="#0"><font size="1">Go to top</font></a>  
# 
# 
# ## <a id="52">Data sparsity per column type</a>
# 
# 
# ### Integer type
# 
# For train data:

# In[ ]:


int_data = []
var = metadata[(metadata.dtype == 'int64') & (metadata.use == 'input')].index
d3 = check_sparsity(train_df[var])


# For test data:

# In[ ]:


d4 = check_sparsity(test_df[var])


# ### Float type
# 
# For train data:

# In[ ]:


var = metadata[(metadata.dtype == 'float64') & (metadata.use == 'input')].index
d5 = check_sparsity(train_df[var])


# For test data:

# In[ ]:


d6 = check_sparsity(test_df[var])


# Let's put together all these data and compare them. For convenience, we will compare the densities.

# In[ ]:


data = {'Dataset': ['Train', 'Test'], 'All': [d1, d2], 'Integer': [d3,d4], 'Float': [d5,d6]}
    
density_data = pd.DataFrame(data)
density_data.set_index('Dataset', inplace=True)
density_data


# The data density for train set is 2.23 times larger than for the test set.   
# 
# As well, the train float data density is 5.7 times larger than the train integer data density.  
# The test float data density is 2.87 time larger than test integer data density.  
# 
# In general density is larger in train set than in test set and in float data than in integer data.

# <a href="#0"><font size="1">Go to top</font></a>  
# 
# 
# ##  <a id="521">Constant columns</a>
# 
# Let's check if there are constant columns.

# In[ ]:


# check constant columns
colsConstant = []
columnsList = [x for x in train_df.columns if not x in ['ID','target']]

for col in columnsList:
    if train_df[col].std() == 0: 
        colsConstant.append(col)
print("There are", len(colsConstant), "constant columns in the train set.")


# Let's mark all these columns to drop.

# In[ ]:


metadata['keep'].loc[colsConstant] = False


# <a href="#0"><font size="1">Go to top</font></a>  
# 
# 
# ##  <a id="53">Target variable</a>
# 
# 
# Let's check the target variable distribution.

# In[ ]:


# Plot distribution of one feature
def plot_distribution(df,feature,color):
    plt.figure(figsize=(10,6))
    plt.title("Distribution of %s" % feature)
    sns.distplot(df[feature].dropna(),color=color, kde=True,bins=100)
    plt.show()   
    
plot_distribution(train_df, "target", "blue")


# Let's check the distribution of log(target).

# In[ ]:


def plot_log_distribution(df,feature,color):
    plt.figure(figsize=(10,6))
    plt.title("Distribution of %s" % feature)
    sns.distplot(np.log1p(df[feature]).dropna(),color=color, kde=True,bins=100)
    plt.title("Distribution of log(target)")
    plt.show()   

plot_log_distribution(train_df, "target", "green")  


# <a href="#0"><font size="1">Go to top</font></a>  
# 
# 
# ##  <a id="54">Distribution of non-zero features values per row</a>
# 
# Let's check what is the distribution of non-zero features values per row in the train set.

# In[ ]:


non_zeros = (train_df.ne(0).sum(axis=1))

plt.figure(figsize=(10,6))
plt.title("Distribution of log(number of non-zeros per row) - train set")
sns.distplot(np.log1p(non_zeros),color="red", kde=True,bins=100)
plt.show()


# Let's check distribution of non-zero features values per row in the test set.

# In[ ]:


non_zeros = (test_df.ne(0).sum(axis=1))

plt.figure(figsize=(10,6))
plt.title("Distribution of log(number of non-zeros per row) - test set")
sns.distplot(np.log1p(non_zeros),color="magenta", kde=True,bins=100)
plt.show()


# Let's separate only the **real** values, excepting the **target**. And let's represent the distribution of non-zero features values only for these.
# 
# ### Distribution of non-zeros for float type features

# In[ ]:


var = metadata[(metadata.dtype == 'float64') & (metadata.use == 'input')].index
non_zeros = (train_df[var].ne(0).sum(axis=1))

plt.figure(figsize=(10,6))
plt.title("Distribution of log(number of non-zeros per row) - floats only - train set")
sns.distplot(np.log1p(non_zeros),color="green", kde=True,bins=100)
plt.show()


# In[ ]:


non_zeros = (test_df[var].ne(0).sum(axis=1))

plt.figure(figsize=(10,6))
plt.title("Distribution of log(number of non-zeros per row) - floats only - test set")
sns.distplot(np.log1p(non_zeros),color="blue", kde=True,bins=100)
plt.show()


# ### Distribution of non-zeros for integer type features

# In[ ]:


var = metadata[(metadata.dtype == 'int64') & (metadata.use == 'input')].index
non_zeros = (train_df[var].ne(0).sum(axis=1))

plt.figure(figsize=(10,6))
plt.title("Distribution of log(number of non-zeros per row) - integers only -  train set")
sns.distplot(np.log1p(non_zeros),color="yellow", kde=True,bins=100)
plt.show()


# In[ ]:


non_zeros = (test_df[var].ne(0).sum(axis=1))

plt.figure(figsize=(10,6))
plt.title("Distribution of log(number of non-zeros per row) - integers only - train set")
sns.distplot(np.log1p(non_zeros),color="cyan", kde=True,bins=100)
plt.show()


# <a href="#0"><font size="1">Go to top</font></a>
# 
# 
# ##  <a id="55">Distribution of non-zero features values per column</a>
# 
# Let's check what is the distribition of non-zero features values per column in the train set.

# In[ ]:


non_zeros = (train_df.ne(0).sum(axis=0))

plt.figure(figsize=(10,6))
plt.title("Distribution of log(number of non-zeros per column) - train set")
sns.distplot(np.log1p(non_zeros),color="darkblue", kde=True,bins=100)
plt.show()


# Let's check distribution of non-zero features values per row in the test set.

# In[ ]:


non_zeros = (test_df.ne(0).sum(axis=0))

plt.figure(figsize=(10,6))
plt.title("Distribution of log(number of non-zeros per column) - test set")
sns.distplot(np.log1p(non_zeros),color="darkgreen", kde=True,bins=100)
plt.show()


# <a href="#0"><font size="1">Go to top</font></a>  
# 
# 
# ##  <a id="56">Float features</a>
# 
# Let's see now  the distribution of the sum of float features values per column.

# In[ ]:


var = metadata[(metadata.dtype == 'float64') & (metadata.use == 'input')].index
val = train_df[var].sum()

fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12,6))
sns.boxplot(val, palette="Blues",  showfliers=False,ax=ax1)
sns.boxplot(val, palette="Greens",  showfliers=True,ax=ax2)
plt.show();


# <a href="#0"><font size="1">Go to top</font></a>  
# 
# 
# ##  <a id="57">Integer features</a>
# 
# Let's see now  the distribution of the sum of integer features values per column.

# In[ ]:


var = metadata[(metadata.dtype == 'int64') & (metadata.use == 'input')].index
val = train_df[var].sum()

fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12,6))
sns.boxplot(val, palette="Reds",  showfliers=False,ax=ax1)
sns.boxplot(val, palette="Blues",  showfliers=True,ax=ax2)
plt.show();


# <a href="#0"><font size="1">Go to top</font></a>  
# 
# 
# ##  <a id="58">Highly correlated features</a>
# 
# 
# We use a code snapshot from <a href="#7">[1]</a> to extract the features that are highly correlated with **target** feature. We select only the features correlated or inverse correlated with **target** and having a corrlelation coefficient 
# 

# In[ ]:


labels = []
values = []
for col in train_df.columns:
    if col not in ["ID", "target"]:
        labels.append(col)
        values.append(np.corrcoef(train_df[col].values, train_df["target"].values)[0,1])
corr_df = pd.DataFrame({'columns_labels':labels, 'corr_values':values})
corr_df = corr_df.sort_values(by='corr_values')
 
corr_df = corr_df[(corr_df['corr_values']>0.25) | (corr_df['corr_values']<-0.25)]
ind = np.arange(corr_df.shape[0])
width = 0.9
fig, ax = plt.subplots(figsize=(10,6))
rects = ax.barh(ind, np.array(corr_df.corr_values.values), color='gold')
ax.set_yticks(ind)
ax.set_yticklabels(corr_df.columns_labels.values, rotation='horizontal')
ax.set_xlabel("Correlation coefficient")
ax.set_title("Correlation coefficient of the variables")
plt.show()


# Let's represent the correlation map between these selected features.

# In[ ]:


temp_df = train_df[corr_df.columns_labels.tolist()]
corrmat = temp_df.corr(method='pearson')
f, ax = plt.subplots(figsize=(12, 12))

# Draw the heatmap using seaborn
sns.heatmap(corrmat, vmax=1., square=True, cmap="YlOrRd")
plt.title("Important variables correlation map", fontsize=15)
plt.show()


# Let's represent, for the highly correlated features, the distribution in the train and test set.

# In[ ]:


corrmat


# In[ ]:


var = temp_df.columns.values

i = 0
sns.set_style('whitegrid')
plt.figure()
fig, ax = plt.subplots(5,4,figsize=(12,15))

for feature in var:
    i += 1
    plt.subplot(5,4,i)
    sns.kdeplot(train_df[feature], bw=0.5,label="train")
    sns.kdeplot(test_df[feature], bw=0.5,label="test")
    plt.xlabel(feature, fontsize=12)
    locs, labels = plt.xticks()
    plt.tick_params(axis='both', which='major', labelsize=12)
plt.show();


# Let's represent the relationship between two of the highest correlated features ('429687d5a';'e4159c59e') and ('6b119d8ce';'e8d9394a0').

# In[ ]:


sns.set_style('whitegrid')
plt.figure()
s = sns.lmplot(x='429687d5a', y='e4159c59e',data=train_df, fit_reg=True,scatter_kws={'s':2})
s = sns.lmplot(x='6b119d8ce', y='e8d9394a0',data=train_df, fit_reg=True,scatter_kws={'s':2})
s = sns.lmplot(x='cbbc9c431', y='f296082ec',data=train_df, fit_reg=True,scatter_kws={'s':2})
s = sns.lmplot(x='cbbc9c431', y='51707c671',data=train_df, fit_reg=True,scatter_kws={'s':2})
plt.show()


# 
# 
# <a href="#0"><font size="1">Go to top</font></a>

# # <a id="6">Model</a>  
# 
# Let's create now a model. We start by droping the duplicate columns.
# 
# 

# In[ ]:


var = metadata[(metadata.keep == False) & (metadata.use == 'input')].index
train_df.drop(var, axis=1, inplace=True)  
test_df.drop(var, axis=1, inplace=True)  


# Let's check the shape of train and test set after droping the columns.

# In[ ]:


print("Santander Value Prediction Challenge train -  rows:",train_df.shape[0]," columns:", train_df.shape[1])
print("Santander Value Prediction Challenge test -  rows:",test_df.shape[0]," columns:", test_df.shape[1])


# Let's add few statistical features. But before let's replace all 0s with NAs.

# In[ ]:


# Replace 0 with NAs
train_df.replace(0, np.nan, inplace=True)
test_df.replace(0, np.nan, inplace=True)


# In[ ]:


all_features = [f for f in train_df.columns if f not in ['target', 'ID']]
for df in [train_df, test_df]:
    df['nans'] = df[all_features].isnull().sum(axis=1)
    # All of the stats will be computed without the 0s 
    df['median'] = df[all_features].median(axis=1)
    df['mean'] = df[all_features].mean(axis=1)
    df['sum'] = df[all_features].sum(axis=1)
    df['std'] = df[all_features].std(axis=1)
    df['kurtosis'] = df[all_features].kurtosis(axis=1)


# We include only the selected input features to keep and the new statistical features calculated. 

# In[ ]:


features = all_features + ['nans', 'median', 'mean', 'sum', 'std', 'kurtosis']


# Let's check again the shape.

# In[ ]:


print("Santander Value Prediction Challenge train -  rows:",train_df.shape[0]," columns:", train_df.shape[1])
print("Santander Value Prediction Challenge test -  rows:",test_df.shape[0]," columns:", test_df.shape[1])


# We create the split with 5 folds. We build the model for training and we init the predictions.

# In[ ]:


# Create folds
folds = KFold(n_splits=5, shuffle=True, random_state=1)


# Convert to lightgbm Dataset
dtrain = lgb.Dataset(data=train_df[features], label=np.log1p(train_df['target']), free_raw_data=False)
# Construct dataset so that we can use slice()
dtrain.construct()
# Init predictions
sub_preds = np.zeros(test_df.shape[0])
oof_preds = np.zeros(train_df.shape[0])


# Let's add the lgb parameters.

# In[ ]:


lgb_params = {
    'objective': 'regression',
    'num_leaves': 100,
    'subsample': 0.8,
    'colsample_bytree': 0.75,
    'verbose': -1,
    'seed': 2018,
    'boosting_type': 'gbdt',
    'max_depth': 10,
    'learning_rate': 0.04,
    'metric': 'l2',
}


# Train and fit the lgb model with 5 folds. Then calculate the Full Out-Of-Fold score according to <a href="#8">[3]</font></a>.

# In[ ]:


# Run KFold
for trn_idx, val_idx in folds.split(train_df):
    # Train lightgbm
    clf = lgb.train(
        params=lgb_params,
        train_set=dtrain.subset(trn_idx),
        valid_sets=dtrain.subset(val_idx),
        num_boost_round=10000, 
        early_stopping_rounds=100,
        verbose_eval=50
    )
    # Predict Out Of Fold and Test targets
    # Using lgb.train, predict will automatically select the best round for prediction
    oof_preds[val_idx] = clf.predict(dtrain.data.iloc[val_idx])
    sub_preds += clf.predict(test_df[features]) / folds.n_splits
    # Display current fold score
    print('Current fold score : %9.6f' % mean_squared_error(np.log1p(train_df['target'].iloc[val_idx]), 
                             oof_preds[val_idx]) ** .5)
    
# Display Full OOF score (square root of a sum is not the sum of square roots)
print('Full Out-Of-Fold score : %9.6f' 
      % (mean_squared_error(np.log1p(train_df['target']), oof_preds) ** .5))


# Let's plot feature importance. We select the first 50 features.

# In[ ]:


fig, ax = plt.subplots(figsize=(14,10))
lgb.plot_importance(clf, max_num_features=50, height=0.8,color="tomato",ax=ax)
plt.show()


# # <a id="7">Submission</a>
# 
# Let's prepare a submission.

# In[ ]:


sub = test_df[['ID']].copy()
sub['target'] = np.expm1(sub_preds)
sub[['ID', 'target']].to_csv('submission.csv', index=False)


# # <a id="8">References</a>  
# 
# [1] <a href="https://www.kaggle.com/sudalairajkumar">SRK</a>, <a href="https://www.kaggle.com/sudalairajkumar/simple-exploration-notebook-santander-value">Simple Exploration Notebook - Santander Value</a>  
# [2] <a href="https://www.kaggle.com/samratp">Samrat Pandiri</a>, <a href="https://www.kaggle.com/samratp/aggregates-sumvalues-sumzeros-k-means-pca">Aggregates + SumValues + SumZeros + K-Means + PCA</a>  
# [3] <a href="https://www.kaggle.com/ogrellier">olivier</a>, <a href="https://www.kaggle.com/ogrellier/santander-46-features">Santander_46_features</a>
