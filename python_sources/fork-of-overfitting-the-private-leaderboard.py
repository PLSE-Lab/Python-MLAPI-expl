#!/usr/bin/env python
# coding: utf-8

# <h1><center><font size="6">Overfitting the private leaderboard</font></center></h1>
# 
# <img src="https://cdn-images-1.medium.com/max/1500/1*_7OPgojau8hkiPUiHoGK_w.png" align="center" width=800/>
# <br>
# 
# # <a id='0'>Content</a>
# 
# - <a href='#1'>Introduction</a>  
# - <a href='#2'>Prepare the data analysis</a>  
# - <a href='#3'>Data exploration</a>   
#  - <a href='#31'>Check the data</a>   
#  - <a href='#32'>Density plots of features</a>   
#  - <a href='#33'>Distribution of mean and std</a>   
#   - <a href='#34'>Distribution of min and max</a>   
#  - <a href='#35'>Distribution of skew and kurtosis</a>     
#  - <a href='#36'>Features correlations</a>   
# - <a href='#4'>Feature engineering</a>  
#  - <a href='#41'>Add features by aggregation</a>  
#  - <a href='#42'>Add noise</a>  
# - <a href='#5'>Model</a>
# - <a href='#6'>Submission</a>  
# - <a href='#7'>References</a>

# # <a id='1'>Introduction</a>  
# 
# In this challenge, Kagglers are invited to not overfit. 
# 
# Train data has 250 rows, test data has 20,000 rows.
# 
# There are 300 features.
# 
# This Kernel will start by exploring the data and check what engineered features will improve the model (well, reduce the overfitting).
# 
# 

# # <a id='2'>Prepare for data analysis</a>  
# 
# 
# ## Load packages
# 

# In[ ]:


import gc
import os
import logging
import datetime
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import lightgbm as lgb
from tqdm import tqdm_notebook
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from scipy import signal
from sklearn.decomposition import FastICA, PCA
warnings.filterwarnings('ignore')


# ## Load data   
# 
# Let's check what data files are available.

# In[ ]:


IS_LOCAL = False
if(IS_LOCAL):
    PATH="../input/dont-overfit-2/"
else:
    PATH="../input/"
os.listdir(PATH)


# Let's load the train and test data files.

# In[ ]:


get_ipython().run_cell_magic('time', '', 'train_df = pd.read_csv(os.path.join(PATH,"train.csv"))\ntest_df = pd.read_csv(os.path.join(PATH,"test.csv"))')


# # <a id='3'>Data exploration</a>  
# 
# ## <a id='31'>Check the data</a>  
# 
# Let's check the train and test set.

# In[ ]:


train_df.shape, test_df.shape


# Train has only 250 rows and has also 302 columns, test has 19,750 rows and 301 columns.  
# 
# Let's glimpse train and test dataset.

# In[ ]:


train_df.head()


# In[ ]:


test_df.head()


# Train contains:  
# 
# * **id** (string);  
# * **target**;  
# * **300** numerical variables, named from **0** to **299**;
# 
# Test contains:  
# 
# * **id* (string);  
# * **300** numerical variables, named from **0** to **299**;
# 
# 
# Let's check if there are any missing data. We will also chech the type of data.
# 
# We check first train.

# In[ ]:


def missing_data(data):
    total = data.isnull().sum()
    percent = (data.isnull().sum()/data.isnull().count()*100)
    tt = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    types = []
    for col in data.columns:
        dtype = str(data[col].dtype)
        types.append(dtype)
    tt['Types'] = types
    return(np.transpose(tt))


# In[ ]:


get_ipython().run_cell_magic('time', '', 'missing_data(train_df)')


# Only **id** is an integer, **target** and the 300 features are float64. There are no missing data.
# 
# Here we check test dataset.

# In[ ]:


get_ipython().run_cell_magic('time', '', 'missing_data(test_df)')


# There are no missing data either in test data. The data types are similar with the ones in train.   
# 
# Let's see the distribution of train and test numerical data, using `describe`.

# In[ ]:


get_ipython().run_cell_magic('time', '', 'train_df.describe()')


# In[ ]:


get_ipython().run_line_magic('time', '')
test_df.describe()


# We can make few observations here:   
# 
# * standard deviation is very close to 1 for all features, in both train and test set;  
# * all features are approximately centered to 0, with mean values close to 0;  
# * min and max absolute values for features in train data looks to be smaller than the ones for the test data;  
# * mean value for target variable is 0.64 which will imply that 64% of target values are 1.
# 
# 

# Let's check the distribution of **target** value in train dataset.

# In[ ]:


sns.countplot(train_df['target'])


# In[ ]:


print("There are {}% target values with 1".format(100 * train_df["target"].value_counts()[1]/train_df.shape[0]))


# Let's plot now the train data (all the data) using a heatmap, separatelly for target values 0 and 1.  
# 

# In[ ]:


t0 = train_df.loc[train_df['target'] == 0, train_df.columns.values[2:302]]
t1 = train_df.loc[train_df['target'] == 1, train_df.columns.values[2:302]]
plt.subplots(1,1,figsize=(20, 7.2))
plt.title("Values in training set for target = 0")
sns.heatmap(t0, cmap="Spectral")
plt.show()


# In[ ]:


plt.subplots(1,1,figsize=(20, 12.8))
plt.title("Values in training set for target = 1")
sns.heatmap(t1, cmap='Spectral')
plt.show()


# 
# ## <a id='32'>Density plots of features</a>  
# 
# Let's show now the density plot of variables in train dataset. 
# 
# We represent with different colors the distribution for values with **target** value **0** and **1**.

# In[ ]:


def plot_feature_distribution(df1, df2, label1, label2, features):
    i = 0
    sns.set_style('whitegrid')
    plt.figure()
    fig, ax = plt.subplots(10,10,figsize=(18,22))

    for feature in features:
        i += 1
        plt.subplot(10,10,i)
        sns.kdeplot(df1[feature], bw=0.5,label=label1)
        sns.kdeplot(df2[feature], bw=0.5,label=label2)
        plt.xlabel(feature, fontsize=9)
        locs, labels = plt.xticks()
        plt.tick_params(axis='x', which='major', labelsize=6, pad=-6)
        plt.tick_params(axis='y', which='major', labelsize=6)
    plt.show();


# The first 100 values are displayed in the following cell. Press <font color='red'>**Output**</font> to display the plots.

# In[ ]:


t0 = train_df.loc[train_df['target'] == 0]
t1 = train_df.loc[train_df['target'] == 1]
features = train_df.columns.values[2:102]
plot_feature_distribution(t0, t1, '0', '1', features)


# The next 100 values are displayed in the following cell. Press <font color='red'>**Output**</font> to display the plots.

# In[ ]:


features = train_df.columns.values[102:202]
plot_feature_distribution(t0, t1, '0', '1', features)


# The next 100 values are displayed in the following cell. Press <font color='red'>**Output**</font> to display the plots.

# In[ ]:


features = train_df.columns.values[202:302]
plot_feature_distribution(t0, t1, '0', '1', features)


# We can observe that most of features present significant different distribution for the two target values.  
# 
# Also some features, like **77**, **95**, **147** shows a distribution that resambles to a bivariate distribution.
# 
# 
# Le't s now look to the distribution of the same features in parallel in train and test datasets. 
# 
# The first 100 values are displayed in the following cell. Press <font color='red'>**Output**</font> to display the plots.

# In[ ]:


features = train_df.columns.values[2:102]
plot_feature_distribution(train_df, test_df, 'train', 'test', features)


# The next 100 values are displayed in the following cell. Press <font color='red'>**Output**</font> to display the plots.

# In[ ]:


features = train_df.columns.values[102:202]
plot_feature_distribution(train_df, test_df, 'train', 'test', features)


# The next 100 values are displayed in the following cell. Press <font color='red'>**Output**</font> to display the plots.

# In[ ]:


features = train_df.columns.values[202:302]
plot_feature_distribution(train_df, test_df, 'train', 'test', features)


# The train and test seems to be well ballanced with respect of  distribution of the numeric variables for most of the features.   
# 
# There are few features that shows some differences in distribution between train and test, for example: **2**, **8**, **12**, **16**, **37**, **72**, **84**, **100**, **103**, **104**, **123**, **144**, **155**, **181**, **202**, **203**, **204**, **229**, **241**, **264**, **288**.
# 
# 
# ## <a id='33'>Distribution of mean and std</a>  
# 
# Let's check the distribution of the mean values per row in the train and test set.

# In[ ]:


plt.figure(figsize=(16,6))
features = train_df.columns.values[2:302]
plt.title("Distribution of mean values per row in the train and test set")
sns.distplot(train_df[features].mean(axis=1),color="green", kde=True,bins=120, label='train')
sns.distplot(test_df[features].mean(axis=1),color="blue", kde=True,bins=120, label='test')
plt.legend()
plt.show()


# Mean values per row for test data are close to a normal distribution while the mean values per row for train data shows multiple peak values. Most of the values are  between +/- 0.1.
# 
# Let's check the distribution of the mean values per columns in the train and test set.

# In[ ]:


plt.figure(figsize=(16,6))
plt.title("Distribution of mean values per column in the train and test set")
sns.distplot(train_df[features].mean(axis=0),color="magenta",kde=True,bins=120, label='train')
sns.distplot(test_df[features].mean(axis=0),color="darkblue", kde=True,bins=120, label='test')
plt.legend()
plt.show()


# These are the values that we already observed earlier that are mostly centered around 0. We can see that train is actually showing a larger spread of these values, while test values have a smaller deviation and a distribution closer to a normal one.
# 
# Let's show the distribution of standard deviation of values per row for train and test datasets.

# In[ ]:


plt.figure(figsize=(16,6))
plt.title("Distribution of std values per row in the train and test set")
sns.distplot(train_df[features].std(axis=1),color="black", kde=True,bins=120, label='train')
sns.distplot(test_df[features].std(axis=1),color="red", kde=True,bins=120, label='test')
plt.legend();plt.show()


# The average standard deviation per rows is 1 and most of values are between 1 +/- 0.1.
# 
# Let's check the distribution of the standard deviation of values per columns in the train and test datasets.

# In[ ]:


plt.figure(figsize=(16,6))
plt.title("Distribution of std values per column in the train and test set")
sns.distplot(train_df[features].std(axis=0),color="blue",kde=True,bins=120, label='train')
sns.distplot(test_df[features].std(axis=0),color="green", kde=True,bins=120, label='test')
plt.legend(); plt.show()


# Standard deviation values per columns in train dataset are between 0.9 and 1.1 while in test dataset are much smaller, confined between 0.99 and 1.01.
# 
# Let's check now the distribution of the mean value per row in the train dataset, grouped by value of target.

# In[ ]:


t0 = train_df.loc[train_df['target'] == 0]
t1 = train_df.loc[train_df['target'] == 1]
plt.figure(figsize=(16,6))
plt.title("Distribution of mean values per row in the train set")
sns.distplot(t0[features].mean(axis=1),color="red", kde=True,bins=120, label='target = 0')
sns.distplot(t1[features].mean(axis=1),color="blue", kde=True,bins=120, label='target = 1')
plt.legend(); plt.show()


# Let's check now the distribution of the mean value per column in the train dataset, grouped by value of target.

# In[ ]:


plt.figure(figsize=(16,6))
plt.title("Distribution of mean values per column in the train set")
sns.distplot(t0[features].mean(axis=0),color="green", kde=True,bins=120, label='target = 0')
sns.distplot(t1[features].mean(axis=0),color="darkblue", kde=True,bins=120, label='target = 1')
plt.legend(); plt.show()


# ## <a id='34'>Distribution of min and max</a>  
# 
# Let's check the distribution of min per row in the train and test set.

# In[ ]:


plt.figure(figsize=(16,6))
features = train_df.columns.values[2:302]
plt.title("Distribution of min values per row in the train and test set")
sns.distplot(train_df[features].min(axis=1),color="red", kde=True,bins=120, label='train')
sns.distplot(test_df[features].min(axis=1),color="orange", kde=True,bins=120, label='test')
plt.legend()
plt.show()


# 
# A long queue to the lower values for both, extended as long as to -5.5 for test set, is observed.
# 
# Let's now show the distribution of min per column in the train and test set.

# In[ ]:


plt.figure(figsize=(16,6))
features = train_df.columns.values[2:302]
plt.title("Distribution of min values per column in the train and test set")
sns.distplot(train_df[features].min(axis=0),color="magenta", kde=True,bins=120, label='train')
sns.distplot(test_df[features].min(axis=0),color="darkblue", kde=True,bins=120, label='test')
plt.legend()
plt.show()


# The distribution of min values per columns (i.e. per variables) is quite notably different for train and test set.
# 
# 
# Let's check now the distribution of max values per rows for train and test set.

# In[ ]:


plt.figure(figsize=(16,6))
features = train_df.columns.values[2:302]
plt.title("Distribution of max values per row in the train and test set")
sns.distplot(train_df[features].max(axis=1),color="brown", kde=True,bins=120, label='train')
sns.distplot(test_df[features].max(axis=1),color="lightblue", kde=True,bins=120, label='test')
plt.legend()
plt.show()


# Both distribution shows a long queue toward larger values, with test extended more, up to 5.5., while train has values as large as 4.5.   
# 
# Let's show now the max distribution on columns for train and test set.

# In[ ]:


plt.figure(figsize=(16,6))
features = train_df.columns.values[2:302]
plt.title("Distribution of max values per column in the train and test set")
sns.distplot(train_df[features].max(axis=0),color="blue", kde=True,bins=120, label='train')
sns.distplot(test_df[features].max(axis=0),color="red", kde=True,bins=120, label='test')
plt.legend()
plt.show()


# The two distributions are neatly separated.  
# 
# 
# Let's  show now the distributions of min values per row in train set, separated on the values of target (0 and 1).

# In[ ]:


t0 = train_df.loc[train_df['target'] == 0]
t1 = train_df.loc[train_df['target'] == 1]
plt.figure(figsize=(16,6))
plt.title("Distribution of min values per row in the train set")
sns.distplot(t0[features].min(axis=1),color="red", kde=True,bins=120, label='target = 0')
sns.distplot(t1[features].min(axis=1),color="blue", kde=True,bins=120, label='target = 1')
plt.legend(); plt.show()


# We show here the distribution of min values per columns in train set.

# In[ ]:


plt.figure(figsize=(16,6))
plt.title("Distribution of min values per column in the train set")
sns.distplot(t0[features].min(axis=0),color="red", kde=True,bins=120, label='target = 0')
sns.distplot(t1[features].min(axis=0),color="blue", kde=True,bins=120, label='target = 1')
plt.legend(); plt.show()


# We can observe a relative good separation between the two distributions, with the values for **target = 0** with lower peaks and with a longer queue toward larger values (up to close to -1), while the mins for **target = 1** are extended only until -1.5.
# 
# Let's show now the distribution of max values per rown in the train set.

# In[ ]:


plt.figure(figsize=(16,6))
plt.title("Distribution of max values per row in the train set")
sns.distplot(t0[features].max(axis=1),color="red", kde=True,bins=120, label='target = 0')
sns.distplot(t1[features].max(axis=1),color="blue", kde=True,bins=120, label='target = 1')
plt.legend(); plt.show()


# Let's show also the distribution of max values per columns in the train set.

# In[ ]:


plt.figure(figsize=(16,6))
plt.title("Distribution of max values per column in the train set")
sns.distplot(t0[features].max(axis=0),color="red", kde=True,bins=120, label='target = 0')
sns.distplot(t1[features].max(axis=0),color="blue", kde=True,bins=120, label='target = 1')
plt.legend(); plt.show()


# We can observe a relative good separation between the two distributions.
# 

# ## <a id='35'>Distribution of skew and kurtosis</a>  
# 
# Let's see now what is the distribution of skew values per rows and columns.
# 
# Let's see first the distribution of skewness calculated per rows in train and test sets.
# 
# 

# In[ ]:


plt.figure(figsize=(16,6))
features = train_df.columns.values[2:302]
plt.title("Distribution of skew per row in the train and test set")
sns.distplot(train_df[features].skew(axis=1),color="red", kde=True,bins=120, label='train')
sns.distplot(test_df[features].skew(axis=1),color="orange", kde=True,bins=120, label='test')
plt.legend()
plt.show()


# Let's see first the distribution of skewness calculated per columns in train and test set.

# In[ ]:


plt.figure(figsize=(16,6))
features = train_df.columns.values[2:302]
plt.title("Distribution of skew per column in the train and test set")
sns.distplot(train_df[features].skew(axis=0),color="magenta", kde=True,bins=120, label='train')
sns.distplot(test_df[features].skew(axis=0),color="darkblue", kde=True,bins=120, label='test')
plt.legend()
plt.show()


# Let's see now what is the distribution of kurtosis values per rows and columns.
# 
# Let's see first the distribution of kurtosis calculated per rows in train and test sets.

# In[ ]:


plt.figure(figsize=(16,6))
features = train_df.columns.values[2:302]
plt.title("Distribution of kurtosis per row in the train and test set")
sns.distplot(train_df[features].kurtosis(axis=1),color="darkblue", kde=True,bins=120, label='train')
sns.distplot(test_df[features].kurtosis(axis=1),color="yellow", kde=True,bins=120, label='test')
plt.legend()
plt.show()


# 
# Let's see first the distribution of kurtosis calculated per columns in train and test sets.

# In[ ]:


plt.figure(figsize=(16,6))
features = train_df.columns.values[2:302]
plt.title("Distribution of kurtosis per column in the train and test set")
sns.distplot(train_df[features].kurtosis(axis=0),color="magenta", kde=True,bins=120, label='train')
sns.distplot(test_df[features].kurtosis(axis=0),color="green", kde=True,bins=120, label='test')
plt.legend()
plt.show()


# Let's see now the distribution of skewness on rows in train separated for values of target 0 and 1.

# In[ ]:


t0 = train_df.loc[train_df['target'] == 0]
t1 = train_df.loc[train_df['target'] == 1]
plt.figure(figsize=(16,6))
plt.title("Distribution of skew values per row in the train set")
sns.distplot(t0[features].skew(axis=1),color="red", kde=True,bins=120, label='target = 0')
sns.distplot(t1[features].skew(axis=1),color="blue", kde=True,bins=120, label='target = 1')
plt.legend(); plt.show()


# Let's see now the distribution of skewness on columns in train separated for values of target 0 and 1.

# In[ ]:


t0 = train_df.loc[train_df['target'] == 0]
t1 = train_df.loc[train_df['target'] == 1]
plt.figure(figsize=(16,6))
plt.title("Distribution of skew values per column in the train set")
sns.distplot(t0[features].skew(axis=0),color="red", kde=True,bins=120, label='target = 0')
sns.distplot(t1[features].skew(axis=0),color="blue", kde=True,bins=120, label='target = 1')
plt.legend(); plt.show()


# Let's see now the distribution of kurtosis on rows in train separated for values of target 0 and 1.

# In[ ]:


t0 = train_df.loc[train_df['target'] == 0]
t1 = train_df.loc[train_df['target'] == 1]
plt.figure(figsize=(16,6))
plt.title("Distribution of kurtosis values per row in the train set")
sns.distplot(t0[features].kurtosis(axis=1),color="red", kde=True,bins=120, label='target = 0')
sns.distplot(t1[features].kurtosis(axis=1),color="blue", kde=True,bins=120, label='target = 1')
plt.legend(); plt.show()


# Let's see now the distribution of kurtosis on columns in train separated for values of target 0 and 1.

# In[ ]:


t0 = train_df.loc[train_df['target'] == 0]
t1 = train_df.loc[train_df['target'] == 1]
plt.figure(figsize=(16,6))
plt.title("Distribution of kurtosis values per column in the train set")
sns.distplot(t0[features].kurtosis(axis=0),color="red", kde=True,bins=120, label='target = 0')
sns.distplot(t1[features].kurtosis(axis=0),color="blue", kde=True,bins=120, label='target = 1')
plt.legend(); plt.show()


# ## <a id='36'>Features correlation</a>  
# 
# We calculate now the correlations between the features in train set.  
# The following table shows the first 10 the least correlated features.

# In[ ]:


get_ipython().run_cell_magic('time', '', 'correlations = train_df[features].corr().abs().unstack().sort_values(kind="quicksort").reset_index()\ncorrelations = correlations[correlations[\'level_0\'] != correlations[\'level_1\']]\ncorrelations.head(10)')


# Let's look to the top most correlated features, besides the same feature pairs.

# In[ ]:


correlations.tail(10)


# Let's see also the least correlated features.

# In[ ]:


correlations.head(10)


# The correlation between the features is  small (in the range of `no correlation` for the best correlated features. 
# 
# 

# # <a id='4'>Feature engineering</a>  
# 
# 
# Let's calculate for starting few aggregated values for the existing features.

# ## <a id='41'>Add features by aggregation</a> 
# 
# We define several features obtained by applying aggregation functions.

# In[ ]:


get_ipython().run_cell_magic('time', '', "features = [c for c in train_df.columns if c not in ['id', 'target']]\nfor df in [test_df, train_df]:\n    df['sum'] = df[features].sum(axis=1)  \n    df['min'] = df[features].min(axis=1)\n    df['max'] = df[features].max(axis=1)\n    df['mean'] = df[features].mean(axis=1)\n    df['std'] = df[features].std(axis=1)\n    df['skew'] = df[features].skew(axis=1)\n    df['kurt'] = df[features].kurtosis(axis=1)\n    df['med'] = df[features].median(axis=1)")


# Let's check the new created features.

# In[ ]:


train_df[train_df.columns[302:]].head()


# In[ ]:


test_df[test_df.columns[301:]].head()


# In[ ]:


def plot_new_feature_distribution(df1, df2, label1, label2, features):
    i = 0
    sns.set_style('whitegrid')
    plt.figure()
    fig, ax = plt.subplots(2,4,figsize=(18,8))

    for feature in features:
        i += 1
        plt.subplot(2,4,i)
        sns.kdeplot(df1[feature], bw=0.5,label=label1)
        sns.kdeplot(df2[feature], bw=0.5,label=label2)
        plt.xlabel(feature, fontsize=11)
        locs, labels = plt.xticks()
        plt.tick_params(axis='x', which='major', labelsize=8)
        plt.tick_params(axis='y', which='major', labelsize=8)
    plt.show();


# Let's check the distribution of these new, engineered features.  
# 
# We plot first the distribution of new features, grouped by value of corresponding `target` values.

# In[ ]:


t0 = train_df.loc[train_df['target'] == 0]
t1 = train_df.loc[train_df['target'] == 1]
features = train_df.columns.values[302:]
plot_new_feature_distribution(t0, t1, 'target: 0', 'target: 1', features)


# Let's show the distribution of new features values for train and test.

# In[ ]:


features = train_df.columns.values[302:]
plot_new_feature_distribution(train_df, test_df, 'train', 'test', features)


# Let's check how many features we have now.

# In[ ]:


print('Train and test columns: {} {}'.format(len(train_df.columns), len(test_df.columns)))


# ## <a id='42'>Add noise</a>  
# 
# We will add more data in the training set by injecting noise in the existing training data. This will account for a data multiplication technique and as well as a regularization technique, aiming to reduce overfitting on the existing training data.

# In[ ]:


def apply_noise(data, noise_level):
    idxt = data[['id', 'target']]
    features = data.columns.values[2:]
    appended_data = []
    for feature in features:
        signal = data[feature]
        noise_factor = (np.abs(signal)).mean() * noise_level
        noise =  np.random.normal(0, noise_level, signal.shape)
        jittered = signal + noise
        appended_data.append(pd.DataFrame(jittered))
    appended_data = pd.concat(appended_data, axis=1)
    data_jittered = pd.concat([idxt, pd.DataFrame(appended_data)], axis=1)
    return data_jittered


# In[ ]:


noise_train_df = []
for i in tqdm_notebook(range(0,2)):
    t = apply_noise(train_df, noise_level = i * 0.025)
    noise_train_df.append(t)
noise_train_df = pd.concat(noise_train_df, axis = 0)


# In[ ]:


print("Shape train with additional rows with noise added:",noise_train_df.shape)
train_df = noise_train_df


# # <a id='5'>Model</a>  
# 
# From the train columns list, we drop the ID and target to form the features list.

# In[ ]:


features = [c for c in train_df.columns if c not in ['id', 'target']]
#using https://www.kaggle.com/alexandregeorges/glmnet-train-rfe-boruta-cv ?
#features = ['33', '65', '217', '91', '199', '69', '82', '117', '73', '295',
#           '130', '108', '258', '18', '189', '194', '43', '145', '80','24', 
#           '56', '214', '268', 'max', 'min', 'std', 'mean', 'skew','kurt' ]
target = train_df['target']


# In[ ]:


len(features)


# We define the hyperparameters for the model.

# In[ ]:


param = {
    'bagging_freq': 5,
    'bagging_fraction': 0.83,
    'boost_from_average':'false',
    'boost': 'gbdt',
    'feature_fraction': 0.81,
    'learning_rate': 0.005,
    'max_depth': -1,  
    'metric':'auc',
    'min_data_in_leaf': 90,
    'min_sum_hessian_in_leaf': 10.0,
    'num_leaves': 11,
    'num_threads': 8,
    'tree_learner': 'serial',
    'objective': 'binary', 
    'verbosity': 1
}


# We run the model.

# In[ ]:


folds = StratifiedKFold(n_splits=9, shuffle=True, random_state=4422)
oof = np.zeros(len(train_df))
predictions = np.zeros(len(test_df))
feature_importance_df = pd.DataFrame()

for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_df.values, target.values)):
    print("Fold {}".format(fold_))
    trn_data = lgb.Dataset(train_df.iloc[trn_idx][features], label=target.iloc[trn_idx])
    val_data = lgb.Dataset(train_df.iloc[val_idx][features], label=target.iloc[val_idx])

    num_round = 100000
    clf = lgb.train(param, trn_data, 
                    num_round, 
                    valid_sets = [trn_data, val_data], 
                    verbose_eval=400, 
                    early_stopping_rounds = 400)
    oof[val_idx] = clf.predict(train_df.iloc[val_idx][features], num_iteration=clf.best_iteration)
    
    fold_importance_df = pd.DataFrame()
    fold_importance_df["Feature"] = features
    fold_importance_df["importance"] = clf.feature_importance()
    fold_importance_df["fold"] = fold_ + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
    
    predictions += clf.predict(test_df[features], num_iteration=clf.best_iteration) / folds.n_splits

print("CV score: {:<8.5f}".format(roc_auc_score(target, oof)))


# Let's check the feature importance.

# In[ ]:


def plot_feature_importance():
    cols = (feature_importance_df[["Feature", "importance"]]
            .groupby("Feature")
            .mean()
            .sort_values(by="importance", ascending=False)[:50].index)
    best_features = feature_importance_df.loc[feature_importance_df.Feature.isin(cols)]
    plt.figure(figsize=(12,10))
    sns.barplot(x="importance", y="Feature", data=best_features.sort_values(by="importance",ascending=False))
    plt.title('Features importance (averaged/folds)')
    plt.tight_layout()
    plt.savefig('FI.png')
plot_feature_importance()


# # <a id='6'>Submission</a>  
# 
# We submit the solution.

# In[ ]:


sub_df = pd.DataFrame({"id":test_df["id"].values})
sub_df["target"] = predictions
sub_df.to_csv("submission.csv", index=False)


# # <a id='7'>References</a>    
# 
# [1] https://www.kaggle.com/gpreda/elo-world-high-score-without-blending  
# [2] https://www.kaggle.com/chocozzz/santander-lightgbm-baseline-lb-0-897   
# [3] https://www.kaggle.com/gpreda/santander-eda-and-prediction   
# 
# 
