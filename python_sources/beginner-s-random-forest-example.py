#!/usr/bin/env python
# coding: utf-8

# # Beginner's Random Forests example
# 
# This is a very simple Random Forests example meant for beginners. This is not meant to achieve a high score, merely a starting point on which to start, without complicated techniques.
# 
# It's recommended that you finish the Kaggle Learn courses (introduction and intermediate machine learning).

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# ## Files
# 
# As can be seen the training data contains two files, `train_transaction.csv` and `train_identity.csv`. These two tables are related to each other via the column `TransactionID`. 

# In[ ]:


train_transaction_df = pd.read_csv('/kaggle/input/train_transaction.csv')
train_identity_df = pd.read_csv('/kaggle/input/train_identity.csv')


# In[ ]:


train_transaction_df.head()


# In[ ]:


train_identity_df.head()


# In order to use these files for training, we'll need to do what's sometimes called denormalising the data. We can do this by doing a left join on both tables using the DataFrame's `merge()` method.

# In[ ]:


df_train = train_transaction_df.merge(train_identity_df, on='TransactionID', how='left')


# Let's do a sanity check whenever we do something like this. Make sure the shape contains the same number of rows and the combined columns:

# In[ ]:


df_train.shape


# In[ ]:


df_train.head().transpose()


# Looks like that worked! But we're now using a lot of RAM (look at the sidebar of your kernel). My kernel currently says I'm at 5 gigabytes, and we haven't even read the test set yet!
# 
# While cleaning the data, it's possible we may need to make copies of (some sections) of the data, so this is obviously not ideal.

# In[ ]:


df_train.memory_usage().sum()


# ## Memory reduction
# 
# As discussed in [my other kernel](https://www.kaggle.com/yoongkang/beginner-memory-reduction-techniques), parsing the training dataset with default settings could take up to 2GBs of memory unnecessarily. With a few techniques (also discussed in the linked kernel) we can cut this down by about a gigabyte. 
# 
# In this kernel, we'll use similar techniques, if you want to see my reasoning for this please refer to the other kernel.
# 
# First we need to determine which numeric columns we have so that we can downcast them (cast from float64 to another type that requires less memory).

# In[ ]:


# Get the categorical and numeric columns
cat_cols = [
    'ProductCD', 'card1', 'card2', 'card3', 'card4', 'card5', 'card6',
    'addr1', 'addr2', 'P_emaildomain', 'R_emaildomain', 'DeviceType', 'DeviceInfo',
] + [f'M{n}' for n in range(1, 10)] + [f'id_{n}' for n in range(12, 39)]
num_cols = list(set(df_train.columns) - set(cat_cols))


# As described in the other kernel, some of these columns are `float64` by default due to the presence of some `NaN` values.
# 
# The fact that they are `NaN` might be meaningful in this context, so completely replacing them (this is called imputation), doesn't sound like a great idea. However, we can add a boolean column to mark that the column has been replaced, and hopefully the training algorithm is smart enough to take care of it. There's no guarantee that it will, though! So you should always challenge your assumptions (e.g. maybe just dropping the columns could give you similar results, and train faster).
# 
# Also bear in mind we'll need to use the same values we're using to impute on the training set on the test set. 

# In[ ]:


a = df_train[num_cols].isnull().any()
train_null_num_cols = a[a].index


# In[ ]:


nas = {}
for n in train_null_num_cols:
    df_train[f'{n}_isna'] = df_train[n].isnull()
    median = df_train[n].median()
    df_train[n].fillna(median, inplace=True)
    nas[n] = median


# Now that we've removed all the `NaN` values, we can downcast the columns to the lowest precision.
# 
# First we'll need to know which columns are integers, though! The following snippet does just that.

# In[ ]:


integer_cols = []
for c in num_cols:
    try:
        if df_train[c].fillna(-1.0).apply(float.is_integer).all():
            integer_cols += [c]
    except Exception as e:
        print("error: ", c, e)


# There will be some errors printed, but that's normal because some numeric columns are already integers. I'm too lazy to fix that right now.
# 
# Let's look at some stats.

# In[ ]:


stats = df_train[integer_cols].describe().transpose()
stats


# So we can see here that there are some very small ranges there -- not all of them will need to be `float64`. Let's downcast them.

# In[ ]:


int8columns = stats[stats['max'] < 256].index
print(int8columns.shape)
print(int8columns)
int16columns = stats[(stats['max'] >= 256) & (stats['max'] <= 32767)].index
print(int16columns.shape)
print(int16columns)


# In[ ]:


for c in int8columns:
    df_train[c] = df_train[c].astype('int8')
    
for c in int16columns:
    df_train[c] = df_train[c].astype('int16')


# In[ ]:


df_train.memory_usage().sum()


# Looks like we shaved a whole gig.
# 
# We're not done yet, we need to make sure we do the same thing on the test set. Let's read it in now, merge the tables, and impute it the same way.

# In[ ]:


test_transaction_df = pd.read_csv('/kaggle/input/test_transaction.csv')
test_identity_df = pd.read_csv('/kaggle/input/test_identity.csv')
df_test = test_transaction_df.merge(test_identity_df, on='TransactionID', how='left')


# We added some columns in our training set and replaced missing values with the medians. We need to add those same columns, and also add the median from the training set for those missing values (the same ones).

# In[ ]:


for k, v in nas.items():
    df_test[f'{k}_isna'] = df_test[k].isnull()
    df_test[k].fillna(v, inplace=True)


# Unfortunately, we're not done yet. We might have some missing values on other numeric columns we didn't anticipate. 
# 
# In practice, we don't always have a "test set". The test set might be new observations that come in the future, could be a list of observations or a single one, so we can't really use statistics from the test set to impute the missing values. So we need to use values from the training set.
# 
# We'll use the median as well.

# In[ ]:


test_num_cols = list(set(num_cols) - set(['isFraud']))
a = df_test[test_num_cols].isnull().any()
test_null_num_cols = a[a].index


# In[ ]:


for n in test_null_num_cols:
    df_test[n].fillna(df_train[n].median(), inplace=True)  # use the training set's median!


# Now, we can downcast numeric columns in the same way

# In[ ]:


# copied from above cells

integer_cols = []
for c in test_num_cols:
    try:
        if df_test[c].fillna(-1.0).apply(float.is_integer).all():
            integer_cols += [c]
    except Exception as e:
        print("error: ", c, e)
stats = df_test[integer_cols].describe().transpose()
int8columns = stats[stats['max'] < 256].index
int16columns = stats[(stats['max'] >= 256) & (stats['max'] <= 32767)].index
for c in int8columns:
    df_test[c] = df_test[c].astype('int8')
    
for c in int16columns:
    df_test[c] = df_test[c].astype('int16')


# ## Categorical values
# 
# Okay, we've reduced memory, but we still need to deal with categorical values. As machine learning algorithms don't understand things like strings, we need to convert them into numbers. This is called encoding.
# 
# We could use either label encoding, which replaces each category into a numerical representation, or use one hot encoding which creates a separate column for each category. In general, one hot encoding performs better. However, in this case we have columns with very high cardinality -- and since we have a large dataset, it's probably more practical to use label encoding which we'll do.
# 
# Two things we need to deal with for label encoding are missing values and unknown values. Missing values means the data is simply not there, whereas unknown values are values in the test set that we don't have in the training set.
# 
# For missing values, we'll just replace them with a label, e.g. the string `"missing"`. That's pretty straightforward.
# 
# For unknown values, that requires a bit more thought. The main question is whether or not we have all the categories a priori. If we know all the possible categories beforehand (i.e. fixed categories like gender, state, postcodes) then we can go ahead and devise a mapping beforehand for all possible values. However, sometimes categories only come in the future, like mobile phone models. In the latter case, we have no way of knowing all the possible future values, and thus we can't map them -- so we'll need another strategy, i.e. replace them with a different label like the string `"unknown"`. We'll be doing that.

# First, we'll replace missing values with the string `"missing"` (we actually don't need to do this since pandas does it automatically, but I like to give it an explicit label, makes it easier to see).

# In[ ]:


for c in cat_cols:
    df_train[c] = df_train[c].fillna("missing")
    
for c in cat_cols:   
    df_test[c] = df_test[c].fillna("missing")


# Next we'll convert the columns in the training set to categorical.

# In[ ]:


cats = {}
for c in cat_cols:
    df_train[c] = df_train[c].astype("category")
    df_train[c].cat.add_categories('unknown', inplace=True)
    cats[c] = df_train[c].cat.categories


# Then we'll convert the test set.

# In[ ]:


for k, v in cats.items():
    df_test[k][~df_test[k].isin(v)] = 'unknown'


# In[ ]:


from pandas.api.types import CategoricalDtype

for k, v in cats.items():
    new_dtype = CategoricalDtype(categories=v, ordered=True)
    df_test[k] = df_test[k].astype(new_dtype)


# In[ ]:


for c in cat_cols:
    df_train[c] = df_train[c].cat.codes
    df_test[c] = df_test[c].cat.codes
    


# Now we're more or less done with the minimum preprocessing required. Let's save our progress to a feather file, so that we don't have to go through it again!

# In[ ]:


df_train.to_feather('df_train')


# In[ ]:


df_test.to_feather('df_test')


# ## Validation set
# 
# Now we can start training our model. But how do we know if a model is good or not? We commonly use something called a validation set, that is separate to the test set. The reason we have a holdout set is that we use the validation set to choose our model (even if we don't use it for training), otherwise our model will overfit. If you're unfamiliar with this, I suggest reading on overfitting and underfitting.
# 
# The data description seems to indicate that the data is time ordered, so we don't really want a random split. So let's hold out a portion of the bottom rows to use as our validation set, and the rest as our training set.

# In[ ]:


idx = int(len(df_train) * 0.8)
training_set, validation_set = df_train[:idx], df_train[idx:]


# In[ ]:


y_train = training_set['isFraud']
X_train = training_set.drop('isFraud', axis=1)
y_valid = validation_set['isFraud']
X_valid = validation_set.drop('isFraud', axis=1)


# In[ ]:


print(X_train.shape, y_train.shape)


# In[ ]:


print(X_valid.shape, y_valid.shape)


# ## Training the model
# 
# Now we can finally train a model. You can iterate on this part.

# In[ ]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_auc_score


# Using the whole training set is too time-consuming for quick iteration, so we can use a sample. 
# 
# We could use a random sample, but since this is time-ordered, I'm guessing the more recent rows would give us better predictive value. So let's just grab the bottom rows.

# In[ ]:


training_sample = training_set[-100000:]
y_train_sample = training_sample['isFraud']
X_train_sample = training_sample.drop('isFraud', axis=1)


# In[ ]:


model = RandomForestRegressor(
    n_estimators=400, max_features=0.3,
    min_samples_leaf=20, n_jobs=-1, verbose=1)


# In[ ]:


model.fit(X_train_sample, y_train_sample)


# In[ ]:


preds_valid = model.predict(X_valid)


# In[ ]:


roc_auc_score(y_valid, preds_valid)


# ## Submission

# Now that we have a decent model, we can actually train on the whole dataset, including the validation set.

# In[ ]:


model = RandomForestRegressor(
    n_estimators=400, max_features=0.3,
    min_samples_leaf=20, n_jobs=-1, verbose=1)


# In[ ]:


y = df_train['isFraud']
X = df_train.drop('isFraud', axis=1)


# In[ ]:


model.fit(X, y)


# In[ ]:


y_preds = model.predict(df_test)


# In[ ]:


submission = pd.read_csv('/kaggle/input/sample_submission.csv')
submission['isFraud'] = y_preds
submission.to_csv('submission.csv', index=False)


# In[ ]:




