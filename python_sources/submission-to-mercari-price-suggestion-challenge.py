#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#required libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import csv
import seaborn as sns
import matplotlib.pyplot as plt
import re
# Evalaluation
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer
import scipy
import lightgbm as lgb
from sklearn.linear_model import Ridge
import time
import gc
from scipy.sparse import csr_matrix, hstack


# **It's always a good practice to specify the dtypes**

# In[ ]:


types_dict_train = {'train_id': 'int64',
             'item_condition_id': 'int64',
             'price': 'float64',
             'shipping': 'int64'}
types_dict_test = {'test_id': 'int64',
             'item_condition_id': 'int64',
             'shipping': 'int64'}


# In[ ]:


#read the datasets
train = pd.read_csv('../input/train.tsv', sep='\t', encoding='utf-8', dtype=types_dict_train)
test=pd.read_csv('../input/test.tsv', sep='\t', encoding='utf-8', dtype=types_dict_train)


# In[ ]:


#shape of the datasets
print(train.shape)
print(test.shape)


# In[ ]:


#check the datatype of training dataset
print(train.dtypes)


# **The training dataset has two types of data-
# Strings:name,category_name,brand_name,item_description
# Numeric:train_id,item_condition_id,shipping,price**

# **Here all categorical variables are stored as 'object' data type.
# So I am going to convert them into Pandas Category**

# In[ ]:


train.category_name = train.category_name.astype('category')
train.item_description = train.item_description.astype('category')

train.name = train.name.astype('category')
train.brand_name = train.brand_name.astype('category')

test.category_name = test.category_name.astype('category')
test.item_description = test.item_description.astype('category')

test.name = test.name.astype('category')
test.brand_name = test.brand_name.astype('category')


# In[ ]:


#let's find out the count of distinct values in each column using Pandas
print("count of distinct values in train dataset:")
train.apply(lambda x: x.nunique())


# In[ ]:


#let's find out the count of distinct values in each column using Pandas
print("count of distinct values in test dataset:")
test.apply(lambda x: x.nunique())


# In[ ]:


#let's find out the missing values
train.isnull().sum(),train.isnull().sum()/train.shape[0]


# In[ ]:


#let's find out the missing values
test.isnull().sum(),test.isnull().sum()/test.shape[0]


# In[ ]:


#peek of the training dataset
train.head()


# In[ ]:


#peek of the test dataset
test.head()


# In[ ]:


#changing train_id/test_id as id column
train = train.rename(columns = {'train_id':'id'})
test = test.rename(columns = {'test_id':'id'})


# In[ ]:


train.head()


# In[ ]:


train['is_train'] = 1
test['is_train'] = 0


# In[ ]:


test.head()


# In[ ]:


#seperate the target variable and combine the dataset
train_test_combine = pd.concat([train.drop(['price'],axis =1),test],axis = 0)
train_test_combine.category_name = train_test_combine.category_name.astype('category')
train_test_combine.item_description = train_test_combine.item_description.astype('category')
train_test_combine.name = train_test_combine.name.astype('category')
train_test_combine.brand_name = train_test_combine.brand_name.astype('category')


# In[ ]:


#droping item description since I don't know about NLP and deep learning
train_test_combine = train_test_combine.drop(['item_description'],axis = 1)


# In[ ]:


#get numeric from categorical variables
train_test_combine.name = train_test_combine.name.cat.codes
train_test_combine.category_name = train_test_combine.category_name.cat.codes
train_test_combine.brand_name = train_test_combine.brand_name.cat.codes


# In[ ]:


train_test_combine.head()


# In[ ]:


train_test_combine.dtypes


# In[ ]:


df_test = train_test_combine.loc[train_test_combine['is_train']==0]
df_train = train_test_combine.loc[train_test_combine['is_train']==1]


# In[ ]:


df_test = df_test.drop(['is_train'],axis=1)


# In[ ]:


df_train = df_train.drop(['is_train'],axis=1)


# In[ ]:


#save the target variable from train dataframe
df_train['price'] = train.price


# In[ ]:


#taking the log of price
df_train['price'] = df_train['price'].apply(lambda x: np.log(x) if x>0 else x)


# In[ ]:


x_train,y_train = df_train.drop(['price'],axis =1),df_train.price


# In[ ]:


#modeling the problem
randomfr = RandomForestRegressor(n_jobs=-1,min_samples_leaf=3,n_estimators=200)
randomfr.fit(x_train, y_train)
randomfr.score(x_train,y_train)


# In[ ]:


preds = randomfr.predict(df_test)


# In[ ]:


preds = pd.Series(np.exp(preds))


# In[ ]:


submit = pd.concat([df_test.id,preds],axis=1)
submit.columns = ['test_id','price']
submit.to_csv("./firstoutput.csv", index=False)


# **From the RandomForest I got a score of 0.53854**
# **Let's try to improve the score by applying light GBM**

# In[ ]:


import lightgbm as lgb
from sklearn.linear_model import Ridge, LogisticRegression
params = {
    'learning_rate': 0.75,
    'application': 'regression',
    'max_depth': 3,
    'num_leaves': 100,
    'verbosity': -1,
    'metric': 'RMSE',
}
train_X, valid_X, train_y, valid_y = train_test_split(x_train, y_train, test_size = 0.1, random_state = 144) 
d_train = lgb.Dataset(train_X, label=train_y, max_bin=8192)
d_valid = lgb.Dataset(valid_X, label=valid_y, max_bin=8192)
watchlist = [d_train, d_valid]

model = lgb.train(params, train_set=d_train, num_boost_round=2200, valid_sets=watchlist, early_stopping_rounds=50, verbose_eval=100) 
preds = model.predict(df_test)

model = Ridge(solver = "lsqr", fit_intercept=False)

print("Fitting Model")
model.fit(x_train, y_train)

#preds += model.predict(df_test)
#preds /= 2
#preds = np.expm1(preds)
preds = pd.Series(np.exp(preds))
submit = pd.concat([df_test.id,preds],axis=1)
submit.columns = ['test_id','price']
submit.to_csv("./submissionusinglgb.csv", index=False)


# **The Light GBM algorithm improves my score from 0.53854 to 0.53284 which improves my rank 6 places up**

# **For improving the model I need to work on the categorical variables more, so let's try new approach**

# """Reference:https://www.kaggle.com/tunguz/more-effective-ridge-lgbm-script-lb-0-44944"""

# In[ ]:


NUM_BRANDS = 4000
NUM_CATEGORIES = 1000
NAME_MIN_DF = 10
MAX_FEATURES_ITEM_DESCRIPTION = 50000


# In[ ]:


def handle_missing_inplace(dataset):
    dataset['category_name'].fillna(value='missing', inplace=True)
    dataset['brand_name'].fillna(value='missing', inplace=True)
    dataset['item_description'].fillna(value='missing', inplace=True)


# In[ ]:


#dealing with missing values in categorical variables
def cutting(dataset):
    pop_brand = dataset['brand_name'].value_counts().loc[lambda x: x.index != 'missing'].index[:NUM_BRANDS]
    dataset.loc[~dataset['brand_name'].isin(pop_brand), 'brand_name'] = 'missing'
    pop_category = dataset['category_name'].value_counts().loc[lambda x: x.index != 'missing'].index[:NUM_BRANDS]
    dataset.loc[~dataset['category_name'].isin(pop_category), 'category_name'] = 'missing'

def to_categorical(dataset):
    dataset['category_name'] = dataset['category_name'].astype('category')
    dataset['brand_name'] = dataset['brand_name'].astype('category')
    dataset['item_condition_id'] = dataset['item_condition_id'].astype('category')


# In[ ]:


def main():
    start_time = time.time()

    train = pd.read_table('../input/train.tsv', engine='c')
    test = pd.read_table('../input/test.tsv', engine='c')
    print('[{}] Finished to load data'.format(time.time() - start_time))
    print('Train shape: ', train.shape)
    print('Test shape: ', test.shape)

    nrow_train = train.shape[0]
    y = np.log1p(train["price"])
    merge: pd.DataFrame = pd.concat([train, test])
    submission: pd.DataFrame = test[['test_id']]

    del train
    del test
    gc.collect()

    handle_missing_inplace(merge)
    print('[{}] Finished to handle missing'.format(time.time() - start_time))

    cutting(merge)
    print('[{}] Finished to cut'.format(time.time() - start_time))

    to_categorical(merge)
    print('[{}] Finished to convert categorical'.format(time.time() - start_time))

    cv = CountVectorizer(min_df=NAME_MIN_DF)
    X_name = cv.fit_transform(merge['name'])
    print('[{}] Finished count vectorize `name`'.format(time.time() - start_time))

    cv = CountVectorizer()
    X_category = cv.fit_transform(merge['category_name'])
    print('[{}] Finished count vectorize `category_name`'.format(time.time() - start_time))

    tv = TfidfVectorizer(max_features=MAX_FEATURES_ITEM_DESCRIPTION,
                         ngram_range=(1, 3),
                         stop_words='english')
    X_description = tv.fit_transform(merge['item_description'])
    print('[{}] Finished TFIDF vectorize `item_description`'.format(time.time() - start_time))

    lb = LabelBinarizer(sparse_output=True)
    X_brand = lb.fit_transform(merge['brand_name'])
    print('[{}] Finished label binarize `brand_name`'.format(time.time() - start_time))

    X_dummies = csr_matrix(pd.get_dummies(merge[['item_condition_id', 'shipping']],
                                          sparse=True).values)
    print('[{}] Finished to get dummies on `item_condition_id` and `shipping`'.format(time.time() - start_time))

    sparse_merge = hstack((X_dummies, X_description, X_brand, X_category, X_name)).tocsr()
    print('[{}] Finished to create sparse merge'.format(time.time() - start_time))

    X = sparse_merge[:nrow_train]
    X_test = sparse_merge[nrow_train:]
    
    d_train = lgb.Dataset(X, label=y, max_bin=8192)
    
    params = {
        'learning_rate': 0.75,
        'application': 'regression',
        'max_depth': 3,
        'num_leaves': 100,
        'verbosity': -1,
        'metric': 'RMSE',
    }
    
    model = lgb.train(params, train_set=d_train, num_boost_round=3200,      verbose_eval=100) 
    preds = 0.6*model.predict(X_test)

    model = Ridge(solver="sag", fit_intercept=True, random_state=205)
    model.fit(X, y)
    print('[{}] Finished to train ridge'.format(time.time() - start_time))
    preds += 0.4*model.predict(X=X_test)
    print('[{}] Finished to predict ridge'.format(time.time() - start_time))

    submission['price'] = np.expm1(preds)
    submission.to_csv("Thirdsubmission_lgbm_ridge.csv", index=False)

if __name__ == '__main__':
    main()

