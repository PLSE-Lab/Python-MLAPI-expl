#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os ; os.environ['OMP_NUM_THREADS'] = '4'


# In[ ]:


import pandas as pd
import operator
import matplotlib.pyplot as plt
import numpy as np 
import nltk
import time


# In[ ]:


start = time.time()
def print_time(start):
    time_now = time.time() - start 
    minutes = int(time_now / 60)
    seconds = int(time_now % 60)
    if seconds < 10:
        print('Elapsed time was %d:0%d.' % (minutes, seconds))
    else:
        print('Elapsed time was %d:%d.' % (minutes, seconds))


# In[ ]:



df = pd.read_csv('../input/train.tsv', sep='\t')
df_sub = pd.read_csv('../input/test.tsv', sep='\t')

submission = pd.DataFrame()
submission['test_id'] = df_sub.test_id.copy()

y_target = list(df.price)


# ## Impute Missing Values

# In[ ]:


def null_percentage(column):
    df_name = column.name
    nans = np.count_nonzero(column.isnull().values)
    total = column.size
    frac = nans / total
    perc = int(frac * 100)
    print('%d%% or %d missing from %s column.' % 
          (perc, nans, df_name))

def check_null(df, columns):
    for col in columns:
        null_percentage(df[col])
        
check_null(df, df.columns)


# In[ ]:


def merc_imputer(df_temp):
    df_temp.brand_name = df_temp.brand_name.replace(np.nan, 'no_brand')
    df_temp.category_name = df_temp.category_name.replace(np.nan, 'uncategorized/uncategorized')
    df_temp.item_description = df_temp.item_description.replace(np.nan, 'No description yet')
    df_temp.item_description = df_temp.item_description.replace('No description yet', 'no_description')
    return df_temp

df = merc_imputer(df)
df_sub = merc_imputer(df_sub)


# In[ ]:


print('Training Data')
check_null(df, df.columns)
print('Submission Data')
check_null(df_sub, df_sub.columns)


# # EDA 

# ## Shipping

# In[ ]:


df.shipping.value_counts()


# In[ ]:


print('%.1f%% of items have free shipping.' % ((663100 / len(df))*100))


# Free shipping items should be priced higher because shipping is included in the price. 

# ### Price

# In[ ]:


df.columns


# In[ ]:


print('$1 items: ' + str(df.price[df.price == 1].count()))
print('$2 items: ' + str(df.price[df.price == 2].count()))
print('$3 items: ' + str(df.price[df.price == 3].count()))


# There is a minimum price of $3.

# In[ ]:


plt.figure('Training Price Dist', figsize=(30,10))
plt.title('Price Distribution for Training - 3 Standard Deviations', fontsize=32)
plt.hist(df.price.values, bins=145, normed=False, 
         range=[0, (np.mean(df.price.values) + 3 * np.std(df.price.values))])
plt.axvline(df.price.values.mean(), color='b', linestyle='dashed', linewidth=2)
plt.xticks(fontsize=24)
plt.yticks(fontsize=26)
plt.show()

print('Line indicates mean price.')


# Most prices are on the lower end of the spectrum, and items priced above 145 are outliers that make up less that 0.3% of the data. Are there free items? 

# In[ ]:


print('Free items: %d, representing %.5f%% of all items.' % 
      (df.price[df.price == 0].count(), 
        (df.price[df.price == 0].count() / df.shape[0])))


# What does free even mean here? 

# In[ ]:


print('Free items where seller pays shipping: %d.' % 
      df.price[operator.and_(df.price == 0, df.shipping == 1)].count())


# This is a tiny outlier. And it seems like some items the sellers actually paid to give away. I'd like to see how many items are listed for a low price but the seller is actually making money off shipping to avoid fees, a common eBay practice. Unfortunately, without data about the actual shipping price, we can't extrapolate any insights here. My approach would be to look at items that are priced lower than average yet have higher than average shipping prices for their name and descriptions. 

# In[ ]:


print('No description:', str(df.item_description[df.item_description == 'no_description'].count()))
print('Uncategorized:',str(df.category_name[df.category_name == 'uncategorized/uncategorized'].count()))


#  Many items lack a description, but few lack a category. 

# ### Category Name

# In[ ]:


cat_counts = np.sort(df.category_name.value_counts())
print(str(len(cat_counts)) + ' categories total.')
print(str(df.shape[0]) + ' records total.')
print('Category frequency percentiles, marked by lines: \n25%%: %d, 50%%: %d, 75%%: %d, 95%%: %d, 97.5%%: %d.' % 
     (cat_counts[int(len(cat_counts)*0.25)], 
      cat_counts[int(len(cat_counts)*0.5)],
      cat_counts[int(len(cat_counts)*0.75)],
      cat_counts[int(len(cat_counts)*0.9)],
      cat_counts[int(len(cat_counts)*0.95)]))

title = 'Category Quantity ECDF Without Top 15 Outliers'
plt.figure(title, figsize=(30,10))
plt.title(title, fontsize=32)
x = np.sort(df.category_name.value_counts())
x = x[0:-15]
y = np.arange(1, len(x) + 1) / len(x)
plt.plot(x, y, marker='.', linestyle='none')
plt.xticks(fontsize=24)
plt.yticks(fontsize=26)
plt.axvline(x=x[int(len(x)*0.25)], linewidth=1, color='b')
plt.axvline(x=x[int(len(x)*0.5)], linewidth=1, color='b')
plt.axvline(x=x[int(len(x)*0.75)], linewidth=1, color='b')
plt.axvline(x=x[int(len(x)*0.95)], linewidth=1, color='b')
plt.axvline(x=x[int(len(x)*0.975)], linewidth=1, color='b')
plt.show()


# In[ ]:


print('The top 75%% of categories represent %.1f%% of the dataset, and the top 50%% represent %.1f%%.' % 
      ((sum([count for count in cat_counts if count > 10]) / len(df))*100, 
       (sum([count for count in cat_counts if count > 76]) / len(df))*100))


# There are a lot of uncommon or unique categories that make up a small percentage of the data. If dimensionality reduction needs to happen here, I think it would be safe to keep only the top half of category names and the remaining ~10th of a percent of data will be grouped together as items with an uncommon category. 

# In[ ]:


title = 'Top 35 Categories'
plt.figure(title, figsize=(30,10))
df.category_name.value_counts()[0:35].plot(kind='bar')
plt.title(title, fontsize=30)
plt.yticks(fontsize=18)
plt.xticks(fontsize=18, rotation=35, ha='right')
plt.show()


# ## Brand Name

# In[ ]:


brand_counts = np.sort(df.brand_name.value_counts())
print(str(len(brand_counts)) + ' brands total.')
print(str(df.shape[0]) + ' records total.')
print('Category frequency percentiles, marked by lines: \n25%%: %d, 50%%: %d, 75%%: %d, 95%%: %d, 97.5%%: %d.' % 
     (brand_counts[int(len(brand_counts)*0.25)], 
      brand_counts[int(len(brand_counts)*0.5)],
      brand_counts[int(len(brand_counts)*0.75)],
      brand_counts[int(len(brand_counts)*0.9)],
      brand_counts[int(len(brand_counts)*0.95)]))

title = 'Brand Quantity ECDF Without Top 25 Outliers'
plt.figure(title, figsize=(30,10))
plt.title(title, fontsize=32)
x = np.sort(df.brand_name.value_counts())
x = x[0:-25]
y = np.arange(1, len(x) + 1) / len(x)
plt.plot(x, y, marker='.', linestyle='none')
plt.xticks(fontsize=24)
plt.yticks(fontsize=26)
plt.axvline(x=x[int(len(x)*0.25)], linewidth=1, color='b')
plt.axvline(x=x[int(len(x)*0.5)], linewidth=1, color='b')
plt.axvline(x=x[int(len(x)*0.75)], linewidth=1, color='b')
plt.axvline(x=x[int(len(x)*0.95)], linewidth=1, color='b')
plt.axvline(x=x[int(len(x)*0.975)], linewidth=1, color='b')
plt.show()


# In[ ]:


print('The top 75%% of categories represent %.1f%% of the dataset, and the top 50%% represent %.1f%%.' % 
      ((sum([count for count in brand_counts if count > 1]) / len(df))*100, 
       (sum([count for count in brand_counts if count > 4]) / len(df))*100))


# A story similar to category_name.

# In[ ]:


print('%d items, or %.2f%%, are missing a brand name.' % 
      (len(df[df.brand_name == 'no_brand']), 
       len(df[df.brand_name == 'no_brand']) / len(df)))


# In[ ]:


title = 'Top 35 Brands'
plt.figure(title, figsize=(30,10))
df.brand_name.value_counts()[1:70].plot(kind='bar')
plt.title(title, fontsize=30)
plt.yticks(fontsize=18)
plt.xticks(fontsize=18, rotation=45, ha='right')
plt.show()


# The most popular brands, PINK and Nike, are an order of magnitude less frequent than unbranded items. It seems there's a mix of company brands and individual product line brands, as we can see both Victoria's Secret and Pink as well as Nintendo and Pokemon. 

# In[ ]:


title = 'Top Half of Brands'
plt.figure(title, figsize=(30,10))
df.brand_name.value_counts()[50:2500].plot(kind='bar')
plt.title(title, fontsize=30)
plt.yticks(fontsize=18)
plt.xticks(fontsize=0, rotation=45, ha='right')
plt.show()


# An exponential growth curve that explodes at the end. I just like making huge charts like this. 

# In[ ]:


df.columns


# # Preprocessing 

# ## Natural Language Processing 

# In[ ]:


import nltk
nltk.data.path.append(r'D:\Python\Data Sets\nltk_data')
from nltk.corpus import stopwords 
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer 
from scipy import sparse 


# ### Category Name

# This is pretty straightforward. Make dummy categories for each categorical value, with uncommon values just zero. But since this dataset is big and there's a large number of categories, the best way to use this is to use CountVectorizer() because it returns a sparse matrix instead of a dense one. 

# In[ ]:


cat_vec = CountVectorizer(stop_words=[stopwords, string.punctuation], max_features=int(len(cat_counts)*0.9))
cat_matrix = cat_vec.fit_transform(df.category_name)
cat_matrix_sub = cat_vec.transform(df_sub.category_name)


# In[ ]:


# For exploring the tokens. The array is an array inside of an array of one, ravel pulls it out. 
cat_tokens = list(zip(cat_vec.get_feature_names(), np.array(cat_matrix.sum(axis=0)).ravel()))


# ### Brand Name 

# In[ ]:


brand_vec = CountVectorizer(stop_words=[stopwords, string.punctuation], max_features=int(len(brand_counts)*0.5))
brand_matrix = brand_vec.fit_transform(df.brand_name)
brand_matrix_sub = brand_vec.transform(df_sub.brand_name)


# In[ ]:


brand_tokens = list(zip(brand_vec.get_feature_names(), np.array(brand_matrix.sum(axis=0)).ravel()))


# ### Item Name

# Item name and description are more complicated. As they are phrases and sentences, the number of words is going to be exponentially larger and the words themselves don't hold equal weight. I'm going to use a statistical method called Term Frequency - Inverse Document Frequency (TF-IDF) that combines the bag of words approach with a weight adjustment based on the overall frequency of each term in the dataset. 

# In[ ]:


name_vec = TfidfVectorizer(min_df=15, stop_words=[stopwords, string.punctuation])
name_matrix = name_vec.fit_transform(df.name)
name_matrix_sub = name_vec.transform(df_sub.name)


# In[ ]:


print('Kept %d words.' % len(name_vec.get_feature_names()))


# ### Description

# In[ ]:


desc_vec = TfidfVectorizer(max_features=200000,
                           stop_words=[stopwords, string.punctuation])
desc_matrix = desc_vec.fit_transform(df.item_description)
desc_matrix_sub= desc_vec.transform(df_sub.item_description)


# ### Condition and Shipping 

# In[ ]:


cond_matrix = sparse.csr_matrix(pd.get_dummies(df.item_condition_id, sparse=True, drop_first=True))
cond_matrix_sub = sparse.csr_matrix(pd.get_dummies(df_sub.item_condition_id, sparse=True, drop_first=True))


# In[ ]:


ship_matrix = sparse.csr_matrix(df.shipping).transpose()
ship_matrix_sub = sparse.csr_matrix(df_sub.shipping).transpose()


# ### Combine Sparse Matrices 

# In[ ]:


sparse_matrix = sparse.csr_matrix(sparse.hstack([cat_matrix, brand_matrix, name_matrix, desc_matrix, 
                               cond_matrix, ship_matrix]))
sparse_matrix_sub = sparse.csr_matrix(sparse.hstack([cat_matrix_sub, brand_matrix_sub, name_matrix_sub, 
                                   desc_matrix_sub, cond_matrix_sub, ship_matrix_sub]))


# In[ ]:


if sparse_matrix.shape[1] == sparse_matrix_sub.shape[1]:
    print('Features check out.')
else:
    print("The number of features in training and test set don't match.")


# ### Garbage Collection

# In[ ]:


import gc
del(cat_matrix, brand_matrix, name_matrix, desc_matrix, cond_matrix, ship_matrix)
del(cat_matrix_sub, brand_matrix_sub, name_matrix_sub, desc_matrix_sub, cond_matrix_sub, ship_matrix_sub)
del(df, df_sub)
gc.collect()


# In[ ]:


print_time(start)


# # Training

# In[ ]:


def rmsle(true, pred):
    assert len(pred) == len(true)
    return np.sqrt(np.mean(np.power(np.log1p(pred)-np.log1p(true), 2)))


# Take the log of the target data to boost training accuracy. 

# In[ ]:


y_target = np.log10(np.array(y_target) + 1)


# Split training and test set 

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(sparse_matrix, y_target, test_size = 0.1)


# In[ ]:


import lightgbm as lgb


# In[ ]:


def rmsle_lgb(y_true, y_pred):
    assert len(y_true) == len(y_pred)
    return np.sqrt(np.mean(np.power(np.log1p(y_true + 1) - np.log1p(y_pred + 1), 2)))


# In[ ]:


print("Cell didn't crash... ")
start = time.time()
params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'rmse',
    'verbose': 1,
    'num_leaves': 31,
    'n_estimators': 1000, 
    'learning_rate': 0.5, 
    'max_depth': 10
}

reg_lgbm = lgb.LGBMRegressor(**params, n_jobs=4)
print('check')
reg_lgbm.fit(X_train, y_train)
y_pred = reg_lgbm.predict(X_test)

print(rmsle(10 ** y_test - 1, 10 ** y_pred - 1))
print_time(start)


# ### Stacking 

# Stacking Function. This function returns dataframes of predictions from each input model that can be merged into the train, test, and submission datasets. I used scikit-learn's 'clone' method, so all the weights will be stripped from the input models. This lets you input either fresh models or previously used models without worrying about it. A cloned model is generated for each training fold. I learned how to do this here, and it's an article worth reading: 
# http://blog.kaggle.com/2016/12/27/a-kagglers-guide-to-model-stacking-in-practice/
# 
# In my experiments, I've gotten 2-3% better performance from stacking a few models together than the best model does on it's own. Interestingly, even adding a crap model (like the Lasso regression with a score of 0.75) increases the overall performance. The stacked models seem to work better with tree based models than regression models. 

# In[ ]:


from sklearn.model_selection import KFold
from sklearn.base import clone
def stack_predictions(X_train, y_train, X_test, submit, K, *models):
    train_preds = pd.DataFrame(index=np.array(range(X_train.shape[0])))
    test_preds = pd.DataFrame(index=np.array(range(X_test.shape[0])))
    submit_preds = pd.DataFrame(index=np.array(range(submit.shape[0])))
    folds = KFold(n_splits=K, shuffle=True)
    
    fold_n = 0
    train_folds = np.zeros(len(train_preds))
    for train_index, test_index in folds.split(X_train):
        train_folds[test_index] = fold_n
        fold_n += 1
    
    fold_n = 0
    test_folds = np.zeros(len(test_preds))
    for train_index, test_index in folds.split(X_test):
        test_folds[test_index] = fold_n
        fold_n += 1
    
    fold_n = 0
    submit_folds = np.zeros(len(submit_preds))
    for train_index, test_index in folds.split(submit):
        submit_folds[test_index] = fold_n
        fold_n += 1
    
    for m, model in enumerate(models):
        print('Selecting model %d.' % (m+1))
        col = 'pred_col_' + str(m)
        train_preds[col] = np.nan
        test_preds[col] = np.nan
        submit_preds[col] = np.nan
        
        for fold in range(K):
            print('Processing a fold...')
            current_model = clone(model)
            current_model.fit(X_train[np.where(train_folds!=fold)], y_train[np.where(train_folds!=fold)])
            
            train_preds[col].iloc[np.where(train_folds==fold)] = current_model.predict(
                X_train[np.where(train_folds==fold)])
            
            test_preds[col].iloc[np.where(test_folds==fold)] = current_model.predict(
                X_test[np.where(test_folds==fold)])
            
            submit_preds[col].iloc[np.where(submit_folds==fold)] = current_model.predict(
                submit[np.where(submit_folds==fold)])  

    return train_preds, test_preds, submit_preds


# ### Create Stacked Models 

# In[ ]:


from sklearn.linear_model import Ridge
reg_ridge = Ridge(solver='sag', alpha=5)

import lightgbm as lgb
params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'rmse',
    'verbose': 0,
    'num_leaves': 31
}

reg_lgbm = lgb.LGBMRegressor(**params)


# In[ ]:


start = time.time()
train_preds, test_preds, sub_preds = stack_predictions(X_train, y_train, X_test, 
                                                       sparse_matrix_sub, 5, 
                                                       reg_ridge, reg_lgbm)
print_time(start)


# In[ ]:


X_train_stacked = sparse.csr_matrix(sparse.hstack([X_train, sparse.csr_matrix(train_preds)]))
X_test_stacked = sparse.csr_matrix(sparse.hstack([X_test, sparse.csr_matrix(test_preds)]))
sub_stacked = sparse.csr_matrix(sparse.hstack([sparse_matrix_sub, sparse.csr_matrix(sub_preds)]))


# In[ ]:


start = time.time()
from sklearn.linear_model import Ridge
reg_ridge = Ridge(solver='sag', alpha=5)
reg_ridge.fit(X_train_stacked, y_train)
y_pred = reg_ridge.predict(X_test_stacked)
print(rmsle(10 ** y_test - 1, 10 ** y_pred - 1))
print_time(start)


# In[ ]:


start = time.time()
params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'rmse',
    'verbose': 0,
    'num_leaves': 35,
    'n_estimators': 1000, 
    'learning_rate': 0.5, 
    'max_depth': 10,
}

reg_lgbm = lgb.LGBMRegressor(**params, n_jobs=4)
reg_lgbm.fit(X_train_stacked, y_train)
r_pred = reg_lgbm.predict(X_test_stacked)

print(rmsle(10 ** y_test - 1,10 ** r_pred - 1))
print_time(start)


# Another LGBM model trained with the additional features from the input Ridge and LGBM models. I think it can evaluate how accurate the LGBM and Ridge predictions are along with the context of all the previous features, and make a more informed prediction. 

# In[ ]:


pred_sub = reg_lgbm.predict(sub_stacked)
lightgbm_submission = submission.copy()
lightgbm_submission['price'] = pd.DataFrame(10 ** pred_sub - 1)

lightgbm_submission.to_csv('stacked_submission_1.csv', index=False)

