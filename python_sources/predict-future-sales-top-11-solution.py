#!/usr/bin/env python
# coding: utf-8

# 

# # Part 1
# 
# What's the plan?
# 
# 1 Understand our data better in **Exploratory Data Analysis**, do necessary data wrangling
# 
# 2 Use sales from Oct 2015 as predictions for Nov 2015(**Previous Value Benchmark**)
# 
# 3 **Quick Baseline**. Apply some variant of decision tree(without any feature engineering, compare this with previous value benchmark)
# 
# 4 Set up **Cross Validation** to try out different feature engineering ideas
# 
# 5 Tune decision tree models, try to tune and get several diverse models with similar performance
# 
# 6 Use Ensemble methods to boost score
#   
# Btw, I'll omit the ploting part of EDA and all outputs of my code, because I am just compiling my notebooks and upload to kaggle as a kernel for future reference. But feel free to use my code here to get started and try my feature engineering ideas!

# # Exploratory Data Analysis

# ## Import necessary libraries

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import time

from math import sqrt
from numpy import loadtxt
from itertools import product
from tqdm import tqdm
from sklearn import preprocessing
from xgboost import plot_tree
from matplotlib import pyplot

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer

kernel_with_output = False


# ## Data loading
# Load all provided datasets and get a feel of the data provided to us

# In[ ]:


if kernel_with_output:
    sales_train = pd.read_csv('data/sales_train.csv')
    items = pd.read_csv('data/items.csv')
    shops = pd.read_csv('data/shops.csv')
    item_categories = pd.read_csv('data/item_categories.csv')
    test = pd.read_csv('data/test.csv')
    sample_submission = pd.read_csv('data/sample_submission.csv')


# ## Insert missing rows and aggregations

# In[ ]:


if kernel_with_output:
    # For every month we create a grid from all shops/items combinations from that month
    grid = []
    for block_num in sales_train['date_block_num'].unique():
        cur_shops = sales_train[sales_train['date_block_num']==block_num]['shop_id'].unique()
        cur_items = sales_train[sales_train['date_block_num']==block_num]['item_id'].unique()
        grid.append(np.array(list(product(*[cur_shops, cur_items, [block_num]])),dtype='int32'))
    index_cols = ['shop_id', 'item_id', 'date_block_num']
    grid = pd.DataFrame(np.vstack(grid), columns = index_cols,dtype=np.int32)

    # Aggregations
    sales_train['item_cnt_day'] = sales_train['item_cnt_day'].clip(0,20)
    groups = sales_train.groupby(['shop_id', 'item_id', 'date_block_num'])
    trainset = groups.agg({'item_cnt_day':'sum', 'item_price':'mean'}).reset_index()
    trainset = trainset.rename(columns = {'item_cnt_day' : 'item_cnt_month'})
    trainset['item_cnt_month'] = trainset['item_cnt_month'].clip(0,20)

    trainset = pd.merge(grid,trainset,how='left',on=index_cols)
    trainset.item_cnt_month = trainset.item_cnt_month.fillna(0)

    # Get category id
    trainset = pd.merge(trainset, items[['item_id', 'item_category_id']], on = 'item_id')
    trainset.to_csv('trainset_with_grid.csv')

    trainset.head()


# # Previous Value Benchmark
# **Copy from coursera**  
# "
# A good exercise is to reproduce previous_value_benchmark. As the name suggest - in this benchmark for the each shop/item pair our predictions are just monthly sales from the previous month, i.e. October 2015.
# 
# The most important step at reproducing this score is correctly aggregating daily data and constructing monthly sales data frame. You need to get lagged values, fill NaNs with zeros and clip the values into [0,20] range. If you do it correctly, you'll get precisely 1.16777 on the public leaderboard.
# 
# Generating features like this is a necessary basis for more complex models. Also, if you decide to fit some model, don't forget to clip the target into [0,20] range, it makes a big difference."
# 
# ** Comments **
# 
# Simply put: Use October 2015 sales(number of items sold) as our predictions for sales of November 2015
# 

# In[ ]:


if kernel_with_output:
    prev_month_selector = (trainset.month == 10) & (trainset.year == 2015)
    train_subset = trainset[prev_month_selector]
    groups = train_subset[['shop_id', 'item_id', 'item_cnt_month']].groupby(by = ['shop_id', 'item_id'])
    train_subset = groups.agg({'item_cnt_month':'sum'}).reset_index()
    train_subset.head(3)


# In[ ]:


if kernel_with_output:
    merged = test.merge(train_subset, on=["shop_id", "item_id"], how="left")[["ID", "item_cnt_month"]]
    merged.isna().sum()


# After merging, we will have lots of missing values of item_cnt_month. This is because we only have so much shop_id/item_id pair from Oct 2015. Fill missing values with 0 and clip values to range (0,20)

# In[ ]:


if kernel_with_output:
    merged['item_cnt_month'] = merged.item_cnt_month.fillna(0).clip(0,20)
    submission = merged.set_index('ID')
    submission.to_csv('benchmark.csv')


# # Quick Baseline with XGBoost
# Here, I'll use only the following features to make a quick baseline solution for the problem  
#   
#   **'shop_id', 'item_id', 'item_category_id', 'date_block_num'**  
#   
# Note that target is **item_cnt_month**

# In[ ]:


if kernel_with_output:
    # Extract features and target we want
    baseline_features = ['shop_id', 'item_id', 'item_category_id', 'date_block_num', 'item_cnt_month']
    train = trainset[baseline_features]
    # Remove pandas index column
    train = train.set_index('shop_id')
    train.item_cnt_month = train.item_cnt_month.astype(int)
    train['item_cnt_month'] = train.item_cnt_month.fillna(0).clip(0,20)
    # Save train set to file
    train.to_csv('train.csv')


# In[ ]:


if kernel_with_output:
    dataset = loadtxt('train.csv', delimiter="," ,skiprows=1, dtype = int)
    trainx = dataset[:, 0:4]
    trainy = dataset[:, 4]

    test_dataset = loadtxt('data/test.csv', delimiter="," ,skiprows=1, usecols = (1,2), dtype=int)
    test_df = pd.DataFrame(test_dataset, columns = ['shop_id', 'item_id'])

    # Make test_dataset pandas data frame, add category id and date block num, then convert back to numpy array and predict
    merged_test = pd.merge(test_df, items, on = ['item_id'])[['shop_id','item_id','item_category_id']]
    merged_test['date_block_num'] = 33
    merged_test.set_index('shop_id')
    merged_test.head(3)

    model = xgb.XGBRegressor(max_depth = 10, min_child_weight=0.5, subsample = 1, eta = 0.3, num_round = 1000, seed = 1)
    model.fit(trainx, trainy, eval_metric='rmse')
    preds = model.predict(merged_test.values)

    df = pd.DataFrame(preds, columns = ['item_cnt_month'])
    df['ID'] = df.index
    df = df.set_index('ID')
    df.to_csv('simple_xgb.csv')


# After my first submission to Kaggle, I get RMSE score of about 15. Definitely not acceptable. After clipping the target to range 0-20, I got RMSE of 2.12. Which is close to what I expect a plain Gradient Boosted Tree can get. In order to improve the score, we'll set up a cross validation scheme below and try different feature engineering ideas and see if we can do better.

# # Part2
# ## Set up some global vars 

# In[ ]:


if kernel_with_output:
    # Set seeds and options
    np.random.seed(10)
    pd.set_option('display.max_rows', 231)
    pd.set_option('display.max_columns', 100)

    # Feature engineering list
    new_features = []
    enable_feature_idea = [True, True, True, True, True, True, True, True, True, True]

    # Some parameters(maybe add more periods, score will be better) [1,2,3,12]
    lookback_range = [1,2,3,4,5,6,7,8,9,10,11,12]

    tqdm.pandas()


# ## Load data

# In[ ]:


if kernel_with_output:
    current = time.time()

    trainset = pd.read_csv('trainset_with_grid.csv')
    items = pd.read_csv('data/items.csv')
    shops = pd.read_csv('data/shops.csv')


    # Only use more recent data
    start_month = 0
    end_month = 33
    trainset = trainset[['shop_id', 'item_id', 'item_category_id', 'date_block_num', 'item_price', 'item_cnt_month']]
    trainset = trainset[(trainset.date_block_num >= start_month) & (trainset.date_block_num <= end_month)]

    print('Loading test set...')
    test_dataset = loadtxt('data/test.csv', delimiter="," ,skiprows=1, usecols = (1,2), dtype=int)
    testset = pd.DataFrame(test_dataset, columns = ['shop_id', 'item_id'])

    print('Merging with other datasets...')
    # Get item category id into test_df
    testset = testset.merge(items[['item_id', 'item_category_id']], on = 'item_id', how = 'left')
    testset['date_block_num'] = 34
    # Make testset contains same column as trainset so we can concatenate them row-wise
    testset['item_cnt_month'] = -1

    train_test_set = pd.concat([trainset, testset], axis = 0) 

    end = time.time()
    diff = end - current
    print('Took ' + str(int(diff)) + ' seconds to train and predict val set')


# ## Fix category

# In[ ]:


if kernel_with_output:
    item_cat = pd.read_csv('data/item_categories.csv')

    # Fix category
    l_cat = list(item_cat.item_category_name)
    for ind in range(0,1):
        l_cat[ind] = 'PC Headsets / Headphones'
    for ind in range(1,8):
        l_cat[ind] = 'Access'
    l_cat[8] = 'Tickets (figure)'
    l_cat[9] = 'Delivery of goods'
    for ind in range(10,18):
        l_cat[ind] = 'Consoles'
    for ind in range(18,25):
        l_cat[ind] = 'Consoles Games'
    l_cat[25] = 'Accessories for games'
    for ind in range(26,28):
        l_cat[ind] = 'phone games'
    for ind in range(28,32):
        l_cat[ind] = 'CD games'
    for ind in range(32,37):
        l_cat[ind] = 'Card'
    for ind in range(37,43):
        l_cat[ind] = 'Movie'
    for ind in range(43,55):
        l_cat[ind] = 'Books'
    for ind in range(55,61):
        l_cat[ind] = 'Music'
    for ind in range(61,73):
        l_cat[ind] = 'Gifts'
    for ind in range(73,79):
        l_cat[ind] = 'Soft'
    for ind in range(79,81):
        l_cat[ind] = 'Office'
    for ind in range(81,83):
        l_cat[ind] = 'Clean'
    l_cat[83] = 'Elements of a food'

    lb = preprocessing.LabelEncoder()
    item_cat['item_category_id_fix'] = lb.fit_transform(l_cat)
    item_cat['item_category_name_fix'] = l_cat
    train_test_set = train_test_set.merge(item_cat[['item_category_id', 'item_category_id_fix']], on = 'item_category_id', how = 'left')
    _ = train_test_set.drop(['item_category_id'],axis=1, inplace=True)
    train_test_set.rename(columns = {'item_category_id_fix':'item_category_id'}, inplace = True)

    _ = item_cat.drop(['item_category_id'],axis=1, inplace=True)
    _ = item_cat.drop(['item_category_name'],axis=1, inplace=True)

    item_cat.rename(columns = {'item_category_id_fix':'item_category_id'}, inplace = True)
    item_cat.rename(columns = {'item_category_name_fix':'item_category_name'}, inplace = True)
    item_cat = item_cat.drop_duplicates()
    item_cat.index = np.arange(0, len(item_cat))


# # Idea 0: Add previous shop/item sales as feature (Lag feature)

# In[ ]:


if kernel_with_output:
    if enable_feature_idea[0]:
        for diff in tqdm(lookback_range):
            feature_name = 'prev_shopitem_sales_' + str(diff)
            trainset2 = train_test_set.copy()
            trainset2.loc[:, 'date_block_num'] += diff
            trainset2.rename(columns={'item_cnt_month': feature_name}, inplace=True)
            train_test_set = train_test_set.merge(trainset2[['shop_id', 'item_id', 'date_block_num', feature_name]], on = ['shop_id', 'item_id', 'date_block_num'], how = 'left')
            train_test_set[feature_name] = train_test_set[feature_name].fillna(0)
            new_features.append(feature_name)
    train_test_set.head(3)


# # Idea 1: Add previous item sales as feature (Lag feature)

# In[ ]:


if kernel_with_output:
    if enable_feature_idea[1]:
        groups = train_test_set.groupby(by = ['item_id', 'date_block_num'])
        for diff in tqdm(lookback_range):
            feature_name = 'prev_item_sales_' + str(diff)
            result = groups.agg({'item_cnt_month':'mean'})
            result = result.reset_index()
            result.loc[:, 'date_block_num'] += diff
            result.rename(columns={'item_cnt_month': feature_name}, inplace=True)
            train_test_set = train_test_set.merge(result, on = ['item_id', 'date_block_num'], how = 'left')
            train_test_set[feature_name] = train_test_set[feature_name].fillna(0)
            new_features.append(feature_name)        
    train_test_set.head(3)


# # Idea 2: Add previous shop/item price as feature (Lag feature)

# In[ ]:


if kernel_with_output:
    if enable_feature_idea[3]:
        groups = train_test_set.groupby(by = ['shop_id', 'item_id', 'date_block_num'])
        for diff in tqdm(lookback_range):
            feature_name = 'prev_shopitem_price_' + str(diff)
            result = groups.agg({'item_price':'mean'})
            result = result.reset_index()
            result.loc[:, 'date_block_num'] += diff
            result.rename(columns={'item_price': feature_name}, inplace=True)
            train_test_set = train_test_set.merge(result, on = ['shop_id', 'item_id', 'date_block_num'], how = 'left')
            train_test_set[feature_name] = train_test_set[feature_name]
            new_features.append(feature_name)        
    train_test_set.head(3)


# # Idea 3: Add previous item price as feature (Lag feature)

# In[ ]:


if kernel_with_output:
    if enable_feature_idea[3]:
        groups = train_test_set.groupby(by = ['item_id', 'date_block_num'])
        for diff in tqdm(lookback_range):
            feature_name = 'prev_item_price_' + str(diff)
            result = groups.agg({'item_price':'mean'})
            result = result.reset_index()
            result.loc[:, 'date_block_num'] += diff
            result.rename(columns={'item_price': feature_name}, inplace=True)
            train_test_set = train_test_set.merge(result, on = ['item_id', 'date_block_num'], how = 'left')
            train_test_set[feature_name] = train_test_set[feature_name]
            new_features.append(feature_name)        
    train_test_set.head(3)


# # Idea 4: Mean encodings for shop/item pairs(Mean encoding, doesnt work for me)

# In[ ]:


def create_mean_encodings(train_test_set, categorical_var_list, target):
    feature_name = "_".join(categorical_var_list) + "_" + target + "_mean"

    df = train_test_set.copy()
    df1 = df[df.date_block_num <= 32]
    df2 = df[df.date_block_num <= 33]
    df3 = df[df.date_block_num == 34]

    # Extract mean encodings using training data(here we don't use month 33 to avoid data leak on validation)
    # If I try to extract mean encodings from all months, then val rmse decreases a tiny bit, but test rmse would increase by 4%
    # So this is important
    mean_32 = df1[categorical_var_list + [target]].groupby(categorical_var_list, as_index=False)[[target]].mean()
    mean_32 = mean_32.rename(columns={target:feature_name})

    # Extract mean encodings using all data, this will be applied to test data
    mean_33 = df2[categorical_var_list + [target]].groupby(categorical_var_list, as_index=False)[[target]].mean()
    mean_33 = mean_33.rename(columns={target:feature_name})

    # Apply mean encodings
    df2 = df2.merge(mean_32, on = categorical_var_list, how = 'left')
    df3 = df3.merge(mean_33, on = categorical_var_list, how = 'left')

    # Concatenate
    train_test_set = pd.concat([df2, df3], axis = 0)
    new_features.append(feature_name)
    return train_test_set


# In[ ]:


if kernel_with_output:
    create_mean_encodings(train_test_set, ['shop_id', 'item_id'], 'item_cnt_month')
    train_test_set.head(3)


# # Idea 5: Mean encodings for item (Mean encoding, doesnt work for me)

# In[ ]:


if kernel_with_output:
    train_test_set = create_mean_encodings(train_test_set, ['item_id'], 'item_cnt_month')
    train_test_set.head(3)


# # Idea 6: Number of month from last sale of shop/item (Use info from past)

# In[ ]:


def create_last_sale_shop_item(row):
    for diff in range(1,33+1):
        feature_name = '_prev_shopitem_sales_' + str(diff)
        if row[feature_name] != 0.0:
            return diff
    return np.nan

if kernel_with_output:
    lookback_range = list(range(1, 33 + 1))
    if enable_feature_idea[6]:
        for diff in tqdm(lookback_range):
            feature_name = '_prev_shopitem_sales_' + str(diff)
            trainset2 = train_test_set.copy()
            trainset2.loc[:, 'date_block_num'] += diff
            trainset2.rename(columns={'item_cnt_month': feature_name}, inplace=True)
            train_test_set = train_test_set.merge(trainset2[['shop_id', 'item_id', 'date_block_num', feature_name]], on = ['shop_id', 'item_id', 'date_block_num'], how = 'left')
            train_test_set[feature_name] = train_test_set[feature_name].fillna(0)
            #new_features.append(feature_name)

    train_test_set.loc[:, 'last_sale_shop_item'] = train_test_set.progress_apply (lambda row: create_last_sale_shop_item(row),axis=1)
    new_features.append('last_sale_shop_item')


# # Idea 7: Number of month from last sale of item(Use info from past)

# In[ ]:


def create_last_sale_item(row):
    for diff in range(1,33+1):
        feature_name = '_prev_item_sales_' + str(diff)
        if row[feature_name] != 0.0:
            return diff
    return np.nan
if kernel_with_output:
    lookback_range = list(range(1, 33 + 1))
    if enable_feature_idea[1]:
        groups = train_test_set.groupby(by = ['item_id', 'date_block_num'])
        for diff in tqdm(lookback_range):
            feature_name = '_prev_item_sales_' + str(diff)
            result = groups.agg({'item_cnt_month':'mean'})
            result = result.reset_index()
            result.loc[:, 'date_block_num'] += diff
            result.rename(columns={'item_cnt_month': feature_name}, inplace=True)
            train_test_set = train_test_set.merge(result, on = ['item_id', 'date_block_num'], how = 'left')
            train_test_set[feature_name] = train_test_set[feature_name].fillna(0)
            new_features.append(feature_name)        
    train_test_set.loc[:, 'last_sale_item'] = train_test_set.progress_apply (lambda row: create_last_sale_item(row),axis=1)


# # Idea 8: Item name (Tfidf text feature)

# In[ ]:


if kernel_with_output:
    items_subset = items[['item_id', 'item_name']]
    feature_count = 25
    tfidf = TfidfVectorizer(max_features=feature_count)
    items_df_item_name_text_features = pd.DataFrame(tfidf.fit_transform(items_subset['item_name']).toarray())

    cols = items_df_item_name_text_features.columns
    for i in range(feature_count):
        feature_name = 'item_name_tfidf_' + str(i)
        items_subset[feature_name] = items_df_item_name_text_features[cols[i]]
        new_features.append(feature_name)

    items_subset.drop('item_name', axis = 1, inplace = True)
    train_test_set = train_test_set.merge(items_subset, on = 'item_id', how = 'left')
    train_test_set.head()


# # Cross Validation

# In[ ]:


if kernel_with_output:
    current = time.time()

    baseline_features = ['shop_id', 'item_id', 'item_category_id', 'date_block_num'] +  new_features + ['item_cnt_month']

    # Clipping to range 0-20
    train_test_set['item_cnt_month'] = train_test_set.item_cnt_month.fillna(0).clip(0,20)

    # train: want rows with date_block_num from 0 to 31
    train_time_range_lo = (train_test_set['date_block_num'] >= 0)
    train_time_range_hi =  (train_test_set['date_block_num'] <= 32)

    # val: want rows with date_block_num from 22
    validation_time =  (train_test_set['date_block_num'] == 33)

    # test: want rows with date_block_num from 34
    test_time =  (train_test_set['date_block_num'] == 34)


    # Retrieve rows for train set, val set, test set
    cv_trainset = train_test_set[train_time_range_lo & train_time_range_hi]
    cv_valset = train_test_set[validation_time]
    cv_trainset = cv_trainset[baseline_features]
    cv_valset = cv_valset[baseline_features]
    testset = train_test_set[test_time]
    testset = testset[baseline_features]

    # Prepare numpy arrays for training/val/test
    cv_trainset_vals = cv_trainset.values.astype(int)
    trainx = cv_trainset_vals[:, 0:len(baseline_features) - 1]
    trainy = cv_trainset_vals[:, len(baseline_features) - 1]

    cv_valset_vals = cv_valset.values.astype(int)
    valx = cv_valset_vals[:, 0:len(baseline_features) - 1]
    valy = cv_valset_vals[:, len(baseline_features) - 1]

    testset_vals = testset.values.astype(int)
    testx = testset_vals[:, 0:len(baseline_features) - 1]

    print('Fitting...')
    model = xgb.XGBRegressor(max_depth = 11, min_child_weight=0.5, subsample = 1, eta = 0.3, num_round = 1000, seed = 1, nthread = 16)
    model.fit(trainx, trainy, eval_metric='rmse')


    preds = model.predict(valx)
    # Clipping to range 0-20
    preds = np.clip(preds, 0,20)
    print('val set rmse: ', sqrt(mean_squared_error(valy, preds)))

    preds = model.predict(testx)
    # Clipping to range 0-20
    preds = np.clip(preds, 0,20)
    df = pd.DataFrame(preds, columns = ['item_cnt_month'])
    df['ID'] = df.index
    df = df.set_index('ID')
    df.to_csv('test_preds.csv')
    print('test predictions written to file')

    end = time.time()
    diff = end - current
    print('Took ' + str(int(diff)) + ' seconds to train and predict val, test set')


# # Model Ensemble: Stacking
# 
# I have tried to combine models from CatBoost, XGboost and LightGBM with stacking, but the results aren't as good as using XGboost alone.

# # Conclusion
# 
# In the end, I got a rmse score of 0.89874 in the public leader board(top 11). For the private grader, I got rmse score of 0.88(yeah!! no overfitting)
# 
# I learned one most important thing in this competition. Feature engineering is the single most important thing in machine learning! If you don't expose the interactions of data to your model explictily, then no matter how you tune your model, it can not learn those interactions between data!
# 
# 
# 
