#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
import math
import gc
import time
#------------------------------------------------------------------------------------------------#

start_time = time.time()

# Read in the datafiles
df_train = pd.read_csv('../input/train.tsv', sep='\t') 
df_test = pd.read_csv('../input/test.tsv', sep='\t')

#Mercari only lets users sell products with a price higher than 3.0, so we only want the 
#data to contain products with a price of 3.0 or higher.
df_train[df_train.price >= 3.0]

# Concatenate the training and test data but save the length of the training data for later use
train_rows = len(df_train)
all_data = pd.concat([df_train, df_test])
print("[{}] Data concatenated.".format(time.time() - start_time))

# Function for filling the NaN values in the dataframe
def transform_values(data):
    data.fillna(value='missing', inplace = True)
    return data

all_data = transform_values(all_data)

# Split the categories into a maincategory, subcategory 1 and subcategory 2
def transform_category_name(category_name):
    # If there is a category split it
    try:
        main, sub1, sub2= category_name.split('/')
        return main, sub1, sub2
    # Else return "missing" for each (sub-)category
    except:
        return 'missing', 'missing', 'missing'

all_data['category_main'], all_data['category_sub1'], all_data['category_sub2'] = zip(*all_data['category_name'].apply(transform_category_name))
# Delete all the entries without an item description
all_data[all_data.item_description != 'missing']
print("[{}] Data transformed.".format(time.time() - start_time))

# Create a vector for all the names with count vectorizer
name_cv = CountVectorizer(ngram_range=(1,2), min_df=7, stop_words='english')
X_name = name_cv.fit_transform(all_data['name'])
print("[{}] Name vector created.".format(time.time() - start_time))

# Create seperate vectors for all the (sub-)categories using a CountVectorizer
cat_cv = CountVectorizer()
X_catmain = cat_cv.fit_transform(all_data['category_main'],)
X_catsub1 = cat_cv.fit_transform(all_data['category_sub1'])
X_catsub2 = cat_cv.fit_transform(all_data['category_sub2'])
print("[{}] Category vectors created.".format(time.time() - start_time))

# Function to get length of item descriptions
def descrLength(description):
    try:
        if description == "missing":
            return 0
        else:
            words = [w for w in description.split(" ")]
            return len(words)
    except:
        return 0
      
# Create an extra column in the dataframe that keeps track of the length of the descriptions
all_data['descr_length'] = all_data['item_description'].apply(lambda x: descrLength(x))
all_data['descr_length'] = all_data['descr_length'].astype(str)
X_length = cat_cv.fit_transform(all_data['descr_length'])
print("[{}] Description length vector created.".format(time.time() - start_time))

# Create a vector with tf-idf values of the words in the descriptions.
vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=90000, strip_accents="ascii",
                             stop_words='english')
X_description = vectorizer.fit_transform(all_data['item_description'])
print("[{}] Description vector created.".format(time.time() - start_time))
       
# Make a vector of the brands using a label binarizer.
lb = LabelBinarizer(sparse_output=True)
X_brand = lb.fit_transform(all_data['brand_name'])
print("[{}] Brand vector created.".format(time.time() - start_time))

# Use get dummies to create a sparse csr matrix of the shipping and item condition values.
X_dummies = csr_matrix(pd.get_dummies(all_data[['item_condition_id', 'shipping']],
                                          sparse=True).values)
print("[{}] Shipping and condition vector created.".format(time.time() - start_time))

train_X = hstack((X_name[:train_rows], X_catmain[:train_rows], X_catsub1[:train_rows], 
                X_catsub2[:train_rows], X_brand[:train_rows], X_length[:train_rows], X_description[:train_rows],
                X_dummies[:train_rows]))
print('[{}] Training data stacked'.format(time.time() - start_time))

test_X = hstack((X_name[train_rows:], X_catmain[train_rows:], X_catsub1[train_rows:], 
                X_catsub2[train_rows:], X_brand[train_rows:], X_length[train_rows:], X_description[train_rows:],
                X_dummies[train_rows:]))
print('[{}] Test data stacked.'.format(time.time() - start_time))

Y_train = np.log1p(df_train['price']) #log1p because is nice
print("[{}] Data processing and feature extraction complete.".format(time.time() - start_time))


# In[ ]:


from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

ridge_time = time.time()

# Split the training data in training and validation sets to use cross validation
data_train_X, data_val_X, data_train_Y, data_val_Y = train_test_split(train_X, Y_train)
print('[{}] Data splitted.'.format(time.time() - ridge_time))

# Training the Ridge algorithm on the trainingsset
clf1 = Ridge(alpha=5.3, fit_intercept=True, normalize=False, 
      copy_X=True, max_iter=None, tol=0.01, solver='auto', random_state=None)
clf1.fit(train_X, Y_train)
print("[{}] Data 1 fitted.".format(time.time() - ridge_time))

clf2 = Ridge(alpha=.75, fit_intercept=True, normalize=False, 
      copy_X=True, max_iter=None, tol=0.01, solver='auto', random_state=100)
clf2.fit(train_X, Y_train)
print("[{}] Data 2 fitted.".format(time.time() - ridge_time))

# Use the Ridge model to predict the Y values for the validation set
#Y_pred1 = clf1.predict(data_val_X)
#Y_pred2 = clf2.predict(data_val_X)

#A function to calculate Root Mean Squared Logarithmic Error (RMSLE)
'''def rmsle(y, y0):
     assert len(y) == len(y0)
     return np.sqrt(np.mean(np.power(np.log1p(y)-np.log1p(y0), 2)))

print("Error on 1:", rmsle(np.expm1(Y_pred1), np.expm1(data_val_Y)))
print("Error on 2:", rmsle(np.expm1(Y_pred2), np.expm1(data_val_Y)))
'''


# 5.3 = 0.4512697
# 
# 0.53 = 0.4590716
# 
# with everything = 0.4504744
# 
# without brand = 0.45662041 --> slechter
# 
# without category = 0.4595244 --> slechter
# 
# without shipping = 0.457 --> slechter
# 
# without item condition = 
# 
# price higher than 3.0 = 0.4488579
# 
# missing ipv no description = 0.4500646/0.4504321/0.4496406 --> geen verbetering
# 
# gewoon no description = 0.4507507/0.4493476/0.4503064
# 
# with description length = 0.4503645/0.44881417/0.4481222 --> beetje verbetering
# 
# with normal brands = 0.4499219
# 
# with extra brands from name = 0.4492130/0.450550142/0.4502868
# 

# In[ ]:


from sklearn.model_selection import train_test_split
import lightgbm as lgb

lgb_time = time.time()

# Split the training data in training and validation sets to use cross validation
data_train_X, data_val_X, data_train_Y, data_val_Y = train_test_split(train_X, Y_train)
print('[{}] Data splitted.'.format(time.time() - lgb_time))

# Create the datasets to use for LGBM
dataset_train = lgb.Dataset(data_train_X, label=data_train_Y)
dataset_val = lgb.Dataset(data_val_X, label=data_val_Y)
# The list of data that LGBM uses to evaluate its results to avoid overfitting
watchlist = [dataset_train, dataset_val]

# Parameters for the LGBM model.
params = {
        'learning_rate': 0.35,
        'application': 'regression',
        'max_depth': 6,
        'num_leaves': 60,
        'verbosity': -1,
        'metric': 'RMSE',
        'data_random_seed': 1,
        'bagging_fraction': 0.5,
        'nthread': 4
    }

# Train the LGBM model on the trainingdata
lgb_model = lgb.train(params, train_set=dataset_train, num_boost_round=7500, valid_sets=watchlist, 
            early_stopping_rounds=1000, verbose_eval=500) 
print("[{}] Model completed.".format(time.time() - lgb_time))

# Predict the values of Y for the validation set with the created model
#lgb_pred =lgb_model.predict(data_val_X)
print("[{}] Prediction completed.".format(time.time() - lgb_time))

#print(rmsle(np.expm1(lgb_pred), np.expm1(data_val_Y)))


# In[ ]:


final_time = time.time()

final_ridge_pred1 = clf1.predict(test_X)
print("[{}] First Ridge prediction completed.".format(time.time() - final_time))
final_ridge_pred2 = clf2.predict(test_X)
print("[{}] Second Ridge prediction completed.".format(time.time() - final_time))
final_lgb_pred = lgb_model.predict(test_X)
print("[{}] Lgb prediction completed.".format(time.time() - final_time))

final_pred = 0.25*final_ridge_pred1 + 0.25*final_ridge_pred2 + 0.5*final_lgb_pred

submission = pd.DataFrame(data=df_test[['test_id']])
submission['price'] = np.expm1(final_pred)
submission.to_csv("submission_ridge_lgbm.csv", index=False)


# In[ ]:


'''
# Try and find brand names in product names
all_brands = set(all_data['brand_name'])
def brands_in_name(entry):
    name = entry[0]
    brand = entry[1]
    new_brand = ''
    split_name = name.split(' ')
    if brand == 'missing':
        for words in split_name:
            if words in all_brands:
                new_brand += words
        if new_brand == '':
            return 'missing'
        else:
            return new_brand
    return brand

all_data['brand_name'] = all_data[['name', 'brand_name']].apply(brands_in_name, axis = 1) 
print('new brands extracted')
'''


# * alpha 5 = 0.45512
# * alpha 5.1 = 0.45513
# * alpha 5.2 = 0.45511
# * alpha 5.3 = 0.45510
# * alpha 5.4 = 0.45512
# * alpha 5.5 = 0.45511
# * alpha 0.5 = 0.461
# * alpha 0.6 = 0.460
# * alpha 15 = 0.458
# * alpha 100 = 0.481
