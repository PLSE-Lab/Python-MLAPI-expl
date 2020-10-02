#!/usr/bin/env python
# coding: utf-8

# 

# In[ ]:


import os
import sys
import operator
import numpy as np
import pandas as pd
from scipy import sparse
import xgboost as xgb
from sklearn import model_selection, preprocessing, ensemble
from sklearn.metrics import log_loss
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


# In[ ]:


data_path = "../input/"
train_file = data_path + "train.json"
test_file = data_path + "test.json"
train = pd.read_json(train_file)
test = pd.read_json(test_file)

train.shape
train.shape
train.head()
test.head()


# In[ ]:


feature engineering


# In[ ]:


listing_id = test_df.listing_id.values

y_map = {'low': 2, 'medium': 1, 'high': 0}
train['interest_level'] = train['interest_level'].apply(lambda x: y_map[x])
y_train = train.interest_level.values

train = train.drop(['listing_id', 'interest_level'], axis=1)
test = test.drop('listing_id', axis=1)

ntrain = train.shape[0]

train_test = pd.concat((train, test), axis=0).reset_index(drop=True)


features_to_use  = ["bathrooms","bedrooms","building_id", "created","latitude", "description",
                    "listing_id","longitude","manager_id", "price", "features", "display_address", 
                    "street_address","feature_count","photo_count", "interest_level"]

train_test["price_per_bed"] = train_df["price"]/train_df["bedrooms"] 
train_test["room_diff"] = train_test["bedrooms"] - train_test["bathrooms"] 
train_test["room_sum"] = train_test["bedrooms"] + train_test["bathrooms"] 
train_test["room_price"] = train_test["price"] / train_test["room_sum"]
train_test["bed_ratio"] = train_test["bedrooms"] / train_test["room_sum"]

train_test["photo_count"] = train_test["photos"].apply(len)
train_test["feature_count"] = train_test["features"].apply(len)

#log transform
train_test["photo_count"] = np.log(train_test["photo_count"] + 1)
train_test["feature_count"] = np.log(train_test["feature_count"] + 1)
train_test["price"] = np.log(train_test["price"] + 1)
train_test["price_per_bed"] = np.log(train_test["price_per_bed"] + 1)
train_test["room_diff"] = np.log(train_test["room_diff"] + 1)
train_test["room_sum"] = np.log(train_test["room_sum"] + 1)
train_test["room_price"] = np.log(train_test["room_price"] + 1)

train_test["description_word_count"] = train_test["description"].apply(lambda x: len(x.split(" ")))

#date time
train_test["created"] = pd.to_datetime(train_test["created"])
train_test["passed"] = train_test["created"].max() - train_test["created"]

train_test["created_year"] = train_test["created"].dt.year
train_test["created_month"] = train_test["created"].dt.month
train_test["created_day"] = train_test["created"].dt.day
train_test["created_hour"] = train_test["created"].dt.hour

 


# In[ ]:


#address
address_map = {
    'w': 'west',
    'st.': 'street',
    'ave': 'avenue',
    'st': 'street',
    'e': 'east',
    'n': 'north',
    's': 'south'
}

def address_map_func(s):
    s = s.split(' ')
    out = []
    for x in s:
        if x in address_map:
            out.append(address_map[x])
        else:
            out.append(x)
    return ' '.join(out)

train_test['dis_address'] = train_test['display_address']
train_test['dis_address'] = train_test['dis_address'].apply(lambda x: x.lower())
train_test['dis_address'] = train_test['dis_address'].apply(lambda x: x.translate(remove_punct_map))
train_test['dis_address'] = train_test['dis_address'].apply(lambda x: address_map_func(x))

new_cols = ['street', 'avenue', 'east', 'west', 'north', 'south']

for col in new_cols:
    train_test[col] = train_test['dis_address'].apply(lambda x: 1 if col in x else 0)
    
train_test['other_address'] = train_test[new_cols].apply(lambda x: 1 if x.sum() == 0 else 0, axis=1)    

#features
train_test['features_count'] = train_test['features'].apply(lambda x: len(x))
train_test['feats'] = train_test['features']
train_test['feats'] = train_test['feats'].apply(lambda x: ' '.join(x))

c_vect = CountVectorizer(stop_words='english', max_features=200, ngram_range=(1, 1))
c_vect.fit(train_test['feats'])

c_vect_sparse_1 = c_vect.transform(train_test['feats'])
c_vect_sparse1_cols = c_vect.get_feature_names()

#manager
def create_top_features(df, top_num, feat_name, count_dict):
    percent = 100
    for i in range(1, top_num):
        df['top_' + str(i) + '_' + feat_name] = df[feat_name].apply(lambda x: 1 if x in count_dict.index.values[count_dict.values >= np.percentile(count_dict.values, percent - i)] else 0)
    return df    
        
managers_count = train_test['manager_id'].value_counts()
train_test = create_top_features(train_test, 5, "manager_id", managers_count)

buildings_count = train_test['building_id'].value_counts()
train_test = create_top_features(train_test, 5, "building_id", buildings_count)

train_test.head()


# In[ ]:





# In[ ]:




