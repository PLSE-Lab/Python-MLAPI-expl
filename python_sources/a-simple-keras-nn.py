#!/usr/bin/env python
# coding: utf-8

# # Working with features

# At first, we will work with data to extract useful features for our model. I will use some methods presented by other users here, too. This kernel is based initially on kernel  Simple Starter Keras NN v2 (https://www.kaggle.com/zeroblue/two-sigma-connect-rental-listing-inquiries/simple-starter-keras-nn-v2) with modifications.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from scipy import sparse
import xgboost as xgb
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.cluster import KMeans
import time
from datetime import timedelta
from sklearn.model_selection import StratifiedKFold
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.regularizers import l2
from keras import optimizers
from keras.callbacks import EarlyStopping
from keras.constraints import max_norm
from keras.layers.advanced_activations import PReLU

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


label_column = 'interest_level'
num_classes = 3

data_path =  "../input/"
train_file = data_path + "train.json"
test_file = data_path + "test.json"
train = pd.read_json(train_file)
test = pd.read_json(test_file)


# In[ ]:


# Make the label numeric
label_map = pd.Series({'low': 2, 'medium': 1, 'high': 0})
train[label_column] = label_map[train[label_column]].values

all_data = train.append(test)
all_data.set_index('listing_id', inplace=True)


# In[ ]:


#Identify bad geographic coordinates
all_data['bad_addr'] = 0
mask = ~all_data['latitude'].between(40.5, 40.9)
mask = mask | ~all_data['longitude'].between(-74.05, -73.7)
bad_rows = all_data[mask]
all_data.loc[mask, 'bad_addr'] = 1


# Here we will create neighborhoods, as explained in https://www.kaggle.com/arnaldcat/two-sigma-connect-rental-listing-inquiries/unsupervised-and-supervised-neighborhood-encoding

# In[ ]:


# Replace bad values with mean
mean_lat = all_data.loc[all_data['bad_addr']==0, 'latitude'].mean()
all_data.loc[all_data['bad_addr']==1, 'latitude'] = mean_lat
mean_long = all_data.loc[all_data['bad_addr']==0, 'longitude'].mean()
all_data.loc[all_data['bad_addr']==1, 'longitude'] = mean_long
kmean_model = KMeans(42)
loc_df = all_data[['longitude', 'latitude']].copy()
standardize = lambda x: (x - x.mean()) / x.std()
loc_df['longitude'] = standardize(loc_df['longitude'])
loc_df['latitude'] = standardize(loc_df['latitude'])
kmean_model.fit(loc_df)
all_data['neighborhoods'] = kmean_model.labels_


# In[ ]:


#distance from center
lat = np.square(all_data['latitude'] - mean_lat)
lng = np.square(all_data['longitude'] - mean_long)
all_data['dist_from_center'] = np.sqrt(lat + lng)


# In[ ]:


#Fix Bathrooms
mask = all_data['bathrooms'] > 9
all_data.loc[mask, 'bathrooms'] = 1


# In[ ]:


#Break up the date data
all_data['created'] = pd.to_datetime(all_data['created'])
#all_data['year'] = all_data['created'].dt.year
all_data['month'] = all_data['created'].dt.month
all_data['day_of_month'] = all_data['created'].dt.day
all_data['weekday'] = all_data['created'].dt.dayofweek
all_data['day_of_year'] = all_data['created'].dt.dayofyear
all_data['hour'] = all_data['created'].dt.hour


# In[ ]:


#Counts
all_data['count_feat'] = all_data['features'].apply(len)
all_data['count_desc'] = all_data['description'].str.split().apply(len)


# In[ ]:


all_data['addr_has_number'] = all_data['display_address'].str.split().str.get(0)
is_digit = lambda x: str(x).isdigit()
all_data['addr_has_number'] = all_data['addr_has_number'].apply(is_digit)


# In[ ]:


#Bed and bath features
all_data['bedrooms'] += 1
all_data['bed_to_bath'] = all_data['bathrooms'] 
all_data['bed_to_bath'] /= all_data['bedrooms']
all_data['price_per_bed'] = all_data['price'] / all_data['bedrooms']
bath = all_data['bathrooms'].copy()
bath.loc[all_data['bathrooms']==0] = 1
all_data['price_per_bath'] = all_data['price'] / bath


# Half bathrooms are not interesting for us (https://www.kaggle.com/arnaldcat/two-sigma-connect-rental-listing-inquiries/a-proxy-for-sqft-and-the-interest-on-1-2-baths/notebook) 

# In[ ]:


all_data['half_bath'] = all_data['bathrooms'] == all_data['bathrooms'] // 1


# In[ ]:


all_data['rooms'] = all_data['bathrooms'] * 0.5 + all_data['bedrooms']
all_data['price_per_room'] = all_data['price'] / all_data['rooms']


# In[ ]:


#Create ratios
median_list = ['bedrooms', 'bathrooms', 'building_id', 'rooms', 'neighborhoods']
for col in median_list:
    median_price = all_data[[col, 'price']].groupby(col)['price'].median()
    median_price = median_price[all_data[col]].values.astype(float)
    all_data['median_' + col] = median_price
    all_data['ratio_' + col] = all_data['price'] / median_price
    all_data['median_' + col] = np.log(all_data['median_' + col].values)

#print('Additional medians and ratios')
median_list = [c for c in all_data.columns if c.startswith('median_')]
all_data['median_mean'] = all_data[median_list].mean(axis=1)
ratio_list = [c for c in all_data.columns if c.startswith('ratio_')]
all_data['ratio_mean'] = all_data[ratio_list].mean(axis=1)


# In[ ]:


#Normalize the price
all_data['price'] = np.log(all_data['price'].values)


# In[ ]:


#Building counts
bldg_count = all_data['building_id'].value_counts()
bldg_count['0'] = 0
all_data['bldg_count'] = np.log1p(bldg_count[all_data['building_id']].values)
all_data['zero_bldg'] = all_data['building_id']=='0'


# In[ ]:


lbl = preprocessing.LabelEncoder()
lbl.fit(list(all_data['manager_id'].values))
all_data['manager_id'] = lbl.transform(list(all_data['manager_id'].values))


# In[ ]:


temp = pd.concat([all_data[all_data['interest_level'].isnull() == False].manager_id,pd.get_dummies(all_data[all_data['interest_level'].isnull() == False].interest_level)], axis = 1).groupby('manager_id').mean()
temp.columns = ['high_frac','low_frac', 'medium_frac']
temp['count'] = all_data[all_data['interest_level'].isnull() == False].groupby('manager_id').count().iloc[:,1]


# In[ ]:


temp.tail(10)


# In[ ]:


# compute skill
temp['manager_skill'] = temp['high_frac']*2 + temp['medium_frac']

# get ixes for unranked managers...
unranked_managers_ixes = temp['count']<20
# ... and ranked ones
ranked_managers_ixes = ~unranked_managers_ixes

# compute mean values from ranked managers and assign them to unranked ones
mean_values = temp.loc[ranked_managers_ixes, ['high_frac','low_frac', 'medium_frac','manager_skill']].mean()
print(mean_values)
temp.loc[unranked_managers_ixes,['high_frac','low_frac', 'medium_frac','manager_skill']] = mean_values.values
print(temp.tail(10))


# In[ ]:


# inner join to assign manager features to the managers in the training dataframe
all_data = all_data.merge(temp.reset_index(),how='left', left_on='manager_id', right_on='manager_id', right_index=True)
all_data.head()


# In[ ]:


def remap_skill(x):
    m_id = x['manager_id_x']
    skill = temp[temp.index == m_id]['manager_skill']
    if len(skill) > 0:
        x['manager_skill'] = skill.values[0]
    else:
        x['manager_skill'] = 0
    return x

all_data = all_data.apply(remap_skill, axis=1)


# In[ ]:


all_data.drop(['manager_id_y', 'high_frac', 'low_frac', 'medium_frac', 'count'], axis=1, inplace=True)


# In[ ]:


all_data.info()


# In[ ]:


#Scale features
scaler = StandardScaler()
cols = [c for c in all_data.columns]
scale_keywords = ['price', 'count', 'ratio', '_to_', 
                  'day_', 'hour', 'median', 'longitude', 'latitude']
scale_list = [c for c in cols if any(w in c for w in scale_keywords)]
print('Scaling features:', scale_list)
all_data[scale_list] = scaler.fit_transform(all_data[scale_list].astype(float))


# In[ ]:


#Create dummies
mask = all_data['bathrooms'] > 3
all_data.loc[mask, 'bathrooms'] = 4
mask = all_data['bedrooms'] >= 5
all_data.loc[mask, 'bedrooms'] = 5
mask = all_data['rooms'] >= 6
all_data.loc[mask, 'rooms'] = 6
cat_cols = ['bathrooms', 'bedrooms', 'month', 'weekday', 'rooms', 
            'neighborhoods']
for col in cat_cols:
    dummy = pd.get_dummies(all_data[col], prefix=col)
    dummy = dummy.astype(bool) 
    all_data = all_data.join(dummy)
all_data.drop(cat_cols, axis=1, inplace=True)


# In[ ]:


#Drop columns
drop_cols = ['description', 'photos', 'display_address', 'street_address', 
             'features', 'created', 'building_id', 'manager_id_x', 
             'longitude', 'latitude'
             ]
             
all_data.drop(drop_cols, axis=1, inplace=True)


# In[ ]:


data_columns = all_data.columns.tolist()
data_columns.remove(label_column)

mask = all_data[label_column].isnull()
train = all_data[~mask].copy()
test = all_data[mask].copy()


# In[ ]:


folds = 5
kf = StratifiedKFold(folds, shuffle=True, random_state=42)
kf = list(kf.split(train, train[label_column]))

train_idx, val_idx = kf[0]
train_cv = train.iloc[train_idx][data_columns].values
train_cv_labels = train.iloc[train_idx][label_column].values
val_cv = train.iloc[val_idx][data_columns].values
val_cv_labels = train.iloc[val_idx][label_column].values


# In[ ]:


def nn_model():
    model = Sequential()
    model.add(Dense(128,  
                    activation='softplus',
                    kernel_initializer='he_normal',
                    kernel_regularizer=l2(0.000025),
                    kernel_constraint=max_norm(2.0),
                    input_shape = (len(data_columns),),))
    #model.add(Dropout(0.25))
    
    model.add(PReLU())
    
    model.add(Dense(64,  
                    activation='softplus',
                    kernel_initializer='he_normal',
                    kernel_regularizer=l2(0.000025),
#                    kernel_constraint=max_norm(2.0),
                    ))    
    model.add(Dropout(0.25))
    
    model.add(Dense(16, 
                    activation='softplus', 
                    kernel_initializer='he_normal',
                    kernel_regularizer=l2(0.000025),
                    #kernel_constraint=max_norm(2.0)
                    ))
#    model.add(Dropout(0.1))
    
    model.add(Dense(32,
                    activation='softplus', 
                    kernel_initializer='he_normal',
                    kernel_regularizer=l2(0.00005),
                    kernel_constraint=max_norm(2.0)
                    ))
    model.add(Dropout(0.25))

    model.add(Dense(units=num_classes, 
                    activation='softmax', 
                    kernel_initializer='he_normal',
                    ))
    opt = optimizers.Adadelta(lr=1)
    model.compile(loss='sparse_categorical_crossentropy', 
                  optimizer=opt,
                  metrics=['accuracy']
                  )
    return(model)

model = nn_model()


# In[ ]:


early_stopping = EarlyStopping(monitor='val_loss', patience=50)

model.fit(train_cv, train_cv_labels, epochs = 400, batch_size=512, verbose = 2, 
          validation_data=[val_cv, val_cv_labels], callbacks=[early_stopping])
val_pred = model.predict_proba(val_cv)
score = log_loss(val_cv_labels, val_pred)
print('Score:', score)


# In[ ]:


test.head()


# In[ ]:


test_pred = model.predict_proba(test[data_columns].values)


# In[ ]:


test_pred


# In[ ]:


test_out = pd.DataFrame(test_pred, columns = ['high', 'medium', 'low'], index=test.index)
test_out.head()


# In[ ]:


test_out.to_csv('submissions.csv')


# In[ ]:




