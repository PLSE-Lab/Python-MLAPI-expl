#!/usr/bin/env python
# coding: utf-8

# # Same data in train and test
# 
# The aim of this notebook is to investigate the amount of data that is present both in train and test datasets.

# In[ ]:


import numpy as np 
import pandas as pd 


# ### Read the data

# In[ ]:


data_path = "../input/"
train_file = data_path + "train.json"
test_file = data_path + "test.json"
train_df = pd.read_json(train_file)
test_df = pd.read_json(test_file)

train_df = train_df.fillna('')
test_df = test_df.fillna('')

train_df['photos_num'] = train_df.photos.apply(lambda x: len(x))
test_df['photos_num'] = test_df.photos.apply(lambda x: len(x))

train_df['features_num'] = train_df.features.apply(lambda x: len(x))
test_df['features_num'] = test_df.features.apply(lambda x: len(x))

print('Shape of train dataset = ' + str(train_df.shape))
print('Shape of test dataset = ' + str(test_df.shape))


# ### Now we will try to find similar and the same data in train and test sets.

# In[ ]:


def find_idx_ainb(a, b, cols):
    if len(cols) == 0:
        return []
    
    ainb = a[cols[0]].isin(b[cols[0]].values)
    if len(cols) > 1:
        for i in range(1, len(cols)):
            ainb &= a[cols[i]].isin(b[cols[i]].values)
            
    return ainb


# If we consider that same data means same values in 'bathrooms', 'bedrooms', 'building_id' and 'description' columns, we obtain next results:

# In[ ]:


cols = ['bathrooms', 'bedrooms', 'building_id', 'description']
print(cols)
idx = find_idx_ainb(train_df, test_df, cols)
print('\nPercent of train data in test = {0:.2f}%'.format(100*np.mean(idx)))


# **36.58%** of train data is present within the test set! I haven't expected such a big percentage. It seems that there are many same apartments in the dataset. Let's add few more parameters. 

# In[ ]:


cols = list(train_df.columns.values)
cols.remove('created')
cols.remove('features')
cols.remove('interest_level')
cols.remove('listing_id')
cols.remove('photos')
print(cols)

idx = find_idx_ainb(train_df, test_df, cols)
print('\nPercent of train data in test = {0:.2f}%'.format(100*np.mean(idx)))


# Wow! **34.64%** of train data is present in the test dataset, but possibly with a different date of creation. 
# 
# Now let's check if there are absolutely the same apartment descriptions.

# In[ ]:


cols = list(train_df.columns.values)
cols.remove('features')
cols.remove('interest_level')
cols.remove('listing_id')
cols.remove('photos')
print(cols)

idx = find_idx_ainb(train_df, test_df, cols)
print('\nPercent of train data in test = {0:.2f}%'.format(100*np.mean(idx)))


# **1.75%** of train data is the same as in the test (didn't check the photos and features, but number of photos and features the same as well). 
# 
# So many same descriptions present both in train and test datasets was found. Now we can find out if 'created' feature is a really important feature. For this purpose let's use sample_submission.csv, which gives **0.78993** score on the leaderboard, and insert in this submission known interest levels for the data present in the train set. 34.64% of train set it is around 17000 samples and approximately 23% of test data.

# ## Generate submission 

# In[ ]:


cols = ['bathrooms', 'bedrooms', 'building_id',         'description', 'display_address', 'latitude',         'longitude', 'manager_id', 'price', 'street_address',         'photos_num', 'features_num']
df_merged = pd.merge(train_df, test_df,                      on=cols,                      suffixes=('_train', '_test'), how='right')
df_merged = df_merged.rename(columns={'listing_id_test': 'listing_id'})
df_merged.head()


# In[ ]:


fname = 'sample_submission.csv'
subm = pd.read_csv(data_path + fname)
subm = subm.merge(df_merged[['listing_id','interest_level']], on='listing_id')


# Just to be sure let's check if there are any duplicated descriptions (same description with different listing_id).

# In[ ]:


print('Number of duplicates = ' + str(np.sum(subm.duplicated(subset='listing_id'))))


# In[ ]:


subm.sort_values('listing_id').loc[subm.duplicated(subset='listing_id', keep=False)].head(10)


# Surprise again. There are many same descriptions within the train set. To confirm that let's do next:

# In[ ]:


print('Number of duplicates in train = ' +       str(np.sum(train_df.duplicated(subset=cols, keep=False))))
print('Number of duplicates in test = ' +       str(np.sum(test_df.duplicated(subset=cols, keep=False))))


# Yeap, there are the same descriptions withing the train data but with different date of creation. Moreover, the same situation is with the test set. Anyway, let's create a modified submission basing on averaging class probabilities for duplicates.

# In[ ]:


subm.low.loc[subm.interest_level=='low'] = 1.0
subm.medium.loc[subm.interest_level=='low'] = 0.0
subm.high.loc[subm.interest_level=='low'] = 0.0

subm.low.loc[subm.interest_level=='medium'] = 0.0
subm.medium.loc[subm.interest_level=='medium'] = 1.0
subm.high.loc[subm.interest_level=='medium'] = 0.0

subm.low.loc[subm.interest_level=='high'] = 0.0
subm.medium.loc[subm.interest_level=='high'] = 0.0
subm.high.loc[subm.interest_level=='high'] = 1.0

subm = subm.groupby('listing_id').mean()

print('subm.shape = ' + str(subm.shape))
subm.head()


# In[ ]:


subm.to_csv('submission.csv', index=True)


# ## Conclusions
# 
# From this investigation it was found:
# 
#  - There are the same or very similar descriptions present both in the train and test datasets;
#  - There are the same descriptions but with different 'created' value within the train and test sets;
#  - Duplicates within the train set may have different interest level, so it may be a sign that 'created' feature may be very important or dataset does not contain enough features.
#  -  Resulting submission gives around **2.0** on the leaderboard, what is much worse then initial sample submission. It means that descriptions in test set with almost the same features may have absolutely different interest level, which may be classified correctly only by 'looking' at 'created' feature.
