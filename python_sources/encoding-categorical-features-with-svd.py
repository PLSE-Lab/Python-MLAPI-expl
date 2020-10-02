#!/usr/bin/env python
# coding: utf-8

# Setting up a few things first, then I'll get into how to encode categorical features with singular value decomposition (SVD).

# In[ ]:


import pandas as pd
from sklearn.preprocessing import LabelEncoder

ks = pd.read_csv('../input/kickstarter-projects/ks-projects-201801.csv',
                 parse_dates=['deadline', 'launched'])

# Drop live projects
ks = ks.query('state != "live"')

# Add outcome column, "successful" == 1, others are 0
ks = ks.assign(outcome=(ks['state'] == 'successful').astype(int))

# Timestamp features
ks = ks.assign(hour=ks.launched.dt.hour,
               day=ks.launched.dt.day,
               month=ks.launched.dt.month,
               year=ks.launched.dt.year)

# Label encoding
cat_features = ['category', 'currency', 'country']
encoder = LabelEncoder()
encoded = ks[cat_features].apply(encoder.fit_transform)

data_cols = ['goal', 'hour', 'day', 'month', 'year', 'outcome']
baseline = ks[data_cols].join(encoded)


# In[ ]:


# Defining  functions that will help us test our encodings
import lightgbm as lgb
from sklearn import metrics

def get_data_splits(dataframe, valid_fraction=0.1):
    valid_fraction = 0.1
    valid_size = int(len(dataframe) * valid_fraction)

    train = dataframe[:-valid_size * 2]
    # valid size == test size, last two sections of the data
    valid = dataframe[-valid_size * 2:-valid_size]
    test = dataframe[-valid_size:]
    
    return train, valid, test

def train_model(train, valid):
    feature_cols = train.columns.drop('outcome')

    dtrain = lgb.Dataset(train[feature_cols], label=train['outcome'])
    dvalid = lgb.Dataset(valid[feature_cols], label=valid['outcome'])

    param = {'num_leaves': 64, 'objective': 'binary', 
             'metric': 'auc', 'seed': 7}
    print("Training model!")
    bst = lgb.train(param, dtrain, num_boost_round=1000, valid_sets=[dvalid], 
                    early_stopping_rounds=10, verbose_eval=False)

    valid_pred = bst.predict(valid[feature_cols])
    valid_score = metrics.roc_auc_score(valid['outcome'], valid_pred)
    print(f"Validation AUC score: {valid_score:.4f}")
    return bst


# In[ ]:


# Training a model on the baseline data
train, valid, _ = get_data_splits(baseline)
bst = train_model(train, valid)


# # Encoding with Singular Value Decomposition
# 
# Here I'll use singular value decomposition (SVD) to learn encodings from pairs of categorical features. SVD is one of the more complex encodings, but it can also be very effective. We'll construct a matrix of co-occurences for each pair of categorical features. Each row corresponds to a value in feature A, while each column corresponds to a value in feature B. Each element is the count of rows where the value in A appears together with the value in B.
# 
# You then use singular value decomposition to find two smaller matrices that equal the count matrix when multiplied.
# 
# <center><img src="https://i.imgur.com/mnnsBKJ.png" width=600px></center>
# 
# You can choose how long each factor vector will be. Longer vectors will contain more information at the cost of more memory/computation. To get the encodings for feature A, you multiply the count matrix by the small matrix for feature B.
# 
# I'll show you how you can do this for one pair of features using scikit-learn's `TruncatedSVD` class.

# In[ ]:


from sklearn.decomposition import TruncatedSVD

# Use 3 components in the latent vectors
svd = TruncatedSVD(n_components=3)


# First we can use `.groupby` to count up co-occurences for any pair of features.

# In[ ]:


train, valid, _ = get_data_splits(baseline)

# Create a sparse matrix with cooccurence counts
pair_counts = train.groupby(['country', 'category'])['outcome'].count()
pair_counts.head(10)


# Now we have a series with a two-level index. We want to convert this into a matrix with `country` on one axis and `category` on the other. To do this, we can use `.unstack`. By default it'll put `NaN`s where data doesn't exist, but we can tell it to fill those spots with zeros.

# In[ ]:


pair_matrix = pair_counts.unstack(fill_value=0)
pair_matrix


# In[ ]:


svd_encoding = pd.DataFrame(svd.fit_transform(pair_matrix))
svd_encoding.head(10)


# This gives us a mapping of the values in the country feature, the index of the dataframe, to our encoded vectors. Next, we need to replace the values in our data with these vectors. We can do this using the `.reindex` method. This method takes the values in the country column and creates a new dataframe from from `svd_encoding` using those values as the index. Then we need to set the index back to the original index. Note that I learned the encodings from the training data, but I'm applying them to the whole dataset.

# In[ ]:


encoded = svd_encoding.reindex(baseline['country']).set_index(baseline.index)
encoded.head(10)


# In[ ]:


# Join encoded feature to the dataframe, with info in the column names
data_svd = baseline.join(encoded.add_prefix("country_category_svd_"))
data_svd.head()


# In[ ]:


train, valid, _ = get_data_splits(data_svd)
bst = train_model(train, valid)


# The baseline score was 0.7467, while we get a slight improvement to 0.7472 with the SVD encodings.
