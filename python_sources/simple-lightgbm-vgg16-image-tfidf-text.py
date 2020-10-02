#!/usr/bin/env python
# coding: utf-8

# This kernel is based on the following notes.
# 
# https://www.kaggle.com/sudalairajkumar/simple-exploration-baseline-notebook-avito
# 
# https://www.kaggle.com/bguberfain/naive-lgb-with-text-images/

# **Import Libraries:**

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn import preprocessing, model_selection, metrics
import lightgbm as lgb

from nltk.corpus import stopwords
from pathlib import PurePath
from scipy import sparse
import gc
import gzip

get_ipython().run_line_magic('matplotlib', 'inline')

pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 999


# **Read Data:**

# In[ ]:


train_df = pd.read_csv("../input/avito-demand-prediction/train.csv", parse_dates=["activation_date"])
test_df = pd.read_csv("../input/avito-demand-prediction/test.csv", parse_dates=["activation_date"])


# ** Image Features:**

# In[ ]:


### Image features ###
def load_imfeatures(folder):
    path = PurePath(folder)
    features = sparse.load_npz(str(path / 'features.npz'))
    return features

ftrain = load_imfeatures('../input/vgg16-train-features/')
ftest = load_imfeatures('../input/vgg16-test-features/')


### Create both dataframe ###
df_both = pd.concat([train_df, test_df])

fboth = sparse.vstack([ftrain, ftest])
del ftrain, ftest
gc.collect()
fboth.shape

### Categorical image feature (max and min VGG16 feature) ###
df_both['im_max_feature'] = fboth.argmax(axis=1)  # This will be categorical
df_both['im_min_feature'] = fboth.argmin(axis=1)  # This will be categorical

df_both['im_n_features'] = fboth.getnnz(axis=1)
df_both['im_mean_features'] = fboth.mean(axis=1)
df_both['im_meansquare_features'] = fboth.power(2).mean(axis=1)

### Let`s reduce 512 VGG16 featues into 32 ###
tsvd = TruncatedSVD(32)
ftsvd = tsvd.fit_transform(fboth)
del fboth
gc.collect()

### Merge image features into df_both ###
df_ftsvd = pd.DataFrame(ftsvd, index=df_both.index).add_prefix('im_tsvd_')

df_both = pd.concat([df_both, df_ftsvd], axis=1)

del df_ftsvd, ftsvd
gc.collect();

###Split df_both in train and test ###
n_train = train_df.shape[0]
train_df = df_both.iloc[:n_train]
test_df = df_both.iloc[n_train:]

del df_both
gc.collect()


# ** "title" Feature:**

# In[ ]:


### Stop Words ###
russian_stop = set(stopwords.words('russian'))

### TFIDF Vectorizer ###
tfidf_vec = TfidfVectorizer(ngram_range=(1,1), stop_words=russian_stop)
full_tfidf = tfidf_vec.fit_transform(train_df['title'].values.tolist() + test_df['title'].values.tolist())
train_tfidf = tfidf_vec.transform(train_df['title'].values.tolist())
test_tfidf = tfidf_vec.transform(test_df['title'].values.tolist())

### SVD Components ###
n_comp = 3
svd_obj = TruncatedSVD(n_components=n_comp, algorithm='arpack')
svd_obj.fit(full_tfidf)
train_svd = pd.DataFrame(svd_obj.transform(train_tfidf))
test_svd = pd.DataFrame(svd_obj.transform(test_tfidf))
train_svd.columns = ['svd_title_'+str(i+1) for i in range(n_comp)]
test_svd.columns = ['svd_title_'+str(i+1) for i in range(n_comp)]
train_df = pd.concat([train_df, train_svd], axis=1)
test_df = pd.concat([test_df, test_svd], axis=1)
del full_tfidf, train_tfidf, test_tfidf, train_svd, test_svd


# ** "description" Feature:**

# In[ ]:


### Filling missing values ###
train_df["description"].fillna("NA", inplace=True)
test_df["description"].fillna("NA", inplace=True)

### TFIDF Vectorizer ###
tfidf_vec = TfidfVectorizer(ngram_range=(1,1), max_features=100000, stop_words=russian_stop)
full_tfidf = tfidf_vec.fit_transform(train_df['description'].values.tolist() + test_df['description'].values.tolist())
train_tfidf = tfidf_vec.transform(train_df['description'].values.tolist())
test_tfidf = tfidf_vec.transform(test_df['description'].values.tolist())

### SVD Components ###
n_comp = 3
svd_obj = TruncatedSVD(n_components=n_comp, algorithm='arpack')
svd_obj.fit(full_tfidf)
train_svd = pd.DataFrame(svd_obj.transform(train_tfidf))
test_svd = pd.DataFrame(svd_obj.transform(test_tfidf))
train_svd.columns = ['svd_desc_'+str(i+1) for i in range(n_comp)]
test_svd.columns = ['svd_desc_'+str(i+1) for i in range(n_comp)]
train_df = pd.concat([train_df, train_svd], axis=1)
test_df = pd.concat([test_df, test_svd], axis=1)
del full_tfidf, train_tfidf, test_tfidf, train_svd, test_svd


# ** "activation_date" Feature:**

# In[ ]:


### Date Variables ###
train_df["activation_weekday"] = train_df["activation_date"].dt.weekday
train_df["activation_weekofyear"] = train_df["activation_date"].dt.week
train_df["activation_weekofmonth"] = train_df["activation_date"].dt.day
test_df["activation_weekday"] = test_df["activation_date"].dt.weekday
test_df["activation_weekofyear"] = test_df["activation_date"].dt.week
test_df["activation_weekofmonth"] = test_df["activation_date"].dt.day


# ** Features Selection: **

# In[ ]:


### Target and ID variables ###
train_y = train_df["deal_probability"].values
test_id = test_df["item_id"].values
    
### Label encode the categorical variables ###
cat_vars = [ "region", "city", "parent_category_name", "category_name", "user_type", "param_1", "param_2", "param_3"]
for col in cat_vars:
    lbl = preprocessing.LabelEncoder()
    lbl.fit(list(train_df[col].values.astype('str')) + list(test_df[col].values.astype('str')))
    train_df[col] = lbl.transform(list(train_df[col].values.astype('str')))
    test_df[col] = lbl.transform(list(test_df[col].values.astype('str')))

cols_to_drop = ["item_id", "user_id", "description", "title", "activation_date", "image"]
train_X = train_df.drop(cols_to_drop + ["deal_probability"], axis=1)
test_X = test_df.drop(cols_to_drop, axis=1)


# ** LightGBM Model: **

# In[ ]:


### LightGBM model ###
def run_lgb(train_X, train_y, val_X, val_y, test_X):

    params = {
        "objective" : "regression",
        "metric" : "rmse",
        "num_leaves" : 30,
        "learning_rate" : 0.1,
        "bagging_fraction" : 0.7,
        "feature_fraction" : 0.7,
        "bagging_frequency" : 5,
        "bagging_seed" : 2018,
        "verbosity" : -1
    }

    
    lgtrain = lgb.Dataset(train_X, label=train_y)
    lgval = lgb.Dataset(val_X, label=val_y)
    evals_result = {}
    model = lgb.train(params, lgtrain, 5000, valid_sets=[lgval], early_stopping_rounds=100, verbose_eval=20, evals_result=evals_result)
    
    pred_test_y = model.predict(test_X, num_iteration=model.best_iteration)
    return pred_test_y, model, evals_result


# ** Model Training: **

# In[ ]:


### Splitting the data for model training ###
dev_X = train_X.iloc[:-200000,:]
val_X = train_X.iloc[-200000:,:]
dev_y = train_y[:-200000]
val_y = train_y[-200000:]
print(dev_X.shape, val_X.shape, test_X.shape)

### Training the model ###
pred_test, model, evals_result = run_lgb(dev_X, dev_y, val_X, val_y, test_X)

### Making a submission file ###
pred_test[pred_test>1] = 1
pred_test[pred_test<0] = 0
sub_df = pd.DataFrame({"item_id":test_id})
sub_df["deal_probability"] = pred_test
sub_df.to_csv("baseline_lgb.csv", index=False)


# ** Feature Importance : **
# 
# 

# In[ ]:


fig, ax = plt.subplots(figsize=(12,18))
lgb.plot_importance(model, max_num_features=50, height=0.8, ax=ax)
ax.grid(False)
plt.title("LightGBM - Feature Importance", fontsize=15)
plt.show()

