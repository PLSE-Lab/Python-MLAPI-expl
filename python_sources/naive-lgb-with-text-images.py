#!/usr/bin/env python
# coding: utf-8

# This kernel is based on work from Beep-beep: https://www.kaggle.com/the1owl/beep-beep

# In[44]:


from nltk.corpus import stopwords
from datetime import datetime
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
import gc
from scipy import sparse
import gzip
from sklearn.decomposition import TruncatedSVD
from pathlib import PurePath
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVR
from scipy import sparse

get_ipython().run_line_magic('matplotlib', 'inline')


# In[45]:


target_col ='deal_probability'
toy = False # Activate for debug proposes
validate = True
best_num_boost_round = 100 if toy else 5000
num_boost_round = 100 if toy else 5000


# In[46]:


get_ipython().run_cell_magic('time', '', 'dtypes = {\n    \'user_id\': \'category\',\n    \'region\': \'category\',\n    \'city\': \'category\',\n    \'parent_category_name\': \'category\',\n    \'category_name\': \'category\',\n    \'param_1\': \'category\',\n    \'param_2\': \'category\',\n    \'param_3\': \'category\',\n    \'title\': \'str\',\n    \'description\': \'str\',\n    \'price\': \'float\',\n    \'item_seq_number\': \'int\',\n    \'activation_date\': \'object\',\n    \'user_type\': \'category\',\n    \'image\': \'str\',\n    \'image_top_1\': \'float\',\n    \'deal_probability\': \'float\'\n}\ndate_cols = [\'activation_date\']\n\n# Replace category by \'object\' for easier join of train and test\ndtypes_load = {k:(\'object\' if v == \'category\' else v) for k, v in dtypes.items()}\n\ndf_train = pd.read_csv(\'../input/avito-demand-prediction/train.csv\', dtype=dtypes_load, parse_dates=date_cols, index_col="item_id", nrows=100000 if toy else None)\ndf_test = pd.read_csv(\'../input/avito-demand-prediction/test.csv\', dtype=dtypes_load, parse_dates=date_cols, index_col="item_id", nrows=100000 if toy else None)\n\nn_train = df_train.shape[0]')


# ## Image features

# In[47]:


def load_imfeatures(folder):
    path = PurePath(folder)
    features = sparse.load_npz(str(path / 'features.npz'))
    
    if toy:
        features = features[:100000]
        
    return features


# In[48]:


ftrain = load_imfeatures('../input/vgg16-train-features/')


# In[49]:


ftest = load_imfeatures('../input/vgg16-test-features/')


# In[50]:


assert df_train.shape[0] == ftrain.shape[0]
assert df_test.shape[0] == ftest.shape[0]


# In[51]:


# Create both dataframe
df_target = df_train[target_col]
df_both = pd.concat([df_train, df_test])

del df_train, df_test
gc.collect();


# In[52]:


fboth = sparse.vstack([ftrain, ftest])
del ftrain, ftest
gc.collect()
fboth.shape


# In[ ]:


# Categorical image feature (max and min VGG16 feature)
df_both['im_max_feature'] = fboth.argmax(axis=1)  # This will be categorical
df_both['im_min_feature'] = fboth.argmin(axis=1)  # This will be categorical

df_both['im_n_features'] = fboth.getnnz(axis=1)
df_both['im_mean_features'] = fboth.mean(axis=1)
df_both['im_meansquare_features'] = fboth.power(2).mean(axis=1)


# In[ ]:


get_ipython().run_cell_magic('time', '', '# Let`s reduce 512 VGG16 featues into 32\ntsvd = TruncatedSVD(32)\nftsvd = tsvd.fit_transform(fboth)\ndel fboth\ngc.collect()')


# ## Feature engineering

# In[ ]:


# Convert df_both categorical cols to 'category' type
for col, dtype in dtypes.items():
    if dtype == 'category':
        df_both[col] = df_both[col].astype('category').cat.codes


# In[ ]:


df_both.dtypes


# In[ ]:


cat_cols = ['region', 'city', 'parent_category_name', 'category_name', 'param_1', 'param_2', 'param_3', 'user_type',
           'im_max_feature', 'im_min_feature']
num_cols = ['price', 'image_top_1', 'deal_probability']

for cat_col in cat_cols:
    df_group = df_both.groupby(cat_col)[num_cols].agg(['mean', 'std'])
    df_group.columns = ['{}_'.format(cat_col) + '-'.join(col).strip() for col in df_group.columns.values]

    df_both = df_both.join(df_group, on=cat_col, how='left')
    del df_group
    print(cat_col)
    
# del df_train
# gc.collect()


# In[ ]:


df_both['activation_dow'] = df_both['activation_date'].dt.dayofweek


# In[ ]:


# Text processing
sw = stopwords.words('russian')
txt_cols = ['title', 'description']

for txt_col in txt_cols:
    df_both[txt_col + '_len'] = df_both[txt_col].str.len()
    df_both[txt_col + '_wc'] = df_both[txt_col].str.count(' ')
    
    if txt_col != 'description':
        feature_cnt = 50

        tfidf = TfidfVectorizer(stop_words=sw, min_df=10, max_df=0.8, dtype=np.float32, max_features=32000)
        X_text = tfidf.fit_transform(df_both[txt_col])
        svr = LinearSVR(C=0.01).fit(X_text[:n_train], df_target)
        fnames = sorted(tfidf.vocabulary_, key=tfidf.vocabulary_.__getitem__)
        
        best_features = np.argsort(svr.coef_)
        
        # Select most positive and negative features
        selected_features = np.concatenate([best_features[:feature_cnt], best_features[-feature_cnt:]])
        features_names = list(map(fnames.__getitem__, selected_features))
        
        df_both_tfidf = pd.DataFrame(X_text[:, selected_features].todense(), columns=features_names, index=df_both.index)
        
        del X_text, features_names, selected_features, best_features, svr, fnames, tfidf

        df_both = df_both.join(df_both_tfidf)
        
        del df_both_tfidf
        gc.collect()
        
    print(txt_col)


# In[ ]:


# Merge image features into df_both
df_ftsvd = pd.DataFrame(ftsvd, index=df_both.index).add_prefix('im_tsvd_')

df_both = pd.concat([df_both, df_ftsvd], axis=1)

del df_ftsvd, ftsvd
gc.collect();


# In[ ]:


# Split df_both in train and test
df_train = df_both.iloc[:n_train]
df_test = df_both.iloc[n_train:]

del df_both
gc.collect()

df_train.shape, df_test.shape


# In[ ]:


ex_cols = {'item_id', 'user_id', 'deal_probability', 'title', 'description', 'image', 'activation_date'}
used_cols = [c for c in df_train.columns if c not in ex_cols]
print('Used cols:', ', '.join(used_cols))


# In[ ]:


if validate:
    fold_train, fold_valid, target_train, target_valid = train_test_split(df_train[used_cols], df_target, test_size=0.2, random_state=42)

    dtrain = lgb.Dataset(fold_train, target_train, categorical_feature=cat_cols)
    dvalid = lgb.Dataset(fold_valid, target_valid, categorical_feature=cat_cols)
    
    valid_sets = [dvalid]
    valid_names = ['valid']
    
    del fold_train, fold_valid, target_train, target_valid
else:
    dtrain = lgb.Dataset(df_train[used_cols], df_target)
    valid_sets = [dtrain]
    valid_names = ['train']
    assert best_num_boost_round is not None


# In[ ]:


get_ipython().run_cell_magic('time', '', "# LGB train\nparams = {\n    'learning_rate': 0.02,\n    'boosting': 'gbdt',\n    'objective': 'regression',\n    'metric': ['rmse'],\n    'is_training_metric': True,\n    'seed': 19,\n    'num_leaves': 31,\n    'feature_fraction': 0.9,\n    'bagging_fraction': 0.8,\n    'bagging_freq': 5\n}\n\nmodel = lgb.train(params, dtrain, num_boost_round=num_boost_round, valid_sets=valid_sets, valid_names=valid_names,\n                  verbose_eval=num_boost_round//20, early_stopping_rounds=50 if validate else None)\n\nif validate:\n    best_num_boost_round = model.best_iteration")


# In[ ]:


fig, ax = plt.subplots(figsize=(8, 25))
lgb.plot_importance(model, ax=ax);


# In[ ]:


print(best_num_boost_round)


# In[ ]:


del df_train, dtrain
gc.collect()


# In[ ]:


df_test.index.name = 'item_id'
df_test['deal_probability'] = model.predict(df_test[used_cols], num_iteration=best_num_boost_round).clip(0., 1.)
df_test[['deal_probability']].to_csv('submission.csv', index=True)


# In[ ]:


get_ipython().system('head submission.csv')


# In[ ]:


df_test.shape

