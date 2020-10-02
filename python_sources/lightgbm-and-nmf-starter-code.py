#!/usr/bin/env python
# coding: utf-8

# ### LightGBM and NMF starter code

# Non-Negative Matrix Factorization (NMF) is described well in the paper by [Lee and Seung, 1999][1].
# 
# **Simply Put**
# 
# NMF takes as an input a [term-document matrix][2] and generates a set of topics that represent weighted sets of co-occurring terms. The discovered topics form a basis that provides an efficient representation of the original documents. 
# 
# **About NMF**
# 
# NMF is used for [feature extraction][3] and is generally seen to be useful when there are many attributes, particularly when the attributes are ambiguous or are not strong predictors. By combining attributes NMF can display patterns, topics, or themes which have importance.
# 
# In practice, one encounters NMF typically where text is involved. Consider an example, where the same word (love) in a document could different meanings:
# 
#  1. I *love* lettuce wraps.
#  2. I *love* the way I feel when I'm on vacation in Mexico.
#  3. I *love* my dog, Euclid.
#  4. I *love* being a Data Scientist.
# 
# In all 4 cases, the word 'love' is used, but it has a different meaning to the reader. By combining attributes, NMF introduces context which creates additional predictive power.
# 
# $"love" + "lettuce \ wraps" \ \Rightarrow \ "pleasure \ by \ food"$
# $"love" + "vacation \ in \ Mexico" \ \Rightarrow \ "pleasure \ by \ relaxation"$
# $"love" + "dog" \ \Rightarrow \ "pleasure \ by \ companionship"$
# $"love" + "Data \ Scientist" \ \Rightarrow  \ "pleasure \ by  \ occupation"$
# 
# **How Does It Happen**
# 
# NMF breaks down the multivariate data by creating a user-defined number of features. Each one of these features is a combination of the original attribute set. It is also key to remember these coefficients of these linear combinations are non-negative. 
# 
# Another way to think about it is that NMF breaks your original data features (let's call it V) into the product of two lower ranked matrices (let's call it W and H). NMF uses an iterative approach to modify the initial values of W and H so that the product approaches V. When the approximation error converges or the user-defined number of iterations is reached, NMF terminates.
# 
# **NMF data preparation**
# 
#  - Numeric attributes are normalized. 
#  - Missing numerical values are replaced with the mean. 
#  - Missing categorical values are replaced with the mode.
# 
# It is important to note that outliers can impact NMF significantly. In practice, most Data Scientist use a clipping transformation before binning or normalizing. In addition, NMF in many cases will benefit from normalization. 
# 
# As in many other algorithmic cases, to improve matrix factorization, one needs to decrease the error tolerance (which will increase compute time). 
# 
# 
#   [1]: https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=1&cad=rja&uact=8&ved=0ahUKEwjQvKC6zc_QAhXLMSYKHY9pDVwQFggcMAA&url=http%3A%2F%2Fwww.columbia.edu%2F~jwp2128%2FTeaching%2FW4721%2Fpapers%2Fnmf_nature.pdf&usg=AFQjCNHOf7BKOMfBKKs1wJ2SxSwfj7bgaA
#   [2]: https://en.wikipedia.org/wiki/Document-term_matrix
#   [3]: https://en.wikipedia.org/wiki/Feature_extraction
#   
# Credits:
# [Oleg Panichev](https://www.kaggle.com/opanichev/lightgbm-and-tf-idf-starter/code)

# #### Essential imports

# In[ ]:


import gc
import numpy as np
import pandas as pd
import os

from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.preprocessing import LabelEncoder

from tqdm import tqdm
import lightgbm as lgb


# #### Load data

# In[ ]:


dtype = {
    'id': str,
    'teacher_id': str,
    'teacher_prefix': str,
    'school_state': str,
    'project_submitted_datetime': str,
    'project_grade_category': str,
    'project_subject_categories': str,
    'project_subject_subcategories': str,
    'project_title': str,
    'project_essay_1': str,
    'project_essay_2': str,
    'project_essay_3': str,
    'project_essay_4': str,
    'project_resource_summary': str,
    'teacher_number_of_previously_posted_projects': int,
    'project_is_approved': np.uint8,
}


# In[ ]:


data_path = os.path.join('..', 'input')
train = pd.read_csv(os.path.join(data_path, 'train.csv'), dtype=dtype, low_memory=True)
test = pd.read_csv(os.path.join(data_path, 'test.csv'), dtype=dtype, low_memory=True)
res = pd.read_csv(os.path.join(data_path, 'resources.csv'))


# #### Preprocess data

# In[ ]:


train['project_essay'] = train.apply(lambda row: ' '.join([
    str(row['project_title']),
    str(row['project_essay_1']), 
    str(row['project_essay_2']), 
    str(row['project_essay_3']), 
    str(row['project_essay_4']),
    str(row['project_resource_summary']),]), axis=1)
test['project_essay'] = test.apply(lambda row: ' '.join([
    str(row['project_title']),
    str(row['project_essay_1']), 
    str(row['project_essay_2']), 
    str(row['project_essay_3']), 
    str(row['project_essay_4']),
    str(row['project_resource_summary']),]), axis=1)


# In[ ]:


def extract_features(df):
    df['project_title_len'] = df['project_title'].apply(lambda x: len(str(x)))
    df['project_essay_1_len'] = df['project_essay_1'].apply(lambda x: len(str(x)))
    df['project_essay_2_len'] = df['project_essay_2'].apply(lambda x: len(str(x)))
    df['project_essay_3_len'] = df['project_essay_3'].apply(lambda x: len(str(x)))
    df['project_essay_4_len'] = df['project_essay_4'].apply(lambda x: len(str(x)))
    df['project_resource_summary_len'] = df['project_resource_summary'].apply(lambda x: len(str(x)))


# In[ ]:


extract_features(train)
extract_features(test)


# In[ ]:


train = train.drop([
    'project_essay_1', 
    'project_essay_2', 
    'project_essay_3', 
    'project_essay_4'], axis=1)
test = test.drop([
    'project_essay_1', 
    'project_essay_2', 
    'project_essay_3', 
    'project_essay_4'], axis=1)


# #### Merge with resources

# In[ ]:


df_all = pd.concat([train, test], axis=0)
gc.collect();


# In[ ]:


res = pd.DataFrame(res[['id', 'price']].groupby('id').price.agg(    [
        'count', 
        'sum', 
        'min', 
        'max', 
        'mean', 
        'std', 
        # 'median',
        lambda x: len(np.unique(x)),
    ])).reset_index()
train = train.merge(res, on='id', how='left')
test = test.merge(res, on='id', how='left')
del res
gc.collect();


# #### Preprocess columns with label encoder

# In[ ]:


cols = [
    'teacher_id', 
    'teacher_prefix', 
    'school_state', 
    'project_grade_category', 
    'project_subject_categories', 
    'project_subject_subcategories'
]


# In[ ]:


for c in cols:
    le = LabelEncoder()
    le.fit(df_all[c].astype(str))
    train[c] = le.transform(train[c].astype(str))
    test[c] = le.transform(test[c].astype(str))
del le
gc.collect();


# #### Preprocess timestamp

# In[ ]:


train['project_submitted_datetime'] = pd.to_datetime(train['project_submitted_datetime']).values.astype(np.int64)
test['project_submitted_datetime'] = pd.to_datetime(test['project_submitted_datetime']).values.astype(np.int64)


# #### Preprocess text

# In[ ]:


def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += " ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
        print("-"*50)


# In[ ]:


n_samples = 2000
n_features = 1000
n_components = 10
n_top_words = 20


# In[ ]:


tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2,
                                   max_features=n_features,
                                   stop_words='english')
tfidf = tfidf_vectorizer.fit_transform(df_all["project_essay"])


# ### NMF

# In[ ]:


nmf = NMF(n_components=n_components, random_state=1,
          alpha=.1, l1_ratio=.5).fit(tfidf)
tfidf_feature_names = tfidf_vectorizer.get_feature_names()
print_top_words(nmf, tfidf_feature_names, n_top_words)


# In[ ]:


train_nmf = pd.DataFrame(nmf.transform(tfidf_vectorizer.transform(train.project_essay)),
                         columns=["nmf_%d"%i for i in range(n_components)])
train = pd.concat([train, train_nmf], axis=1)


# In[ ]:


test_nmf = pd.DataFrame(nmf.transform(tfidf_vectorizer.transform(test.project_essay)),
                         columns=["nmf_%d"%i for i in range(n_components)])
test = pd.concat([test, test_nmf], axis=1)


# #### Pepare data

# In[ ]:


cols_to_drop = [
    'id',
    'project_title', 
    'project_essay', 
    'project_resource_summary',
    'project_is_approved',
]
X = train.drop(cols_to_drop, axis=1, errors='ignore')
y = train['project_is_approved']
X_test = test.drop(cols_to_drop, axis=1, errors='ignore')
id_test = test['id'].values
feature_names = list(X.columns)


# In[ ]:


del train, test
gc.collect();


# #### Build the LGB model

# In[ ]:


cnt = 0
p_buf = []
n_splits = 10
n_repeats = 1
kf = RepeatedKFold(
    n_splits=n_splits, 
    n_repeats=n_repeats, 
    random_state=0)
auc_buf = []  


# In[ ]:


for train_index, valid_index in kf.split(X):
    print('Fold {}/{}'.format(cnt + 1, n_splits))
    params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'auc',
        'max_depth': 16,
        'num_leaves': 64,
        'learning_rate': 0.025,
        'verbose': 0,
        'num_threads': 1,
        'lambda_l2': 0.7,
    }  

    
    model = lgb.train(
        params,
        lgb.Dataset(X.loc[train_index], y.loc[train_index], feature_name=feature_names),
        num_boost_round=10000,
        valid_sets=[lgb.Dataset(X.loc[valid_index], y.loc[valid_index])],
        early_stopping_rounds=100,
        verbose_eval=100,
    )

    if cnt == 0:
        importance = model.feature_importance()
        model_fnames = model.feature_name()
        tuples = sorted(zip(model_fnames, importance), key=lambda x: x[1])[::-1]
        tuples = [x for x in tuples if x[1] > 0]
        print('Important features:')
        print(tuples[:50])

    p = model.predict(X.loc[valid_index], num_iteration=model.best_iteration)
    auc = roc_auc_score(y.loc[valid_index], p)

    print('{} AUC: {}'.format(cnt, auc))

    p = model.predict(X_test, num_iteration=model.best_iteration)
    if len(p_buf) == 0:
        p_buf = np.array(p)
    else:
        p_buf += np.array(p)
        
    auc_buf.append(auc)
    cnt += 1

    del model
    gc.collect();


# In[ ]:


auc_mean = np.mean(auc_buf)
auc_std = np.std(auc_buf)
print('AUC = {:.6f} +/- {:.6f}'.format(auc_mean, auc_std))


# In[ ]:


preds = p_buf/cnt
subm = pd.DataFrame()
subm['id'] = id_test
subm['project_is_approved'] = preds
subm.to_csv('submission_nmf.csv', index=False)

