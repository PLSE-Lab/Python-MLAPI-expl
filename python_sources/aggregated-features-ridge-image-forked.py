#!/usr/bin/env python
# coding: utf-8

# I found some interesting [public](http://https://www.kaggle.com/him4318/avito-lightgbm-with-ridge-feature-v-2-0/code) [kernals](http://). I tried to fork one of the kernals cause I found the aggregated feature file from [this](http://https://www.kaggle.com/bminixhofer/aggregated-features-lightgbm) one is quite useful. Here, I'll try to add image blurrness score features I extracted from one of my other [kernals](http://https://www.kaggle.com/sukhyun9673/image-processing-600000-to-750000). (Later, I'll also try adding [regional information I scraped from Wikipedia.](http://https://www.kaggle.com/sukhyun9673/scraping-regional-info-population-time-zone-etc))
# 
# This kernal includes forks and reference from : https://www.kaggle.com/him4318/avito-lightgbm-with-ridge-feature-v-2-0/code, and https://www.kaggle.com/bminixhofer/aggregated-features-lightgbm
# Thanks to original authors for inspiration!

# In[1]:


import pandas as pd
import numpy as np
import gc
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib_venn import venn2, venn2_circles
import string
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import nltk
from nltk.corpus import stopwords
import scipy
import lightgbm as lgb
from sklearn.linear_model import Ridge
from sklearn.cross_validation import KFold
from math import sqrt

sns.set()
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


train = pd.read_csv('../input/avito-demand-prediction/train.csv')
test = pd.read_csv('../input/avito-demand-prediction/test.csv')

train_sorted = train.sort_values(by = ["image"])
test_sorted = test.sort_values(by = ["image"])

#Load image blurrness for train

tr_1 = pd.read_csv("../input/train-data-image-blurrness/1.csv")
tr_2 = pd.read_csv("../input/train-data-image-blurrness/2.csv")
tr_3 = pd.read_csv("../input/train-data-image-blurrness/3.csv")
tr_4 = pd.read_csv("../input/train-data-image-blurrness/4.csv")
tr_5 = pd.read_csv("../input/train-data-image-blurrness/5.csv")
tr_6 = pd.read_csv("../input/train-data-image-blurrness/6.csv")
tr_7 = pd.read_csv("../input/train-data-image-blurrness/7.csv")
tr_8 = pd.read_csv("../input/train-data-image-blurrness/8.csv")
tr_9 = pd.read_csv("../input/train-data-image-blurrness/9.csv")
tr_10 = pd.read_csv("../input/train-data-image-blurrness/10.csv")
tr_11 = pd.read_csv("../input/train-data-image-blurrness/11(12_13.5).csv")
tr_12 = pd.read_csv("../input/train-data-image-blurrness/last.csv")

frames = [tr_1, tr_2, tr_3, tr_4, tr_5, tr_6, tr_7, tr_8, tr_9, tr_10, tr_11, tr_12]
new = pd.concat(frames)
new["File"] = new["File"].apply(lambda x : x.split("/")[-1].split(".")[0])
new = new.sort_values(by = ["File"])
scores = list(new["Score"].values) + [-1] * (len(train)-len(new))
train_sorted["image_blurrness_score"] = scores
train = train_sorted.sort_index()


##Testing
te_1 = pd.read_csv("../input/image-blurrness-test/test_1.csv")
te_2 = pd.read_csv("../input/image-blurrness-test/test_2.csv")
te_3 = pd.read_csv("../input/image-blurrness-test/test_3.csv")
te_4 = pd.read_csv("../input/image-blurrness-test/test_4.csv")
te_5 = pd.read_csv("../input/image-blurrness-test/test_5.csv")

frames_te = [te_1, te_2, te_3, te_4, te_5]
new_te = pd.concat(frames_te)
new_te["File"] = new_te["File"].apply(lambda x : x.split("/")[-1].split(".")[0])
new_te = new_te.sort_values(by = ["File"])
scores_te = list(new_te["Score"].values) + [-1] * (len(test)-len(new_te))

test_sorted["image_blurrness_score"] = scores_te
test = test_sorted.sort_index()


# In[3]:


gp = pd.read_csv("../input/aggregate/aggregated_features.csv")
train = train.merge(gp, on='user_id', how='left')
test = test.merge(gp, on='user_id', how='left')

agg_cols = list(gp.columns)[1:]

del gp
gc.collect()


# Now, I have  the train / test dataset to use. 

# In[4]:


count = lambda l1,l2: sum([1 for x in l1 if x in l2])


for df in [train, test]:
    df["price"] = np.log(df["price"]+0.001)
    df["image_blurrness_score"] = np.log(df["image_blurrness_score"]+0.001)
    df['description'].fillna('unknowndescription', inplace=True)
    df['title'].fillna('unknowntitle', inplace=True)
    
    df['weekday'] = pd.to_datetime(df['activation_date']).dt.day
    for col in ['description', 'title']:
        df['num_words_' + col] = df[col].apply(lambda comment: len(comment.split()))
        df['num_unique_words_' + col] = df[col].apply(lambda comment: len(set(w for w in comment.split())))

    df['words_vs_unique_title'] = df['num_unique_words_title'] / df['num_words_title'] * 100
    df['words_vs_unique_description'] = df['num_unique_words_description'] / df['num_words_description'] * 100
    
    df['city'] = df['region'] + '_' + df['city']
    df['num_desc_punct'] = df['description'].apply(lambda x: count(x, set(string.punctuation)))
    
    for col in agg_cols:
        df[col].fillna(-1, inplace=True)

for df in [train, test]:
    df.price.replace(to_replace=[np.inf, -np.inf,np.nan], value=-1,inplace=True)
    df.image_blurrness_score.replace(to_replace=[np.inf, -np.inf,np.nan], value=-1,inplace=True)


# In[5]:


count_vectorizer_title = CountVectorizer(stop_words=stopwords.words('russian'), lowercase=True, min_df=25)

title_counts = count_vectorizer_title.fit_transform(train['title'].append(test['title']))

train_title_counts = title_counts[:len(train)]
test_title_counts = title_counts[len(train):]


count_vectorizer_desc = TfidfVectorizer(stop_words=stopwords.words('russian'), 
                                        lowercase=True, ngram_range=(1, 2),
                                        max_features=15000)

desc_counts = count_vectorizer_desc.fit_transform(train['description'].append(test['description']))

train_desc_counts = desc_counts[:len(train)]
test_desc_counts = desc_counts[len(train):]

train_title_counts.shape, train_desc_counts.shape

train


# In[8]:


target = 'deal_probability'
predictors = [
    'num_desc_punct', 
    'words_vs_unique_description', 'num_unique_words_description', 'num_unique_words_title', 'num_words_description', 'num_words_title',
    'avg_times_up_user', 'avg_days_up_user', 'n_user_items', 
    'price', 'item_seq_number', "image_blurrness_score"
]
categorical = [
    'image_top_1', 'param_1', 'param_2', 'param_3', 
    'city', 'region', 'category_name', 'parent_category_name', 'user_type'
]

predictors = predictors + categorical


# In[9]:


for feature in categorical:
    print(f'Transforming {feature}...')
    encoder = LabelEncoder()
    encoder.fit(train[feature].append(test[feature]).astype(str))
    
    train[feature] = encoder.transform(train[feature].astype(str))
    test[feature] = encoder.transform(test[feature].astype(str))
    
train


# Ridge feature here
# 

# In[53]:


df = pd.concat([train[predictors], test[predictors]], axis = 0)
NFOLDS = 5
SEED = 42
VALID = False

traindex = train.index
testdex = test.index
ntrain = train.shape[0]
ntest = test.shape[0]
kf = KFold(ntrain, n_folds=NFOLDS, shuffle=True, random_state=SEED)



y = train.deal_probability.copy()

class SklearnWrapper(object):
    def __init__(self, clf, seed=0, params=None, seed_bool = True):
        if(seed_bool == True):
            params['random_state'] = seed
        self.clf = clf(**params)

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)
        
def get_oof(clf, x_train, y, x_test):
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((NFOLDS, ntest))

    for i, (train_index, test_index) in enumerate(kf):
        print('\nFold {}'.format(i))
        x_tr = x_train[train_index]
        y_tr = y[train_index]
        x_te = x_train[test_index]

        clf.train(x_tr, y_tr)

        oof_train[test_index] = clf.predict(x_te)
        oof_test_skf[i, :] = clf.predict(x_test)

    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)

ridge_params = {'alpha':20.0, 'fit_intercept':True, 'normalize':False, 'copy_X':True,
                'max_iter':None, 'tol':0.001, 'solver':'auto', 'random_state':SEED}
ridge = SklearnWrapper(clf=Ridge, seed = SEED, params = ridge_params)
ridge_oof_train, ridge_oof_test = get_oof(ridge, np.array(df[:ntrain]), y, np.array(df[ntrain:]))


def rmse(y, y0):
    assert len(y) == len(y0)
    return np.sqrt(np.mean(np.power((y - y0), 2)))
rms = sqrt(mean_squared_error(y, ridge_oof_train))
print('Ridge OOF RMSE: {}'.format(rms))

ridge_preds = np.concatenate([ridge_oof_train, ridge_oof_test])
df['ridge_preds'] = ridge_preds

train = df[:ntrain]
train["deal_probability"] = y
test = df[ntrain:]
predictors.append("ridge_preds")


# In[25]:


feature_names = np.hstack([
    count_vectorizer_desc.get_feature_names(),
    count_vectorizer_title.get_feature_names(),
    predictors
])
print('Number of features:', len(feature_names))


# In[60]:


train_index, valid_index = train_test_split(np.arange(len(train)), test_size=0.1, random_state=42)

x_train = scipy.sparse.hstack([
        train_desc_counts[train_index],
        train_title_counts[train_index],
        train.loc[train_index, predictors]
], format='csr')
y_train = train.loc[train_index, target]

x_valid = scipy.sparse.hstack([
    train_desc_counts[valid_index],
    train_title_counts[valid_index],
    train.loc[valid_index, predictors]
], format='csr')
y_valid = train.loc[valid_index, target]

x_test = scipy.sparse.hstack([
    test_desc_counts,
    test_title_counts,
    test.loc[:, predictors]
], format='csr')

dtrain = lgb.Dataset(x_train, label=y_train,
                     feature_name=list(feature_names), 
                     categorical_feature=categorical)
dvalid = lgb.Dataset(x_valid, label=y_valid,
                     feature_name=list(feature_names), 
                     categorical_feature=categorical)


# In[65]:


rounds = 16000
early_stop_rounds = 500
params = {
    'objective' : 'regression',
    'metric' : 'rmse',
    'num_leaves' : 32,
    'max_depth': 15,
    'learning_rate' : 0.02,
    'feature_fraction' : 0.6,
    'verbosity' : -1
}


evals_result = {}
model = lgb.train(params, dtrain, 
                  valid_sets=[dtrain, dvalid], 
                  valid_names=['train', 'valid'],
                  num_boost_round=rounds, 
                  early_stopping_rounds=early_stop_rounds, 
                  verbose_eval=500)


# That looks good. But the model is kind of a black box. It is a good idea to plot the feature importances for our model now.

# In[30]:


fig, ax = plt.subplots(figsize=(10, 14))
lgb.plot_importance(model, max_num_features=50, ax=ax)
plt.title("Light GBM Feature Importance")


# In[32]:


subm = pd.read_csv('../input/avito-demand-prediction/sample_submission.csv')
subm['deal_probability'] = np.clip(model.predict(x_test), 0, 1)
subm.to_csv('Aggregate_Ridge.csv', index=False)

