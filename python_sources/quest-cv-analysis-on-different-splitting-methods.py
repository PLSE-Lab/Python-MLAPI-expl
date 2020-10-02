#!/usr/bin/env python
# coding: utf-8

# # 1. This notebook is about "Splitting and CV"
# Good day everybody! In this competition, one important aspect is that **"Public LB is based on 13% of total test data, which is only 476 examples"**. So it is intuitively clear that our own CV score is more reliable than public LB, e.g. if we do 5-folds split, we will have 20% of training data, around 1200 examples. Moreover, we can average over 5-folds and get even more reliable number, right? It seems to me that in this *imbalance-multi-label* problem (explained below), splitting is not easy. So in this notebook, we will investigate and compare 3 splitting approaches which I know of:
# 
# - `KFold`
# - `GroupKFold`
# - `MultilabelStratifiedKFold`
# 
# The goal of this notebook is to compare CV from these methods among themselves, and also to Public LB. Since this notebook is not about optimizing LB, we will save time by using a very fast model training from [@abhishek's kernel](https://www.kaggle.com/abhishek/distilbert-use-features-oof) which in turn originated from [@abazdyrev's kernel](https://www.kaggle.com/abazdyrev/use-features-oof). 
# 
# **We will see that this same model will have different CV behaviors depending on splitting methods. Therefore, when talking about CV vs. LB, it's important to understand this different behaviors. In particular, when teaming up with other people, be sure that different CV calculation (if any) will not mislead your team.**
# 
# Note that this is all I know about splitting. At the end of the article, if anybody know a better method, or have better insights, and would like to share in the comment section, it will be very much appreciate!!
# 

# ## 1.1 Should we simply do K-folds split?
# So we should just use `sklearn.model_selection.KFold` and that's all? Unfortunately, in this problem, they are at least two subtleties which we have to be careful!
# 
# ### Problem A. There can be leakage between train and validation
# This [nice EDA kernel](https://www.kaggle.com/sediment/a-gentle-introduction-eda-tfidf-word2vec) shows that there are some questions that appear many times (with different answers each time). For example the most frequent question is
# 
# > What is the best introductory Bayesian statistics?
# 
# which appears 12 times in the training set. Therefore, this means that **if we split the data naively, this same question can be in both train and valid sets, and since many labels are related to "questions", there will be leakage between train and valid sets**.
# 
# > So naive `KFold` can give you an over-estimated CV
# 
# ## 1.2 GroupKFold
# 
# In contrast, this [top kernel](https://www.kaggle.com/akensert/bert-base-tf2-0-minimalistic) from my friend @akensert smartly avoid this problem by using `sklearn.model_selection.GroupKFold` instead! By using `GroupKFold` it is possible to guarantee that *all data with the same question, will always stay in the same fold*. Therefore, the above leakage will not happen, and CV is more likely to be more accurate compared to `KFold`.
# 
# So does `GroupKFold` solve the problem? In my understanding, unfortunately, we will face the other issue instead. As we can see from his kernel (Version 5), that in some fold, we cannot get the CV score! (`nan`). Why does this happen ??
# 
# ### Problem B. Some labels having very few non-zero values, possibly causes `spearmanr = nan`
# Concretely, if you take a look at the label `question_type_spelling`, there will be only 11 / 6079 non-zero values!!! That's are very few! And if we are not careful it will make CV score `nan` (explained below)

# In[ ]:


import numpy as np
import pandas as pd

df_train = pd.read_csv("../input/google-quest-challenge/train.csv").fillna("none")
spell = df_train['question_type_spelling'].values
print('There is only %d from %d examples!!!' % (np.sum(spell > 0), len(spell)))


# Moreover, if we print these 11 examples out we can see that they come from only 8 questions. Therefore, when we use`GroupKFold`, there can be some fold which have no example of positive value on `question_type_spelling` label at all. Note that in this multi-label problem, there are other labels having (less extreme) this kind of imbalance distribution too.

# In[ ]:


df_train[spell > 0]['question_title'].sort_values().values


# The problem is, if some fold has only 0-value of `question_type_spelling`, the calculation of spearman correlation will results in `nan`. Informally speaking, a correlation is related to a **slope** of 'best linear regression' between true values and predict values. Therefore, if either true or predict sets contains only one value, the slope is not well-defined.

# In[ ]:


from scipy.stats import spearmanr
print(spearmanr([1,2,3],[-0.00005,-0.00005,0]).correlation)
print(spearmanr([1,2,3],[-5,-5,-5]).correlation)
print(spearmanr([1,1,1],[-5,-4,-3]).correlation)


# Therefore, 
# 
# > `GroupKFold` has higher chance than `KFold` to get `nan` especially when increasing the `NUM_FOLDS` number
# 
# And we have to be careful, when we *average* the `GroupKFold` performance, not to include the nan fold, (You will see later that `np.nanmean()` will not help too).
# 
# ## 1.3 Multi-label Stratified-KFold
# Unlike the above two methods, `MultilabelStratifiedKFold` is designed directly for multi-labels problem. It will try to make *each fold have similar number of examples for each label*. Therefore, `MultilabelStratifiedKFold` is a method with the least chance to have  `nan` problem. Also, it should give the most stable estimation since the label distributions are similar (less biased toward extreme) to all folds.
# 
# However,it still have a leakage problem. So we still have a possibly over-estimated CV value. Here is [a very good slide](https://www.slideshare.net/tsoumakas/on-the-stratification-of-multilabel-data) explaining how it work.
# 
# In order to use `MultilabelStratifiedKFold` offline, you can use this script which copied from the [original repo](https://github.com/trent-b/iterative-stratification)
# 
# - https://www.kaggle.com/ratthachat/ml-stratifiers (In your notebook/kernel --> **go to File, and select "Adding utility script"**). Please upvote it if you find it useful :)
# 

# # 2. Take-away message 
# 
# ## 2.1 CV vs. LB
# For anybody who doesn't want to bother with the detailed experiments below, I summarize the main finding here. This summary is based on the fact that the baseline got 0.36x LB (after removing Elastic Network).
# 
# (1) For small number of folds e.g. `NUM_FOLDS=4` both `KFold` and `MultilabelStratifiedKFold` give similar CV results. I found CV is consistently around 0.02x more than LB possibly due to the leakage problem mentioned above. (If you choose an unlucky SEED, `KFold` can also give `nan` in some fold) 
# 
# In contrast, `GroupKFold` has much more chance to have a `nan` CV, and this is true with the default `SEED=42`. If we ignore the nan-fold, we will get a smaller gap of 0.01x between CV and LB. This should be because `GroupKFold` doesn't have the mentioned leakage problem. However, I also note that using `np.nanmean()` to average the results can lead to over-estimated CV. Therefore, do not use `np.nanmean()`.
# 
# (2) When `NUM_FOLDS=10`, `GroupKFold` will inevitably has at least two nan-folds due to its definition. In `SEED=42`, `KFold` will also face `nan`. Only `MultilabelStratifiedKFold` will not have a nan fold. Moreover, `MultilabelStratifiedKFold` should give the most stable estimation since the label distributions are similar (less biased toward extreme) to all folds.
# 
# ## 2.2 Be careful, when teaming up !
# Does not mean to discourage about teaming up, but I think we should make sure that all people in the same team use the same CV methodology, since if one person gets over-estimated CV, and another person gets less-over-estimated CV, it can mislead the team's best combination.
# 
# Another issue is that this problem is quite small, so one person alone can train many models. But there is a 2-hours running-time limitation, so the team should handle number of inference models for each member carefully.
# 
# ### My personal note
# - I would like to avoid `nan` completely, so I personally go to `MultilabelStratifiedKFold`. Therefore, my own CV will always be a bit over-score (gap around 0.02x). But still LB and CV are very consistent. Whenever CV increase, I always observe LB increases in similar magnitudes.

# # 3. Experiments Outline
# 
# As mentioned, we will compare different CVs of [@abhishek's kernel](https://www.kaggle.com/abhishek/distilbert-use-features-oof) which got LB 0.36x (removing Elastic Network to speed up). We will run the following 6 experiments.
# 
# ### 4-fold splitting for the 3 methods. 
# Here, you will see that 
# - both `KFold` and `MultilabelStratifiedKFold` have the leakage problem, so CVs are around 0.38x. But they will not have `nan` problem. Therefore, there are around 0.02x gap for these methods.
# - `GroupKFold` will have `nan` in one fold. In this case, if we ignore the `nan` fold, we will get CV 0.37x which is closer to LB. Therefore, ignoring `nan`,  there are around 0.01x gap for this method. This observation consistent with [@akensert's kernel](https://www.kaggle.com/akensert/bert-base-tf2-0-minimalistic) where he got average CV around 0.39x with LB 0.382 (Version 9).
# - I tried to use `np.nanmean()` to average over the nan-fold, but the result is misleading. It usually lead to 0.4x to even 0.5 CV. And I don't yet understand why this happen.
# 
# ### 10-fold splitting for the 3 methods.
# Here, when we increase the number of folds to 10, you will see that 
# - both `KFold` and `GroupKFold` will have `nan` problem. But there will be 3-4 `nan` folds for `GroupKFold`.
# - `MultilabelStratifiedKFold` will have no `nan`, but still have a bit over-estimated CV (around 0.02x gap).

# ## 3.1 Here's we begin the main code
# Since the main code is very similar to the original kernel, I hide most of the cells.

# In[ ]:


get_ipython().system('pip install ../input/sacremoses/sacremoses-master/ > /dev/null')

import os
import sys
import glob
import torch

sys.path.insert(0, "../input/transformers/transformers-master/")
import transformers
import math


# This is how we import the 3 splitting methods

# In[ ]:


from ml_stratifiers import MultilabelStratifiedShuffleSplit, MultilabelStratifiedKFold
from sklearn.model_selection import StratifiedKFold, KFold, GroupKFold
SEED = 42


# In[ ]:


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


# In[ ]:


def fetch_vectors(string_list, batch_size=64):
    # inspired by https://jalammar.github.io/a-visual-guide-to-using-bert-for-the-first-time/
    DEVICE = torch.device("cuda")
    tokenizer = transformers.DistilBertTokenizer.from_pretrained("../input/distilbertbaseuncased/")
    model = transformers.DistilBertModel.from_pretrained("../input/distilbertbaseuncased/")
    model.to(DEVICE)

    fin_features = []
    for data in chunks(string_list, batch_size):
        tokenized = []
        for x in data:
            x = " ".join(x.strip().split()[:300])
            tok = tokenizer.encode(x, add_special_tokens=True)
            tokenized.append(tok[:512])

        max_len = 512
        padded = np.array([i + [0] * (max_len - len(i)) for i in tokenized])
        attention_mask = np.where(padded != 0, 1, 0)
        input_ids = torch.tensor(padded).to(DEVICE)
        attention_mask = torch.tensor(attention_mask).to(DEVICE)

        with torch.no_grad():
            last_hidden_states = model(input_ids, attention_mask=attention_mask)

        features = last_hidden_states[0][:, 0, :].cpu().numpy()
        fin_features.append(features)

    fin_features = np.vstack(fin_features)
    return fin_features


# In[ ]:


df_train = pd.read_csv("../input/google-quest-challenge/train.csv").fillna("none")
df_test = pd.read_csv("../input/google-quest-challenge/test.csv").fillna("none")

sample = pd.read_csv("../input/google-quest-challenge/sample_submission.csv")
target_cols = list(sample.drop("qa_id", axis=1).columns)

train_question_body_dense = fetch_vectors(df_train.question_body.values)
train_answer_dense = fetch_vectors(df_train.answer.values)

test_question_body_dense = fetch_vectors(df_test.question_body.values)
test_answer_dense = fetch_vectors(df_test.answer.values)


# In[ ]:


import os
import re
import gc
import pickle  
import random
import keras

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import keras.backend as K

from keras.models import Model
from keras.layers import Dense, Input, Dropout, Lambda
from keras.optimizers import Adam
from keras.callbacks import Callback
from scipy.stats import spearmanr, rankdata
from os.path import join as path_join
from numpy.random import seed
from urllib.parse import urlparse
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold
from sklearn.linear_model import MultiTaskElasticNet

seed(42)
tf.random.set_seed(42)
random.seed(42)


# In[ ]:


data_dir = '../input/google-quest-challenge/'
train = pd.read_csv(path_join(data_dir, 'train.csv'))
test = pd.read_csv(path_join(data_dir, 'test.csv'))
print(train.shape, test.shape)
train.head()


# In[ ]:


targets = [
        'question_asker_intent_understanding',
        'question_body_critical',
        'question_conversational',
        'question_expect_short_answer',
        'question_fact_seeking',
        'question_has_commonly_accepted_answer',
        'question_interestingness_others',
        'question_interestingness_self',
        'question_multi_intent',
        'question_not_really_a_question',
        'question_opinion_seeking',
        'question_type_choice',
        'question_type_compare',
        'question_type_consequence',
        'question_type_definition',
        'question_type_entity',
        'question_type_instructions',
        'question_type_procedure',
        'question_type_reason_explanation',
        'question_type_spelling',
        'question_well_written',
        'answer_helpful',
        'answer_level_of_information',
        'answer_plausible',
        'answer_relevance',
        'answer_satisfaction',
        'answer_type_instructions',
        'answer_type_procedure',
        'answer_type_reason_explanation',
        'answer_well_written'    
    ]

input_columns = ['question_title', 'question_body', 'answer']


# ## Features

# Here we construct features from **Universal Sentence Encoder** and some smart extra features credited to original authors mentioned above.

# In[ ]:


find = re.compile(r"^[^.]*")

train['netloc'] = train['url'].apply(lambda x: re.findall(find, urlparse(x).netloc)[0])
test['netloc'] = test['url'].apply(lambda x: re.findall(find, urlparse(x).netloc)[0])

features = ['netloc', 'category']
merged = pd.concat([train[features], test[features]])
ohe = OneHotEncoder()
ohe.fit(merged)

features_train = ohe.transform(train[features]).toarray()
features_test = ohe.transform(test[features]).toarray()


# In[ ]:


module_url = "../input/universalsentenceencoderlarge4/"
embed = hub.load(module_url)


# In[ ]:


embeddings_train = {}
embeddings_test = {}
for text in input_columns:
    print(text)
    train_text = train[text].str.replace('?', '.').str.replace('!', '.').tolist()
    test_text = test[text].str.replace('?', '.').str.replace('!', '.').tolist()
    
    curr_train_emb = []
    curr_test_emb = []
    batch_size = 4
    ind = 0
    while ind*batch_size < len(train_text):
        curr_train_emb.append(embed(train_text[ind*batch_size: (ind + 1)*batch_size])["outputs"].numpy())
        ind += 1
        
    ind = 0
    while ind*batch_size < len(test_text):
        curr_test_emb.append(embed(test_text[ind*batch_size: (ind + 1)*batch_size])["outputs"].numpy())
        ind += 1    
        
    embeddings_train[text + '_embedding'] = np.vstack(curr_train_emb)
    embeddings_test[text + '_embedding'] = np.vstack(curr_test_emb)
    
del embed
K.clear_session()
gc.collect()


# In[ ]:


l2_dist = lambda x, y: np.power(x - y, 2).sum(axis=1)

cos_dist = lambda x, y: (x*y).sum(axis=1)

dist_features_train = np.array([
    l2_dist(embeddings_train['question_title_embedding'], embeddings_train['answer_embedding']),
    l2_dist(embeddings_train['question_body_embedding'], embeddings_train['answer_embedding']),
    l2_dist(embeddings_train['question_body_embedding'], embeddings_train['question_title_embedding']),
    cos_dist(embeddings_train['question_title_embedding'], embeddings_train['answer_embedding']),
    cos_dist(embeddings_train['question_body_embedding'], embeddings_train['answer_embedding']),
    cos_dist(embeddings_train['question_body_embedding'], embeddings_train['question_title_embedding'])
]).T

dist_features_test = np.array([
    l2_dist(embeddings_test['question_title_embedding'], embeddings_test['answer_embedding']),
    l2_dist(embeddings_test['question_body_embedding'], embeddings_test['answer_embedding']),
    l2_dist(embeddings_test['question_body_embedding'], embeddings_test['question_title_embedding']),
    cos_dist(embeddings_test['question_title_embedding'], embeddings_test['answer_embedding']),
    cos_dist(embeddings_test['question_body_embedding'], embeddings_test['answer_embedding']),
    cos_dist(embeddings_test['question_body_embedding'], embeddings_test['question_title_embedding'])
]).T

X_train = np.hstack([item for k, item in embeddings_train.items()] + [features_train, dist_features_train])
X_test = np.hstack([item for k, item in embeddings_test.items()] + [features_test, dist_features_test])
y_train = train[targets].values


# In[ ]:


X_train = np.hstack((X_train, train_question_body_dense, train_answer_dense))
X_test = np.hstack((X_test, test_question_body_dense, test_answer_dense))


# ## Modeling

# I modify a little on Spearman Callback, and remove the use of Elastic network since it consumes a lot of time.

# In[ ]:


'''Here, I modify a little
(1) recount bad_epoch every time new best_weight is founded
(2) save and load best model
(3) I use pure spearman correlation, no random noise added
'''

class SpearmanRhoCallback(Callback):
    def __init__(self, training_data, validation_data, patience, model_name):
        self.x = training_data[0]
        self.y = training_data[1]
        self.x_val = validation_data[0]
        self.y_val = validation_data[1]
        
        self.patience = patience
        self.value = -1
        self.bad_epochs = 0
        self.model_name = model_name

    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return
    
    def on_epoch_end(self, epoch, logs={}):
        y_pred_val = self.model.predict(self.x_val)
#         rho_val = np.mean([spearmanr(self.y_val[:, ind], y_pred_val[:, ind] + np.random.normal(0, 1e-7, y_pred_val.shape[0])).correlation for ind in range(y_pred_val.shape[1])])
        rho_val = np.mean([ spearmanr(self.y_val[:, ind], y_pred_val[:, ind]).correlation for ind in range(y_pred_val.shape[1]) ])
        if rho_val >= self.value:
            self.value = rho_val
            self.bad_epochs = 0
            print('\nsave best weights\n')
            self.model.save_weights(self.model_name)
        else:
            self.bad_epochs += 1
            print('bad epochs = ',self.bad_epochs)
        if self.bad_epochs >= self.patience:
            print("Epoch %05d: early stopping Threshold -- load best weights" % epoch)
            try:
                self.model.load_weights(self.model_name)
            except:
                print('could not load a model')
            self.model.stop_training = True
            
#             
        print('\rval_spearman-rho: %s' % (str(round(rho_val, 4))), end=100*' '+'\n')
        return rho_val

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return


# In[ ]:


def create_model():
    inps = Input(shape=(X_train.shape[1],))
    x = Dense(512, activation='elu')(inps)
    x = Dropout(0.2)(x)
    x = Dense(y_train.shape[1], activation='sigmoid')(x)
    model = Model(inputs=inps, outputs=x)
    model.compile(
        optimizer=Adam(lr=1e-4),
        loss=['binary_crossentropy']
    )
    model.summary()
    return model


# # 4. Experiments with NUM_FOLDS = 4
# 
# If the number of folds is small, e.g. `NUM_FOLDS=4`. We will see that `KFold` usually works fine like `MultilabelStratifiedKFold` (but there's no guaranteed if we change splitting SEED (default=42) ). In the experiment, I run the 4-fold training and record all CVs.
# 
# Only `GroupKFold` will have `nan` with this SEED . So, either we have to change SEED or we have to carefully ignore `nan`. **Important note is that we should not use `np.nanmean` since it will give us dramatically high CV for that nan-fold, so that we will have an over-estimated CV instead. See details in `GroupKFold` experiment below **
# 
# To summarize empirical results which you can see below, `KFold` and `MultilabelStratifiedKFold` will give us average CV around 0.38x with standard deviation in a 3rd decimal. While `GroupKFold` if ignoring `nan` properly usually give us average CV around 0.37x which is closer to 0.36x LB. However, as noted above, if we use `np.nanmean` in the nan-fold, we will have strangely-high CV, and CV standard deviation becomes 2nd decimal indicating unreliability.
# 
# ## Plain KFold - 4 folds

# In[ ]:


NUM_FOLDS=4
kf = KFold(n_splits=NUM_FOLDS, random_state=SEED, shuffle=True)


# In[ ]:


all_predictions = []
rho_kfolds = []

for ind, (tr, val) in enumerate(kf.split(X_train)):
    X_tr = X_train[tr]
    y_tr = y_train[tr]
    X_vl = X_train[val]
    y_vl = y_train[val]
    
    model = create_model()
    model.fit(
        X_tr, y_tr, epochs=100, batch_size=32, validation_data=(X_vl, y_vl), verbose=True, 
        callbacks=[SpearmanRhoCallback(training_data=(X_tr, y_tr), validation_data=(X_vl, y_vl),
                                       patience=15, model_name=f'best_model_batch{ind}.h5')]
    )
    all_predictions.append(model.predict(X_test))

    y_pred_val = model.predict(X_vl)
    rho_list = [ spearmanr(y_vl[:, ind], y_pred_val[:, ind]).correlation for ind in range(y_pred_val.shape[1]) ]
    rho_kfolds.append(rho_list)


# In[ ]:


print('Each fold : ', np.mean(rho_kfolds,axis=1))
print('Using nanmean, each fold : ', np.nanmean(rho_kfolds,axis=1))
print('Average performance : %.4f +/- %.4f'% ( np.mean(rho_kfolds), np.std(np.mean(rho_kfolds,axis=1)) ) )

print('Each label : ')
spearman_avg_per_label = np.mean(rho_kfolds,axis=0) # metric for each label -- use print line-by-line for better illustration
spearman_std_per_label = np.std(rho_kfolds,axis=0)
for ii in range(len(target_cols)):
    print('%d - %.3f +/- %.3f - %s' % (ii+1,spearman_avg_per_label[ii],spearman_std_per_label[ii],
                                       target_cols[ii] ))
    
rho_kfolds_plain = np.array(rho_kfolds) # saving for later use


# ## Multi-label Stratified-KFold - 4 folds

# In[ ]:


all_predictions = []

rho_kfolds = []

kf = MultilabelStratifiedKFold(n_splits = NUM_FOLDS, random_state = SEED)
for ind, (tr, val) in enumerate(kf.split(X_train,y_train)):
    X_tr = X_train[tr]
    y_tr = y_train[tr]
    X_vl = X_train[val]
    y_vl = y_train[val]
    
    model = create_model()
    model.fit(
        X_tr, y_tr, epochs=100, batch_size=32, validation_data=(X_vl, y_vl), verbose=True, 
        callbacks=[SpearmanRhoCallback(training_data=(X_tr, y_tr), validation_data=(X_vl, y_vl),
                                       patience=15, model_name=f'best_model_batch{ind}.h5')]
    )
    all_predictions.append(model.predict(X_test))

    y_pred_val = model.predict(X_vl)
    rho_list = [ spearmanr(y_vl[:, ind], y_pred_val[:, ind]).correlation for ind in range(y_pred_val.shape[1]) ]
    rho_kfolds.append(rho_list)


# In[ ]:


print('Each fold : ', np.mean(rho_kfolds,axis=1))
print('Using nanmean, each fold : ', np.nanmean(rho_kfolds,axis=1))
print('Average performance : %.4f +/- %.4f'% ( np.mean(rho_kfolds), np.std(np.mean(rho_kfolds,axis=1)) ) )

'''I hide details for read-ability. You can comment out to see per-label details'''
# print('Each label : ')
# spearman_avg_per_label = np.mean(rho_kfolds,axis=0) # metric for each label -- use print line-by-line for better illustration
# spearman_std_per_label = np.std(rho_kfolds,axis=0)
# for ii in range(len(target_cols)):
#     print('%d - %.3f +/- %.3f - %s' % (ii+1,spearman_avg_per_label[ii],spearman_std_per_label[ii],
#                                        target_cols[ii] ))
    
rho_kfolds_multi = np.array(rho_kfolds) # saving for later use


# ## GroupKFold - 4 folds

# In[ ]:


all_predictions = []

rho_kfolds = []

kf = GroupKFold(n_splits=NUM_FOLDS).split(X=df_train.question_body, groups=df_train.question_body)
for ind, (tr, val) in enumerate(kf):
    X_tr = X_train[tr]
    y_tr = y_train[tr]
    X_vl = X_train[val]
    y_vl = y_train[val]
    
    model = create_model()
    model.fit(
        X_tr, y_tr, epochs=100, batch_size=32, validation_data=(X_vl, y_vl), verbose=True, 
        callbacks=[SpearmanRhoCallback(training_data=(X_tr, y_tr), validation_data=(X_vl, y_vl),
                                       patience=15, model_name=f'best_model_batch{ind}.h5')]
    )
    all_predictions.append(model.predict(X_test))

    y_pred_val = model.predict(X_vl)
    rho_list = [ spearmanr(y_vl[:, ind], y_pred_val[:, ind]).correlation for ind in range(y_pred_val.shape[1]) ]
    rho_kfolds.append(rho_list)


# Here, we got `nan` in the last fold, so we have to fix it. **NOTE the absurdly high CV for `np.nanmean` in the last fold! So don't use it**. If we exclude the nan-fold properly, see `rho_kfolds_group_nonan`, we will get a good CV in this case.

# In[ ]:


print('Each fold : ', np.mean(rho_kfolds,axis=1))
print('**NOTE** Using nanmean, each fold : ', np.nanmean(rho_kfolds,axis=1))
print('Average performance : %.4f +/- %.4f'% ( np.mean(rho_kfolds), np.std(np.mean(rho_kfolds,axis=1)) ) )

print('Each label : ')
spearman_avg_per_label = np.mean(rho_kfolds,axis=0) # metric for each label -- use print line-by-line for better illustration
spearman_std_per_label = np.std(rho_kfolds,axis=0)
for ii in range(len(target_cols)):
    print('%d - %.3f +/- %.3f - %s' % (ii+1,spearman_avg_per_label[ii],spearman_std_per_label[ii],
                                       target_cols[ii] ))
    
rho_kfolds_group = np.array(rho_kfolds) # saving for later use

rho_kfolds_group_nonan = [] 
mm = np.mean(rho_kfolds,axis=1)
for ii in range(len(mm)):    
    if np.isnan(mm[ii]) == False:
        rho_kfolds_group_nonan.append(rho_kfolds[ii])
        
print('Average performance with ignoring nan : %.4f +/- %.4f'% ( np.mean(rho_kfolds_group_nonan), np.std(np.mean(rho_kfolds_group_nonan,axis=1)) ) )


# # Summary for 4 folds

# In[ ]:


pd.set_option('precision',4)
pd.set_option('display.precision',4)
pd.set_option('display.float_format','{:.4f}'.format)

df = pd.DataFrame(columns=['label','plain','multi','group_nanmean','group_nonan'])
df['label'] = df_train.columns[11:]
df['plain'] = np.nanmean(rho_kfolds_plain,axis=0)
df['multi'] = np.nanmean(rho_kfolds_multi,axis=0)
df['group_nanmean'] = np.nanmean(rho_kfolds_group,axis=0)
df['group_nonan'] = np.mean(rho_kfolds_group_nonan,axis=0)

df2 = pd.DataFrame([['average',np.mean(rho_kfolds_plain), 
                     np.mean(rho_kfolds_multi), 
                     np.nanmean(rho_kfolds_group),
                     np.mean(rho_kfolds_group_nonan),
                    ]],
                     columns=df.columns)
df2 = df.append(df2)

df2.head(35)


# In[ ]:





# # 5. Experiments with NUM_FOLDS = 10
# 
# ## Plain KFold - 10 folds
# When we step up to 10 folds, now KFold will also face `nan` problem by chance. Note that some SEED may not produce `nan`, but the current SEED=42 which usually a default number in public kernel will have `nan` in 2 folds. So now in order to calculate average CV, we have to ignore the nan fold.

# In[ ]:


NUM_FOLDS=10


# In[ ]:


all_predictions = []
rho_kfolds = []

kf = KFold(n_splits=NUM_FOLDS, random_state=SEED, shuffle=True)
for ind, (tr, val) in enumerate(kf.split(X_train)):
    X_tr = X_train[tr]
    y_tr = y_train[tr]
    X_vl = X_train[val]
    y_vl = y_train[val]
    
    model = create_model()
    model.fit(
        X_tr, y_tr, epochs=100, batch_size=32, validation_data=(X_vl, y_vl), verbose=True, 
        callbacks=[SpearmanRhoCallback(training_data=(X_tr, y_tr), validation_data=(X_vl, y_vl),
                                       patience=15, model_name=f'best_model_batch{ind}.h5')]
    )
    all_predictions.append(model.predict(X_test))

    y_pred_val = model.predict(X_vl)
    rho_list = [ spearmanr(y_vl[:, ind], y_pred_val[:, ind]).correlation for ind in range(y_pred_val.shape[1]) ]
    rho_kfolds.append(rho_list)


# In[ ]:


print('Each fold : ', np.mean(rho_kfolds,axis=1))
print('Using nanmean, each fold : ', np.nanmean(rho_kfolds,axis=1))
print('Average performance : %.4f +/- %.4f'% ( np.mean(rho_kfolds), np.std(np.mean(rho_kfolds,axis=1)) ) )

print('Each label : ')
spearman_avg_per_label = np.mean(rho_kfolds,axis=0) # metric for each label -- use print line-by-line for better illustration
spearman_std_per_label = np.std(rho_kfolds,axis=0)
for ii in range(len(target_cols)):
    print('%d - %.3f +/- %.3f - %s' % (ii+1,spearman_avg_per_label[ii],spearman_std_per_label[ii],
                                       target_cols[ii] ))
    
rho_kfolds_plain = np.array(rho_kfolds) # saving for later use

rho_kfolds_plain_nonan = [] 
mm = np.mean(rho_kfolds,axis=1)
for ii in range(len(mm)):    
    if np.isnan(mm[ii]) == False:
        rho_kfolds_plain_nonan.append(rho_kfolds[ii])
        
print('Average performance with ignoring nan : %.4f +/- %.4f'% ( np.mean(rho_kfolds_plain_nonan), np.std(np.mean(rho_kfolds_plain_nonan,axis=1)) ) )


# ## Multi-label StratifiedKFold  - 10 folds
# Using this method, even 10 fold, we will not have `nan` problem.

# In[ ]:


all_predictions = []

rho_kfolds = []

kf = MultilabelStratifiedKFold(n_splits = NUM_FOLDS, random_state = SEED)
for ind, (tr, val) in enumerate(kf.split(X_train,y_train)):
    X_tr = X_train[tr]
    y_tr = y_train[tr]
    X_vl = X_train[val]
    y_vl = y_train[val]
    
    model = create_model()
    model.fit(
        X_tr, y_tr, epochs=100, batch_size=32, validation_data=(X_vl, y_vl), verbose=True, 
        callbacks=[SpearmanRhoCallback(training_data=(X_tr, y_tr), validation_data=(X_vl, y_vl),
                                       patience=15, model_name=f'best_model_batch{ind}.h5')]
    )
    all_predictions.append(model.predict(X_test))

    y_pred_val = model.predict(X_vl)
    rho_list = [ spearmanr(y_vl[:, ind], y_pred_val[:, ind]).correlation for ind in range(y_pred_val.shape[1]) ]
    rho_kfolds.append(rho_list)


# In[ ]:


print('Each fold : ', np.mean(rho_kfolds,axis=1))
print('Using nanmean, each fold : ', np.nanmean(rho_kfolds,axis=1))
print('Average performance : %.4f +/- %.4f'% ( np.mean(rho_kfolds), np.std(np.mean(rho_kfolds,axis=1)) ) )

'''I hide details for read-ability. You can comment out to see per-label details'''
# print('Each label : ')
# spearman_avg_per_label = np.mean(rho_kfolds,axis=0) # metric for each label -- use print line-by-line for better illustration
# spearman_std_per_label = np.std(rho_kfolds,axis=0)
# for ii in range(len(target_cols)):
#     print('%d - %.3f +/- %.3f - %s' % (ii+1,spearman_avg_per_label[ii],spearman_std_per_label[ii],
#                                        target_cols[ii] ))
    
rho_kfolds_multi = np.array(rho_kfolds) # saving for later use


# ## GroupKFolds - 10 folds
# Now we will have 3-4 `nan` (both by chance and by definition) since only 8-`nan`-questions have to be divided randomly into 10 folds.

# In[ ]:


all_predictions = []

rho_kfolds = []

kf = GroupKFold(n_splits=NUM_FOLDS).split(X=df_train.question_body, groups=df_train.question_body)
for ind, (tr, val) in enumerate(kf):
    X_tr = X_train[tr]
    y_tr = y_train[tr]
    X_vl = X_train[val]
    y_vl = y_train[val]
    
    model = create_model()
    model.fit(
        X_tr, y_tr, epochs=100, batch_size=32, validation_data=(X_vl, y_vl), verbose=True, 
        callbacks=[SpearmanRhoCallback(training_data=(X_tr, y_tr), validation_data=(X_vl, y_vl),
                                       patience=15, model_name=f'best_model_batch{ind}.h5')]
    )
    all_predictions.append(model.predict(X_test))

    y_pred_val = model.predict(X_vl)
    rho_list = [ spearmanr(y_vl[:, ind], y_pred_val[:, ind]).correlation for ind in range(y_pred_val.shape[1]) ]
    rho_kfolds.append(rho_list)


# In[ ]:


print('Each fold : ', np.mean(rho_kfolds,axis=1))
print('Using nanmean, each fold : ', np.nanmean(rho_kfolds,axis=1))
print('Average performance : %.4f +/- %.4f'% ( np.mean(rho_kfolds), np.std(np.mean(rho_kfolds,axis=1)) ) )

print('Each label : ')
spearman_avg_per_label = np.mean(rho_kfolds,axis=0) # metric for each label -- use print line-by-line for better illustration
spearman_std_per_label = np.std(rho_kfolds,axis=0)
for ii in range(len(target_cols)):
    print('%d - %.3f +/- %.3f - %s' % (ii+1,spearman_avg_per_label[ii],spearman_std_per_label[ii],
                                       target_cols[ii] ))
    
rho_kfolds_group = np.array(rho_kfolds) # saving for later use
rho_kfolds_group_nonan = [] 
mm = np.mean(rho_kfolds,axis=1)
for ii in range(len(mm)):    
    if np.isnan(mm[ii]) == False:
        rho_kfolds_group_nonan.append(rho_kfolds[ii])
        
print('Average performance with ignoring nan : %.4f +/- %.4f'% ( np.mean(rho_kfolds_group_nonan), np.std(np.mean(rho_kfolds_group_nonan,axis=1)) ) )


# # Summary for 10 folds
# 
# Note that `GroupKFold` can give us most accurate average CV if ignoring nan, but can come with high standard deviation of 2nd-decimal point.

# In[ ]:


pd.set_option('precision',4)
pd.set_option('display.precision',4)
pd.set_option('display.float_format','{:.4f}'.format)

df = pd.DataFrame(columns=['label','plain_nonan','multi','group_nanmean','group_nonan'])
df['label'] = df_train.columns[11:]
df['plain_nonan'] = np.mean(rho_kfolds_plain_nonan,axis=0)
df['multi'] = np.mean(rho_kfolds_multi,axis=0)
df['group_nanmean'] = np.nanmean(rho_kfolds_group,axis=0)
df['group_nonan'] = np.mean(rho_kfolds_group_nonan,axis=0)

df2 = pd.DataFrame([['average',np.mean(rho_kfolds_plain_nonan), 
                     np.mean(rho_kfolds_multi), 
                     np.nanmean(rho_kfolds_group),
                     np.mean(rho_kfolds_group_nonan),
                    ]],
                     columns=df.columns)
df2 = df.append(df2)

df2.head(35)


# In[ ]:




