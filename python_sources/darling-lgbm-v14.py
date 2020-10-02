#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import gc
import glob
import os
import json
import pprint
import warnings
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from joblib import Parallel, delayed
from tqdm import tqdm, tqdm_notebook
from nltk.corpus import stopwords
from scipy.sparse import csr_matrix, hstack
np.random.seed(seed=1337)
warnings.filterwarnings('ignore')


# In[ ]:


train = pd.read_csv('../input/petfinder-adoption-prediction/train/train.csv')
test = pd.read_csv('../input/petfinder-adoption-prediction/test/test.csv')
sample_submission = pd.read_csv('../input/petfinder-adoption-prediction/test/sample_submission.csv')


# In[ ]:


labels_breed = pd.read_csv('../input/petfinder-adoption-prediction/breed_labels.csv')
labels_state = pd.read_csv('../input/petfinder-adoption-prediction/color_labels.csv')
labels_color = pd.read_csv('../input/petfinder-adoption-prediction/state_labels.csv')


# In[ ]:


train_image_files = sorted(glob.glob('../input/petfinder-adoption-prediction/train_images/*.jpg'))
train_metadata_files = sorted(glob.glob('../input/petfinder-adoption-prediction/train_metadata/*.json'))
train_sentiment_files = sorted(glob.glob('../input/petfinder-adoption-prediction/train_sentiment/*.json'))
test_image_files = sorted(glob.glob('../input/petfinder-adoption-prediction/test_images/*.jpg'))
test_metadata_files = sorted(glob.glob('../input/petfinder-adoption-prediction/test_metadata/*.json'))
test_sentiment_files = sorted(glob.glob('../input/petfinder-adoption-prediction/test_sentiment/*.json'))


# In[ ]:


import scipy as sp
from collections import Counter
from functools import partial
from math import sqrt
from sklearn.metrics import cohen_kappa_score, mean_squared_error
from sklearn.metrics import confusion_matrix as sk_cmatrix

def confusion_matrix(rater_a, rater_b, min_rating=None, max_rating=None):
    assert(len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(rater_a + rater_b)
    if max_rating is None:
        max_rating = max(rater_a + rater_b)
    num_ratings = int(max_rating - min_rating + 1)
    conf_mat = [[0 for i in range(num_ratings)] for j in range(num_ratings)]
    for a, b in zip(rater_a, rater_b):
        conf_mat[a - min_rating][b - min_rating] += 1
    return conf_mat

def histogram(ratings, min_rating=None, max_rating=None):
    if min_rating is None:
        min_rating = min(ratings)
    if max_rating is None:
        max_rating = max(ratings)
    num_ratings = int(max_rating - min_rating + 1)
    hist_ratings = [0 for x in range(num_ratings)]
    for r in ratings:
        hist_ratings[r - min_rating] += 1
    return hist_ratings

def quadratic_weighted_kappa(y, y_pred):
    rater_a, rater_b =  y, y_pred
    min_rating, max_rating = None, None
    rater_a = np.array(rater_a, dtype=int)
    rater_b = np.array(rater_b, dtype=int)
    assert(len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(min(rater_a), min(rater_b))
    if max_rating is None:
        max_rating = max(max(rater_a), max(rater_b))
    conf_mat = confusion_matrix(rater_a, rater_b,min_rating, max_rating)
    num_ratings = len(conf_mat)
    num_scored_items = float(len(rater_a))
    hist_rater_a = histogram(rater_a, min_rating, max_rating)
    hist_rater_b = histogram(rater_b, min_rating, max_rating)
    numerator, denominator = 0.0, 0.0
    for i in range(num_ratings):
        for j in range(num_ratings):
            expected_count = (hist_rater_a[i] * hist_rater_b[j]/ num_scored_items)
            d = pow(i - j, 2.0) / pow(num_ratings - 1, 2.0)
            numerator += d * conf_mat[i][j] / num_scored_items
            denominator += d * expected_count / num_scored_items
    return (1.0 - numerator / denominator)


# In[ ]:


user_ID = train.groupby('RescuerID')['PetID'].count().sort_values().reset_index()
user_ID['class'] = (user_ID['PetID']//3).clip(0,5)
user_ID.loc[user_ID['class'] == 3, 'class'] = 2
user_ID.loc[user_ID['class'] == 4, 'class'] = 2


# In[ ]:


def get_adoptionspeed(arr):
    n = arr.shape[0]
    c0,c1,c2,c3 = np.int(n*0.05), np.int(n*0.25),np.int(n*0.5), np.int(n*0.75)
    arr = np.argsort(np.argsort(arr))
    arr = [0 if x < c0
          else 1 if x < c1
          else 2 if x < c2
          else 3 if x < c3
          else 4 for x in arr]
    return arr


# In[ ]:


def kappa(preds, train_data):
    labels = train_data.get_label()
    pred_class = get_adoptionspeed(preds)
    return 'kappa', quadratic_weighted_kappa(labels, pred_class), True


# In[ ]:


all_data = pd.concat([train[test.columns], test], ignore_index=True)


# In[ ]:


train['Fee_v1'] = (train['Fee'] > 0).astype('int')
train['Sterilized'].replace(3, np.nan, inplace = True)
train['Vaccinated'].replace(3, np.nan, inplace = True)

test['Fee_v1'] = (test['Fee'] > 0).astype('int')
test['Sterilized'].replace(3, np.nan, inplace = True)
test['Vaccinated'].replace(3, np.nan, inplace = True)


# In[ ]:


user_ID.columns = ['RescuerID', 'cntPet','class']
train = train.merge(user_ID, on=['RescuerID'], how='left')
test = test.merge(user_ID, on=['RescuerID'], how='left')


# In[ ]:


labels_breed['First'] = labels_breed['BreedName'].str[0]
labels_breed.columns = ['Breed1','Type','BreedName','First']
all_data = all_data.merge(labels_breed[['Breed1','First']], on=['Breed1'], how='left')


# In[ ]:


more_dim = 0
lb = LabelBinarizer(sparse_output=True)
X_Breed1 = lb.fit_transform(all_data['Breed1'])
more_dim += X_Breed1.shape[1]
X_Breed2 = lb.fit_transform(all_data['Breed1'])
more_dim += X_Breed2.shape[1]
X_State = lb.fit_transform(all_data['State'])
more_dim += X_State.shape[1]
X_Color1 = lb.fit_transform(all_data['Color1'])
more_dim += X_Color1.shape[1]
X_Color2 = lb.fit_transform(all_data['Color2'])
more_dim += X_Color2.shape[1]
X_Color3 = lb.fit_transform(all_data['Color3'])
more_dim += X_Color3.shape[1]
# X_Breed1_F = lb.fit_transform(all_data['First'].fillna("NULL"))
# more_dim += X_Breed1_F.shape[1]


# In[ ]:


res = []
for i in tqdm_notebook(train_sentiment_files + test_sentiment_files):
    with open(i, 'r', encoding='utf-8') as f:
        tmp = json.load(f)
        res.append((i[-14:-5],  len(tmp['sentences']), len(tmp['tokens']),                  tmp['documentSentiment']['magnitude'], tmp['documentSentiment']['score']))

tmp = pd.DataFrame(res)
tmp.columns = ['PetID', 'len_des','n_tokens','magnitude','score']
train = train.merge(tmp, on=['PetID'], how='left')
test = test.merge(tmp, on=['PetID'], how='left')


# In[ ]:


tmp = all_data.groupby(['RescuerID'])['Age','Fee','PhotoAmt','VideoAmt','Quantity']              .agg(['min','max','mean','sum']).reset_index()
tmp.columns = ['RescuerID','min_age','max_age','mean_age','sum_age',               'min_Fee','max_Fee','mean_Fee','sum_Fee',              'min_PhotoAmt','max_PhotoAmt','mean_PhotoAmt','sum_PhotoAmt',              'min_VideoAmt','max_VideoAmt','mean_VideoAmt','sum_VideoAmt',              'min_Quantity','max_Quantity','mean_Quantity','sum_Quantity']
train = train.merge(tmp, on=['RescuerID'], how='left')
test = test.merge(tmp, on=['RescuerID'], how='left')


# In[ ]:


tmp = all_data[all_data['MaturitySize'] > 0].groupby(['RescuerID'])['MaturitySize'].agg(['min','max','mean']).reset_index()
tmp.columns = ['RescuerID','min_MaturitySize_1','max_MaturitySize_1','mean_MaturitySize_1']
train = train.merge(tmp, on=['RescuerID'], how='left')
test = test.merge(tmp, on=['RescuerID'], how='left')


# In[ ]:


tmp = all_data[all_data['Health'] > 0].groupby(['RescuerID'])['MaturitySize'].agg(['min','max','mean']).reset_index()
tmp.columns = ['RescuerID','min_Health','max_Health','mean_Health']
train = train.merge(tmp, on=['RescuerID'], how='left')
test = test.merge(tmp, on=['RescuerID'], how='left')


# In[ ]:


train.columns


# In[ ]:


count_cols = []
for i in ['Breed1','Color1','Sterilized','Vaccinated','MaturitySize','FurLength','Health']:
    tmp = all_data.groupby(['RescuerID'])[i].value_counts().unstack().replace(np.nan, 0).reset_index()
    tmp.columns = [tmp.columns[0]] + ['C_{}_C_{}'.format(i, str(j)) for j in range(len(tmp.columns) - 1)]
    count_cols = count_cols + list(tmp.columns[1:])
    train = train.merge(tmp, on = ['RescuerID'], how='left')
    test = test.merge(tmp, on = ['RescuerID'], how='left')


# In[ ]:


get_ipython().run_cell_magic('time', '', 'pet_id, x, y,red, green,blue,score, pixelFraction, confindence, annot_first, annot_last, annot_des =\\\n                [],[],[],[],[],[],[],[],[],[],[],[]\nfor i in tqdm_notebook(train_metadata_files  + test_metadata_files):\n    with open(i, \'r\', encoding=\'utf-8\') as f:\n        tmp = json.load(f)\n        pet_id.append(i.split("-")[-2][-9:])\n        x.append(tmp[\'cropHintsAnnotation\'][\'cropHints\'][0][\'boundingPoly\'][\'vertices\'][2][\'x\'])\n        y.append(tmp[\'cropHintsAnnotation\'][\'cropHints\'][0][\'boundingPoly\'][\'vertices\'][2][\'y\'])\n        confindence.append(tmp[\'cropHintsAnnotation\'][\'cropHints\'][0][\'confidence\'])\n        sub_tmp = tmp[\'imagePropertiesAnnotation\'][\'dominantColors\'][\'colors\'][0][\'color\']\n        \n        if (len(sub_tmp)>0):\n            red.append(sub_tmp[\'red\'])\n            green.append(sub_tmp[\'green\'])\n            blue.append(sub_tmp[\'blue\'])\n        else:\n            red.append(np.nan)\n            green.append(np.nan)\n            blue.append(np.nan)\n            \n        sub_tmp = tmp[\'imagePropertiesAnnotation\'][\'dominantColors\'][\'colors\'][0]\n        if (len(sub_tmp) > 0):\n            score.append(sub_tmp[\'score\'])\n            pixelFraction.append(sub_tmp[\'pixelFraction\'])\n        else:\n            score.append(np.nan)\n            pixelFraction.append(np.nan)\n            \n        if (\'labelAnnotations\' in tmp):\n            annot_first.append(tmp[\'labelAnnotations\'][0][\'score\'])\n            annot_last.append(tmp[\'labelAnnotations\'][-1][\'score\'])\n            annot_des.append(" ".join([k[\'description\'] for k in tmp[\'labelAnnotations\']]))\n        else:\n            annot_first.append(np.nan)\n            annot_last.append(np.nan)\n            annot_des.append(" ")\ntmp = pd.DataFrame({0: pet_id, 1: x, 2:y,3:red, 4:green, 5:blue, 6:score,\\\n                    7:pixelFraction,8:confindence, 9:annot_first, 10:annot_last})\\\n            .groupby([0])[1,2,3,4,5,6,7,8,9,10].agg([\'min\',\'max\',\'mean\']).reset_index()\ntmp.columns = [\'PetID\',\'min_x\',\'max_x\', \'mean_x\',\'min_y\',\'max_y\',\'mean_y\',\\\n               \'min_red_0\',\'max_red_0\', \'mean_red_0\',\\\n               \'min_green_0\',\'max_green_0\',\'mean_green_0\',\\\n              \'min_blue_0\',\'max_blue_0\', \'mean_blue_0\',\\\n              \'min_pixelscore\',\'max_pixelscore\', \'mean_pixelscore\',\\\n               \'min_pixelFraction\',\'max_pixelFraction\',\'mean_pixelFraction\',\\\n               \'min_confidence\',\'max_confidence\',\'mean_confidence\',\\\n               \'min_score_labelAnnotations\',\'max_score_labelAnnotations\',\'mean_score_labelAnnotations\',\\\n               \'min_score_labelAnnotations_1\',\'max_score_labelAnnotations_1\',\'mean_score_labelAnnotations_1\'\n              ]\n\ntrain = train.merge(tmp, on = [\'PetID\'], how=\'left\')\ntest = test.merge(tmp, on=[\'PetID\'], how=\'left\')')


# In[ ]:


tmp = pd.DataFrame({0:pet_id, 1:annot_des}).groupby([0])[1].apply(lambda x: " ".join(x)).reset_index()
tmp.columns = ['PetID','labelAnnotations_des']
train = train.merge(tmp, on=['PetID'], how='left')
test = test.merge(tmp, on=['PetID'], how='left')
stopWords = stopwords.words('english')
tfidf =  TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='word',
    min_df = 4,
    token_pattern=r'\w{1,}',
    stop_words= stopWords,
    ngram_range=(1, 1),
    max_features=30000)
vectorizer = tfidf.fit(train['labelAnnotations_des'].fillna("NULL"))
train_labelAnnotations_des = tfidf.transform(train['labelAnnotations_des'].fillna("NULL"))
test_labelAnnotations_des = tfidf.transform(test['labelAnnotations_des'].fillna("NULL"))


# In[ ]:


tmp = train.groupby(['RescuerID'])['len_des'].agg(['min','max','mean']).reset_index()
tmp.columns = ['RescuerID','min_len_des','max_len_des','mean_len_des']
train = train.merge(tmp, on=['RescuerID'], how='left')
tmp = test.groupby(['RescuerID'])['len_des'].agg(['min','max','mean']).reset_index()
tmp.columns = ['RescuerID','min_len_des','max_len_des','mean_len_des']
test = test.merge(tmp, on=['RescuerID'], how='left')


# In[ ]:


countvectorizer =  CountVectorizer(min_df=4)
vectorizer = countvectorizer.fit(all_data['Description'].fillna("NULL"))
train_count_des = countvectorizer.transform(train['Description'].fillna("NULL"))
test_count_des = countvectorizer.transform(test['Description'].fillna("NULL"))


# In[ ]:


countvectorizer =  CountVectorizer(min_df=4)
vectorizer = countvectorizer.fit(all_data['Name'].fillna("NULL"))
train_count_name = countvectorizer.transform(train['Name'].fillna("NULL"))
test_count_name = countvectorizer.transform(test['Name'].fillna("NULL"))


# In[ ]:


from keras.models import Sequential
from keras.models import Model, load_model
from keras.layers import CuDNNGRU, CuDNNLSTM, Dense, Bidirectional, Input, SpatialDropout1D,Embedding, Dropout,        BatchNormalization, GlobalMaxPooling1D, GlobalAveragePooling1D, concatenate, Conv1D, Multiply, Add,Flatten, Ave
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints
from keras import backend as K
from keras.initializers import he_uniform
from keras.utils.np_utils import to_categorical
from keras.metrics import top_k_categorical_accuracy
def top_2_accuracy(x,y): return top_k_categorical_accuracy(x,y, 2)


# In[ ]:


img_feats = np.load("../input/darling-lgbm-output-img-feats/img_feats.npy")


# In[ ]:


img_id = pd.read_csv("../input/darling-lgbm-output-img-feats/metadata_img.csv")


# In[ ]:


lb = LabelEncoder()
img_id['Breed1'] = lb.fit_transform(img_id['Breed1'])


# In[ ]:


def build_model(state, img_feats_xception):
    x1 = Embedding(img_id['Breed1'].max() + 1, 10,trainable= True)(state)
    #x2 = Embedding(img_feats_id['Breed1'].max() + 1, 60,trainable= True)(breed1)
    
    x = Dense(196, activation="sigmoid",kernel_initializer=he_uniform(seed=0))(img_feats_xception)
    x = concatenate([Flatten()(x1),x])
    #x = Dense(64, activation="sigmoid",kernel_initializer=he_uniform(seed=0))(x)
    #x = Dense(64, activation="sigmoid",kernel_initializer=he_uniform(seed=0))(x)
    #x = Dense(128, activation="relu",kernel_initializer=he_uniform(seed=0))(x)
    #x = concatenate([Flatten()(x1),Flatten()(x2), x3])
    x = Dense(5, activation='softmax',kernel_initializer=he_uniform(seed=0))(x)
    return x


# In[ ]:


img_oof_cache = []; f = 0
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=500)
for dev_index, val_index in (kf.split(user_ID, user_ID['class'])):
    dev_user, val_user = user_ID.iloc[dev_index].RescuerID, user_ID.iloc[val_index].RescuerID
    X_train_idx, X_val_idx = list(img_id[img_id['RescuerID'].isin(dev_user)].index.values),                     list(img_id[img_id['RescuerID'].isin(val_user)].index.values)
    img_feats_xception = Input((2048,), name="i1")
    state = Input(shape=[1], name="i2")
    #breed1 = Input(shape=[1], name="i3")
    
    #input_layer_name = Input((50, ), name = "i2")
    output_layer = build_model(state, img_feats_xception)
    model = Model([state, img_feats_xception], output_layer)
    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['categorical_accuracy', top_2_accuracy])
    if (f == 0):
        model.summary()
        f += 1
    dev_img_feats, dev_other_feats = img_feats[X_train_idx], img_id.iloc[X_train_idx]#['AdoptionSpeed'].values
    val_img_feats, val_other_feats = img_feats[X_val_idx], img_id.iloc[X_val_idx]#['AdoptionSpeed'].values
    model.fit({"i1":dev_img_feats,"i2":dev_other_feats['Breed1']}, to_categorical(dev_other_feats['AdoptionSpeed'].values),              batch_size=32, verbose=2,shuffle=True,epochs=30,              validation_data=({"i1":val_img_feats,"i2":val_other_feats['Breed1']}, to_categorical(val_other_feats['AdoptionSpeed'].values)))
    img_oof = pd.DataFrame()
    img_oof['PetID'] =  val_other_feats['PetID'].values
    val_pres =   model.predict({"i1":val_img_feats,"i2":val_other_feats['Breed1']}, batch_size=128, verbose=2)
    for i in range(5):
        img_oof['target_img_' + str(i)] = val_pres[:,i]
    img_oof_cache.append(img_oof)


# In[ ]:


img_oof = pd.concat(img_oof_cache, ignore_index=True)
img_oof = img_oof.groupby(['PetID']).mean().reset_index()
for i in range(5):
    c = 'target_img_' + str(i)
    if c in train.columns:
        train.drop([c], axis=1, inplace = True)
    train = train.merge(img_oof[['PetID',c]], on=['PetID'], how='left')
    test[c] = 0


# In[ ]:


train.columns


# In[ ]:


from sklearn.decomposition import PCA
pca = PCA(n_components=5, random_state = 10)
tmp = pd.DataFrame(pca.fit_transform(img_feats))
tmp.columns = ['pca_' + str(i) for i in range(5)]
tmp['PetID'] = img_id['PetID']
tmp = tmp.groupby(['PetID']).mean().reset_index()
for i in range(5):
    c = 'pca_' + str(i)
    if c in train.columns:
        train.drop([c], axis=1, inplace = True)
    train = train.merge(tmp[['PetID',c]], on=['PetID'], how='left')
    test[c] = 0


# In[ ]:


train['Description'].str.count(" ")


# In[ ]:


res = []; n_train = train.shape[0]; output = 0
params = {
        'boosting': 'gbdt',
        'objective': 'regression',
        'metric': 'rmse',
        'learning_rate': 0.01,
        'num_leaves': 30,  
        'max_depth': 15,  
        'colsample_bytree': 0.2,  
        'min_data_in_leaf':50,
        'bagging_freq': 10,
        'bagging_fraction': 0.7,
        'min_data_per_group': 20,
        'seed': 100
            }
train['len_des_1'] = train['Description'].fillna("").astype(str).str.len()
test['len_des_1'] = test['Description'].fillna("").astype(str).str.len()
train['len_word'] = train['Description'].fillna("").str.count(" ")
test['len_word'] = test['Description'].fillna("").str.count(" ")
train['NoName'] = ((train['Name'].fillna("NULL").str.contains("No Name")) | train['Name'].isnull()).astype('int')
test['NoName'] = ((test['Name'].fillna("NULL").str.contains("No Name")) | test['Name'].isnull()).astype('int')
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=500)
predictors = ['magnitude','Age','Type','Breed1','Breed2','Color1','Color2','Color3','PhotoAmt','VideoAmt','State',              'Sterilized','Vaccinated','Quantity','Fee','Fee_v1','MaturitySize', 'Gender','FurLength','Health',             'cntPet','len_des','len_des_1','min_age','max_age','mean_age','min_Fee','max_Fee',              'mean_Fee','min_PhotoAmt','max_PhotoAmt','mean_PhotoAmt',              'min_Quantity','max_Quantity','mean_Quantity','min_VideoAmt','max_VideoAmt','mean_VideoAmt',
             'min_MaturitySize_1','max_MaturitySize_1','mean_MaturitySize_1','min_Health','max_Health','mean_Health',\
             'min_score_labelAnnotations','max_score_labelAnnotations','mean_score_labelAnnotations',\
              'min_score_labelAnnotations_1','max_score_labelAnnotations_1','mean_score_labelAnnotations_1',\
              'min_x','max_x', 'mean_x','min_y','max_y','mean_y',
              'min_pixelscore','max_pixelscore', 'mean_pixelscore','min_pixelFraction','max_pixelFraction','mean_pixelFraction',\
               'min_red_0','max_red_0', 'mean_red_0','min_green_0','max_green_0','mean_green_0',\
               'min_blue_0','max_blue_0', 'mean_blue_0',
           'min_len_des','max_len_des','mean_len_des', 
             #'mean_score_labelAnnotations_2'
             #'min_confidence','max_confidence','mean_confidence'
             ] + list(count_cols)# + ['pca_' + str(i) for i in range(5)]
              #  + list(cnt_breeds.columns[1:])# + list(cnt_health.columns[1:])
categorical = ['Type','State','Gender','Breed1','Breed2','Color1','Color2','Color3','Fee_v1','MaturitySize', 'FurLength','Health']

Xspr_train = csr_matrix(hstack([csr_matrix(train[predictors]), train_count_des, train_labelAnnotations_des,                                X_Breed1[:n_train], X_Breed2[:n_train], X_State[:n_train],                                X_Color1[:n_train], X_Color2[:n_train], X_Color3[:n_train]]))
Xspr_test = csr_matrix(hstack([csr_matrix(test[predictors]), test_count_des, test_labelAnnotations_des,                                X_Breed1[n_train:], X_Breed2[n_train:], X_State[n_train:],                                X_Color1[n_train:], X_Color2[n_train:], X_Color3[n_train:]]))
for dev_index, val_index in (kf.split(user_ID, user_ID['class'])):
    dev_user, val_user = user_ID.iloc[dev_index].RescuerID, user_ID.iloc[val_index].RescuerID
    #X_train, X_val = train[train['RescuerID'].isin(dev_user)],train[train['RescuerID'].isin(val_user)]
    X_train_idx, X_val_idx = list(train[train['RescuerID'].isin(dev_user)].index.values),                     list(train[train['RescuerID'].isin(val_user)].index.values)
    X_val = train.iloc[X_val_idx]
    print(len(X_train_idx), len(X_val_idx))
    dtrain = lgb.Dataset(Xspr_train[X_train_idx], label=train.iloc[X_train_idx]['AdoptionSpeed'].values,
                              feature_name=predictors+ ['tfidf_' + str(i) for i in range(\
                                                            train_count_des.shape[1]+ train_labelAnnotations_des.shape[1])] +[str(i) for i in range(more_dim)],
                              categorical_feature=categorical
                              )
    dvalid = lgb.Dataset(Xspr_train[X_val_idx], label=train.iloc[X_val_idx]['AdoptionSpeed'].values,
                          feature_name=predictors + ['tfidf_' for i in range(
                                                            train_count_des.shape[1] + train_labelAnnotations_des.shape[1])] +[str(i) for i in range(more_dim)],
                          categorical_feature=categorical
                          )
    lgb_model = lgb.train(params, dtrain, 
                         valid_sets=[dtrain, dvalid], valid_names=['train','valid'], 
                         num_boost_round= 3000,early_stopping_rounds=300,
                        verbose_eval=100,)
    oof = pd.DataFrame()
    oof['PetID'] = X_val['PetID']
    oof['target'] = X_val['AdoptionSpeed']
    oof['preds'] = lgb_model.predict(Xspr_train[X_val_idx], num_iteration=lgb_model.best_iteration)
    res.append(oof)
    output +=  lgb_model.predict(Xspr_test, num_iteration=lgb_model.best_iteration)
oof = pd.concat(res, ignore_index=True)
quadratic_weighted_kappa(oof['target'], get_adoptionspeed(oof['preds']))


# In[ ]:





# In[ ]:


submission = pd.DataFrame({'PetID': test['PetID'].values, 'RescuerID': test['RescuerID'].values,                            'AdoptionSpeed': get_adoptionspeed(output)})


# In[ ]:


tmp = submission.groupby(['RescuerID'])['AdoptionSpeed'].apply(lambda x: x.mode()[0]).reset_index()
submission.drop(['AdoptionSpeed'], axis=1, inplace = True)
submission = submission.merge(tmp, on=['RescuerID'], how='left')


# In[ ]:


submission[['PetID','AdoptionSpeed']].to_csv('submission.csv', index=False)


# In[ ]:





# In[ ]:


submission['AdoptionSpeed'].value_counts()

