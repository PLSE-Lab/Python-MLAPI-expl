#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from scipy.sparse import coo_matrix, hstack
from catboost import CatBoostRegressor
from matplotlib import pyplot
import lightgbm as lgb
import xgboost as xgb
from PIL import Image
from sklearn import *
from glob import glob
import pandas as pd
import numpy as np
import json

train = pd.read_csv('../input/train/train.csv').fillna(-99)
test = pd.read_csv('../input/test/test.csv').fillna(-99)
sub = pd.read_csv('../input/test/sample_submission.csv')
breed_labels = pd.read_csv('../input/breed_labels.csv')
#state_labels = pd.read_csv('../input/state_labels.csv')
#color_labels = pd.read_csv('../input/color_labels.csv')

more_data = []
for path in ['train_images', 'train_metadata', 'train_sentiment', 'test_images', 'test_metadata', 'test_sentiment']:
    more_data += list(glob('../input/'+path+'/**'))
more_data = pd.DataFrame(more_data, columns=['path'])
more_data['type1'] = more_data['path'].map(lambda x: x.split('/')[2].split('_')[0])
more_data['type2'] = more_data['path'].map(lambda x: x.split('/')[2].split('_')[1])
more_data['PetID'] = more_data['path'].map(lambda x: x.split('/')[3].split('-')[0].split('.')[0])

print(train.shape, test.shape, breed_labels.shape)


# In[ ]:


def get_sentiment(path):
    d = json.load(open(path))
    return d['documentSentiment']['score']

def get_magnitude(path):
    d = json.load(open(path))
    return d['documentSentiment']['magnitude']

sentiment = more_data[more_data['type2']=='sentiment'].copy()
sentiment['Sentiment_score'] = sentiment['path'].map(lambda x: get_sentiment(x))
sentiment['Magnitude_score'] = sentiment['path'].map(lambda x: get_magnitude(x))
train = pd.merge(train, sentiment[['PetID', 'Sentiment_score', 'Magnitude_score']], how='left', on=['PetID']).fillna(-99).reset_index(drop=True)
test = pd.merge(test, sentiment[['PetID', 'Sentiment_score', 'Magnitude_score']], how='left', on=['PetID']).fillna(-99).reset_index(drop=True)


# In[ ]:


#https://www.kaggle.com/abhishek/maybe-something-interesting-here
def pet_image_info(PetIDArr, tt='train'):
    df = []
    for pet in PetIDArr:
        try:
            with open('../input/' + tt + '_metadata/' + pet + '-1.json', 'r') as f:
                data = json.load(f)
            vertex_xs = data['cropHintsAnnotation']['cropHints'][0]['boundingPoly']['vertices'][2]['x']
            vertex_ys = data['cropHintsAnnotation']['cropHints'][0]['boundingPoly']['vertices'][2]['y']
            bounding_confidences = data['cropHintsAnnotation']['cropHints'][0]['confidence']
            bounding_importance_fracs = data['cropHintsAnnotation']['cropHints'][0].get('importanceFraction', -1)
            dominant_blues = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['color']['blue']
            dominant_greens = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['color']['green']
            dominant_reds = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['color']['red']
            dominant_pixel_fracs = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['pixelFraction']
            dominant_scores = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['score']
            if data.get('labelAnnotations'):
                label_descriptions = data['labelAnnotations'][0]['description']
                label_scores = data['labelAnnotations'][0]['score']
            else:
                label_descriptions = 'nothing'
                label_scores = -1
        except FileNotFoundError:
            vertex_xs = -1
            vertex_ys = -1
            bounding_confidences = -1
            bounding_importance_fracs = -1
            dominant_blues = -1
            dominant_greens = -1
            dominant_reds = -1
            dominant_pixel_fracs = -1
            dominant_scores = -1
            label_descriptions = -1
            label_scores = -1
            
        df.append([pet, vertex_xs, vertex_ys, bounding_confidences, bounding_importance_fracs, dominant_blues, dominant_greens, dominant_reds, dominant_pixel_fracs, dominant_scores, label_descriptions, label_scores])
    df = pd.DataFrame(df, columns=['PetID', 'vertex_xs', 'vertex_ys', 'bounding_confidences', 'bounding_importance_fracs', 'dominant_blues', 'dominant_greens', 'dominant_reds', 'dominant_pixel_fracs', 'dominant_scores', 'label_descriptions', 'label_scores'])
    return df

train = pd.merge(train, pet_image_info(train.PetID.values, 'train'), how='left', on=['PetID'])
test = pd.merge(test, pet_image_info(test.PetID.values, 'test'), how='left', on=['PetID'])
train.shape, test.shape


# In[ ]:


def description_features(df):
    df['Description_len'] = df['Description'].map(lambda x: len(str(x)))
    df['Description_wc'] = df['Description'].map(lambda x: len(str(x).split(' ')))
    df['Description_wcu'] = df['Description'].map(lambda x: len(set(str(x).split(' '))))
    df["Description_mwl"] = df['Description'].map(lambda x: np.mean([len(w) for w in str(x).split()]))
    df['Description_wcu%'] = df['Description_wcu'] / df['Description_wc']
    return df

train = description_features(train); print(train.shape)
test = description_features(test); print(test.shape)


# In[ ]:


tfidf = feature_extraction.text.TfidfVectorizer(min_df=3,  max_features=10000, strip_accents='unicode', token_pattern=r'\w{1,}', ngram_range=(1, 3), sublinear_tf=True, stop_words = 'english')
svd = decomposition.TruncatedSVD(n_components=120)

tfidf.fit(pd.concat((train.apply(lambda r: ' '.join([str(r['Name']), str(r['Description']), str(r['label_descriptions'])]), axis=1), 
                     test.apply(lambda r: ' '.join([str(r['Name']), str(r['Description']), str(r['label_descriptions'])]), axis=1))))

#trainf = hstack([coo_matrix(train[col]), tfidf.transform(train.apply(lambda r: ' '.join([str(r['Name']), str(r['Description']), str(r['label_descriptions'])]), axis=1).astype(str))]); print(trainf.shape)
#testf = hstack([coo_matrix(test[col]), tfidf.transform(test.apply(lambda r: ' '.join([str(r['Name']), str(r['Description']), str(r['label_descriptions'])]), axis=1).astype(str))]); print(testf.shape)

trainf = tfidf.transform(train.apply(lambda r: ' '.join([str(r['Name']), str(r['Description']), str(r['label_descriptions'])]), axis=1).astype(str))
trainf = svd.fit_transform(trainf)
trainf = pd.DataFrame(trainf, columns=['SVD_' + str(i).zfill(3) for i in range(120)])
testf = tfidf.transform(test.apply(lambda r: ' '.join([str(r['Name']), str(r['Description']), str(r['label_descriptions'])]), axis=1).astype(str))
testf = svd.fit_transform(testf)
testf = pd.DataFrame(testf, columns=['SVD_' + str(i).zfill(3) for i in range(120)])

train = pd.concat((train, trainf), axis=1); print(train.shape)
test = pd.concat((test, testf), axis=1); print(test.shape)


# In[ ]:


col = [c for c in train.columns if c not in ['PetID', 'AdoptionSpeed', 'Description', 'RescuerID', 'label_descriptions', 'Name']]


# In[ ]:


x1, x2, y1, y2 = model_selection.train_test_split(train[col], train['AdoptionSpeed'], test_size=0.2, random_state=5)
params = {'eta': 0.02, 'objective': 'reg:linear', 'max_depth': 7, 'subsample': 0.9, 'colsample_bytree': 0.9,  'eval_metric': 'rmse', 'seed': 3, 'silent': True}

def ks_xgb(pred, y):
    y = y.get_label()
    pred = pred.round().astype(int).clip(0,4)
    return 'kappa', metrics.cohen_kappa_score(y, pred, weights='quadratic')

watchlist = [(xgb.DMatrix(x1, y1), 'train'), (xgb.DMatrix(x2, y2), 'valid')]
model = xgb.train(params, xgb.DMatrix(x1, y1), 2500,  watchlist, feval=ks_xgb, maximize=True, verbose_eval=100, early_stopping_rounds=200)
test['AdoptionSpeed'] = (model.predict(xgb.DMatrix(test[col]), ntree_limit=model.best_ntree_limit))


# In[ ]:


xgb.plot_importance(model, importance_type='weight', max_num_features=20)


# In[ ]:


#https://www.kaggle.com/tayyabali55/complete-pet-finder-analysis-with-lgbm-4-0
x1, x2, y1, y2 = model_selection.train_test_split(train[col], train['AdoptionSpeed'], test_size=0.2, random_state=6)
params = {'learning_rate': 0.02,'max_depth': 9, 'num_leaves': 80, 'application': 'regression', 'boosting': 'gbdt', 'metric': 'rmse', 'seed': 3}

def ks_lgb(pred, dtrain):
    y = list(dtrain.get_label())
    pred = pred.round().astype(int).clip(0,4)
    score = metrics.cohen_kappa_score(y, pred, weights='quadratic')
    return 'kappa', score, True

model = lgb.train(params, lgb.Dataset(x1, label=y1), 2500, lgb.Dataset(x2, label=y2), feval=ks_lgb, verbose_eval=100, early_stopping_rounds=200)
test['AdoptionSpeed'] += model.predict(test[col], num_iteration=model.best_iteration)


# In[ ]:


lgb.plot_importance(model, importance_type='split', max_num_features=20)


# In[ ]:


#https://www.kaggle.com/skooch/petfinder-simple-catboost-baseline
#x1, x2, y1, y2 = model_selection.train_test_split(trainf.toarray(), train['AdoptionSpeed'], test_size=0.1, random_state=4)
x1, x2, y1, y2 = model_selection.train_test_split(train[col], train['AdoptionSpeed'], test_size=0.2, random_state=7)
params = {'depth': 9,'eta': 0.05, 'task_type' :'GPU', 'random_strength': 1.5, 'one_hot_max_size': 2,
          'reg_lambda': 6,'od_type': 'Iter', 'fold_len_multiplier': 2, 'border_count': 128,
          'bootstrap_type' : "Bayesian", 'bagging_temperature': 1,
          'random_seed': 217, 'early_stopping_rounds':100, 'num_boost_round': 2500}

model = CatBoostRegressor(**params)
model.fit(x1, y1, eval_set=(x2,y2), verbose=100)
print(metrics.cohen_kappa_score(y2, model.predict(x2).round().astype(int).clip(0,4), weights='quadratic'))

test['AdoptionSpeed'] += model.predict(test[col]) #
test['AdoptionSpeed'] = (test['AdoptionSpeed'] / 3).round().astype(int).clip(0,4) 
test[['PetID', 'AdoptionSpeed']].to_csv('submission.csv', index=False)


# In[ ]:


img = Image.open('../input/test_images/0df5238d7-13.jpg')
pyplot.imshow(img)


# In[ ]:


get_ipython().system('rm -r catboost_info')

