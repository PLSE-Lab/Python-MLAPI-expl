#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import lightgbm as lgb
from sklearn import *
import pandas as pd
import numpy as np

#from top scoring kernels and blends - for testing only
sub1 = pd.read_csv('../input/best-ensemble-score-made-available-0-68/SHAZ13_ENS_LEAKS.csv')
#sub2 = pd.read_csv('../input/best-ensemble-score-made-available-0-67/SHAZ13_ENS_LEAKS.csv')
sub3 = pd.read_csv('../input/feature-scoring-vs-zeros/leaky_submission.csv')
sub2 = pd.read_csv('../input/feature-scoring-vs-zeros/leaky_submission.csv')

#standard
train = pd.read_csv('../input/santander-value-prediction-challenge/train.csv')
test = pd.read_csv('../input/santander-value-prediction-challenge/test.csv')
print(train.shape, test.shape)


# In[ ]:


from PIL import Image, ImageDraw, ImageColor

col = [c for c in train.columns if c not in ['ID', 'target']]
xtrain = train[col].copy().values
target = train['target'].values

im = Image.new('RGBA', xtrain.shape)
wh = ImageColor.getrgb('white')
re = ImageColor.getrgb('red')
gr = ImageColor.getrgb('green')
ga = ImageColor.getrgb('gray')

for x in range(xtrain.shape[0]):
    for y in range(xtrain.shape[1]):
        if xtrain[x][y] == 0:
            im.putpixel((x,y), wh)
        elif xtrain[x][y] == target[x]:
            im.putpixel((x,y), re)
        elif (np.abs(xtrain[x][y] - target[x]) / target[x]) < 0.05:
            im.putpixel((x,y), gr)
        else:
            im.putpixel((x,y), ga)
im.save('leak.bmp')


# In[ ]:


leak_col = []
for c in col:
    leak1 = np.sum((train[c]==train['target']).astype(int))
    leak2 = np.sum((((train[c] - train['target']) / train['target']) < 0.05).astype(int))
    if leak1 > 30 and leak2 > 3500:
        leak_col.append(c)
print(len(leak_col))


# In[ ]:


col = list(leak_col)
train = train[col +  ['ID', 'target']]
test = test[col +  ['ID']]


# In[ ]:


#https://www.kaggle.com/johnfarrell/baseline-with-lag-select-fake-rows-dropped
train["nz_mean"] = train[col].apply(lambda x: x[x!=0].mean(), axis=1)
train["nz_max"] = train[col].apply(lambda x: x[x!=0].max(), axis=1)
train["nz_min"] = train[col].apply(lambda x: x[x!=0].min(), axis=1)
train["ez"] = train[col].apply(lambda x: len(x[x==0]), axis=1)
train["mean"] = train[col].apply(lambda x: x.mean(), axis=1)
train["max"] = train[col].apply(lambda x: x.max(), axis=1)
train["min"] = train[col].apply(lambda x: x.min(), axis=1)

test["nz_mean"] = test[col].apply(lambda x: x[x!=0].mean(), axis=1)
test["nz_max"] = test[col].apply(lambda x: x[x!=0].max(), axis=1)
test["nz_min"] = test[col].apply(lambda x: x[x!=0].min(), axis=1)
test["ez"] = test[col].apply(lambda x: len(x[x==0]), axis=1)
test["mean"] = test[col].apply(lambda x: x.mean(), axis=1)
test["max"] = test[col].apply(lambda x: x.max(), axis=1)
test["min"] = test[col].apply(lambda x: x.min(), axis=1)
col += ['nz_mean', 'nz_max', 'nz_min', 'ez', 'mean', 'max', 'min']


# In[ ]:


for i in range(2, 100):
    train['index'+str(i)] = ((train.index + 2) % i == 0).astype(int)
    test['index'+str(i)] = ((test.index + 2) % i == 0).astype(int)
    col.append('index'+str(i))


# In[ ]:


test = pd.merge(test, sub1, how='left', on='ID',)


# In[ ]:


from scipy.sparse import csr_matrix, vstack
train = train.replace(0, np.nan)
test = test.replace(0, np.nan)
train = pd.concat((train, test), axis=0, ignore_index=True)


# In[ ]:


test['target'] = 0.0
folds = 5
for fold in range(folds):
    x1, x2, y1, y2 = model_selection.train_test_split(train[col], np.log1p(train.target.values), test_size=0.20, random_state=fold)
    params = {'learning_rate': 0.02, 'max_depth': 7, 'boosting': 'gbdt', 'objective': 'regression', 'metric': 'rmse', 'is_training_metric': True, 'feature_fraction': 0.9, 'bagging_fraction': 0.8, 'bagging_freq': 5, 'seed':fold}
    model = lgb.train(params, lgb.Dataset(x1, label=y1), 3000, lgb.Dataset(x2, label=y2), verbose_eval=200, early_stopping_rounds=100)
    test['target'] += np.expm1(model.predict(test[col], num_iteration=model.best_iteration))
test['target'] /= folds
test[['ID', 'target']].to_csv('submission.csv', index=False)


# In[ ]:


b1 = sub1.rename(columns={'target':'dp1'})
b2 = pd.read_csv('submission.csv').rename(columns={'target':'dp2'})
b1 = pd.merge(b1, b2, how='left', on='ID')
b1['target'] = (b1['dp1'] * 0.8) + (b1['dp2'] * 0.2)
b1[['ID','target']].to_csv('blend01.csv', index=False)


# In[ ]:


b1 = sub2.rename(columns={'target':'dp1'})
b2 = pd.read_csv('blend01.csv').rename(columns={'target':'dp2'})
b1 = pd.merge(b1, b2, how='left', on='ID')
b1['target'] = (b1['dp1'] * 0.8) + (b1['dp2'] * 0.2)
b1[['ID','target']].to_csv('blend02.csv', index=False)


# In[ ]:


b1 = sub2.rename(columns={'target':'dp1'})
b2 = pd.read_csv('blend02.csv').rename(columns={'target':'dp2'})
b1 = pd.merge(b1, b2, how='left', on='ID')
b1['target'] = (b1['dp1'] * 0.5) + (b1['dp2'] * 0.5)
b1[['ID','target']].to_csv('blend03.csv', index=False)


# In[ ]:


b1 = sub3.rename(columns={'target':'dp1'})
b2 = pd.read_csv('blend03.csv').rename(columns={'target':'dp2'})
b1 = pd.merge(b1, b2, how='left', on='ID')
b1['target'] = (b1['dp1'] * 0.6) + (b1['dp2'] * 0.4)
b1[['ID','target']].to_csv('blend04.csv', index=False)


# In[ ]:


#!rm submission.csv
#!rm blend01.csv
#!rm blend02.csv
#!rm blend03.csv

