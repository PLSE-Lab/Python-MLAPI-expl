#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import gc
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score


# In[ ]:


train_df = pd.read_csv("../input/train.csv")
test_df  = pd.read_csv("../input/test.csv")
features = [x for x in train_df.columns if x.startswith("var")]


# In[ ]:


#Reverse some features.
#Not really necessary for LGB, but helps a little
for var in features:
    if np.corrcoef( train_df['target'], train_df[var] )[1][0] < 0:
        train_df[var] = train_df[var] * -1
        test_df[var]  = test_df[var]  * -1


# In[ ]:


#count train values to split Rare/NonRare values
var_stats = {}
for var in features:
    var_stats[var] = train_df[var].value_counts()


# In[ ]:


def logit(p):
    return np.log(p) - np.log(1 - p)

def var_to_feat(vr, var_stats, feat_id ):
    new_df = pd.DataFrame()
    new_df["var"] = vr.values
    new_df["hist"] = pd.Series(vr).map(var_stats)
    new_df["feature_id"] = feat_id
    new_df["var_rank"] = new_df["var"].rank()/200000.
    return new_df.values


# In[ ]:


TARGET = np.array( list(train_df['target'].values) * 200 )

TRAIN = []
var_mean = {}
var_var  = {}
for var in features:
    tmp = var_to_feat(train_df[var], var_stats[var], int(var[4:]) )
    var_mean[var] = np.mean(tmp[:,0]) 
    var_var[var]  = np.var(tmp[:,0])
    tmp[:,0] = (tmp[:,0]-var_mean[var])/var_var[var]
    TRAIN.append( tmp )
TRAIN = np.vstack( TRAIN )

target = train_df['target'].values
del train_df
_=gc.collect()

print( TRAIN.shape, len( TARGET ) )


# In[ ]:


model = lgb.LGBMClassifier(**{
     'learning_rate': 0.03,
     'num_leaves': 31,
     'max_bin': 1023,
     'min_child_samples': 1000,
     'feature_fraction': 1.0,
     'bagging_freq': 1,
     'bagging_fraction': 0.85,
     'objective': 'binary',
     'n_jobs': -1,
     'n_estimators':200,})

NFOLDS = 10
predtrain = np.zeros( len(TARGET) )
MODELS = []
skf = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=11111)
for fold_, (train_indexes, valid_indexes) in enumerate(skf.split(TRAIN, TARGET)):
    print('Fold:', fold_ )
    model = model.fit( TRAIN[train_indexes], TARGET[train_indexes],
                      eval_set = (TRAIN[valid_indexes], TARGET[valid_indexes]),
                      verbose = 100,
                      eval_metric='auc',
                      early_stopping_rounds=20,
                      categorical_feature = [2] )
    MODELS.append( model )
    predtrain[valid_indexes] = model.predict_proba( TRAIN[valid_indexes] )[:,1] 

#Reshape to original format 200k x 200
pred = np.reshape( predtrain , (200000,200) , order='F' )
#Use logit for better performance
print( NFOLDS,'-Fold CV AUC:',roc_auc_score( target, np.mean( logit(pred),axis=1)  ) )
_=gc.collect()


# In[ ]:


ypred = np.zeros( (200000,200) )
for feat,var in enumerate(features):
    #build dataset
    tmp = var_to_feat(test_df[var], var_stats[var], int(var[4:]) )
    #Standard Scale feature according train statistics
    tmp[:,0] = (tmp[:,0]-var_mean[var])/var_var[var]
    tmp[:,1] = tmp[:,1] + 1
    #Write 1 to frequency of values not seem in trainset
    tmp[ np.isnan(tmp) ] = 1
    #Predict testset for N folds
    for model_id in range(NFOLDS):
        model = MODELS[model_id]
        ypred[:,feat] += model.predict_proba( tmp )[:,1] / NFOLDS
ypred = np.mean( logit(ypred), axis=1 )

sub = test_df[['ID_code']]
sub['target'] = ypred
sub['target'] = sub['target'].rank() / 200000.
sub.to_csv('golden_sub.csv', index=False)
print( sub.head(20) )


# In[ ]:




