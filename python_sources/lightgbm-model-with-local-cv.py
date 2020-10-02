#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# General imports
import numpy as np
import pandas as pd
import lightgbm as lgb
import sklearn
import random, os, gc, warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score


# In[ ]:


# General settings
seed = 42
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed) 
TARGET = 'Cover_Type'


# In[ ]:


# Import files
train = pd.read_csv("../input/learn-together/train.csv")
train = train.drop(["Id"], axis = 1)
test = pd.read_csv("../input/learn-together/test.csv")
test_ids = test["Id"]
test = test.drop(["Id"], axis = 1)
test[TARGET] = 0
train.head()


# In[ ]:


# Basic helper functions
def make_predictions(tr_df, tt_df, target, lgb_params, NFOLDS=2):
    folds = KFold(n_splits=NFOLDS, shuffle=True, random_state=seed)
    features_columns = [i for i in tr_df.columns if i != target]
    X,y = tr_df[features_columns], tr_df[target]    
    P,P_y = tt_df[features_columns], tt_df[target]  
    fold_p = np.zeros((P.shape[0], 7))
    oof = np.zeros(len(tr_df))
    for fold_, (trn_idx, val_idx) in enumerate(folds.split(X, y)):
        print('Fold:',fold_)
        tr_x, tr_y = X.iloc[trn_idx,:], y[trn_idx]
        vl_x, vl_y = X.iloc[val_idx,:], y[val_idx]
        print(len(tr_x),len(vl_x))
        clf = lgb.LGBMClassifier(**lgb_params)   
        clf.fit(tr_x, tr_y, eval_set=[(tr_x, tr_y), (vl_x, vl_y)], verbose=50)
        temp_p = clf.predict_proba(P)
        fold_p += temp_p/NFOLDS
        oof_p = clf.predict_proba(X.iloc[val_idx,:])
        oof[val_idx] = 1 + np.argmax(oof_p, axis=1)
        del tr_x, tr_y, vl_x, vl_y
        gc.collect()
    predictions = 1 + np.argmax(fold_p, axis=1)
    print('OOF ACCURACY:', accuracy_score(y, oof))
    return predictions

def make_test_predictions(tr_df, tt_df, target, lgb_params, NFOLDS=2):
    features_columns = [i for i in tr_df.columns if i != target]
    X,y = tr_df[features_columns], tr_df[target]    
    P,P_y = tt_df[features_columns], tt_df[target]  
    for col in list(X):
        if X[col].dtype=='O':
            X[col] = X[col].fillna('unseen_before_label')
            P[col] = P[col].fillna('unseen_before_label')
            X[col] = train_df[col].astype(str)
            P[col] = test_df[col].astype(str)
            le = LabelEncoder()
            le.fit(list(X[col])+list(P[col]))
            X[col] = le.transform(X[col])
            P[col]  = le.transform(P[col])
            X[col] = X[col].astype('category')
            P[col] = P[col].astype('category')
    clf = lgb.LGBMClassifier(**lgb_params)   
    clf.fit(X, y, eval_set = [(X, y)], verbose=10)    
    proba_p = clf.predict_proba(P)
    predictions = 1 + np.argmax(proba_p, axis=1)
    print(predictions.shape)
    return predictions


# In[ ]:


lgb_params = {
    'n_estimators': 500,
    'seed': seed,
    'early_stopping_rounds':100, 
}


# In[ ]:


test_pred = make_predictions(train, test, target=TARGET, lgb_params=lgb_params, NFOLDS=5)


# Clearly, this is overfit!

# In[ ]:


# Generate submission
output = pd.DataFrame({'Id': test_ids,'Cover_Type': test_pred})
output.to_csv('submission.csv', index=False)

