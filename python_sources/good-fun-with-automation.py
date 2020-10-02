#!/usr/bin/env python
# coding: utf-8

# This notebook uses the automatic features generated to make a prediction. The way the final score is put together using lgb doesn't change from the original found here https://www.kaggle.com/ogrellier/good-fun-with-ligthgbm/code . 
# 
# Feel free to explore the newly created variables or ask any questions !

# In[2]:


import pandas as pd
train = pd.read_csv("../input/automation-of-feature-creation/train.csv")
test = pd.read_csv("../input/automation-of-feature-creation/test.csv")
# Any results you write to the current directory are saved as output.
tmp = pd.read_csv("../input/home-credit-default-risk/application_test.csv")
tmp_train = pd.read_csv("../input/home-credit-default-risk/application_train.csv")


# In[3]:


train['SK_ID_CURR'] = tmp_train['SK_ID_CURR']
test['SK_ID_CURR'] = tmp['SK_ID_CURR']


# In[4]:


probss = test['ProbTARGET1']
del test['ProbTARGET1']
del train['ProbTARGET1']
y = train['TARGET']
del train['TARGET']


# In[5]:


def mean_(x):
    if '{' in x:
        return x
    if x=='Missing':
        return -1
    if '+inf' in x:
        return float(x.replace(']','').replace('[','').split(';')[0])
    if '-inf' in x:
        return float(x.replace(']','').replace('[','').split(';')[1])
    l = x.replace(']','').replace('[','').split(';')
    return (float(l[0])+float(l[1]))/2

dictionnary = {}
for i in train.columns:
    if train[i].dtype!='object':
        continue
    try :
        for j in train[i].unique():
            dictionnary[j] = mean_(j)
    except :
        continue
for i in train.columns:
    if train[i].dtype!='object':
        continue
    train[i] = train[i].map(dictionnary)


# In[6]:


for i in test.columns:
    if test[i].dtype!='object':
        continue
    try :
        for j in test[i].unique():
            dictionnary[j] = mean(j)
    except :
        continue
for i in test.columns:
    if test[i].dtype!='object':
        continue
    test[i] = test[i].map(dictionnary)


# In[7]:


to_dummy =[]
for i in test.columns:
    if test[i].dtype==object:
        try :
            test[i]=test[i].astype(float)
        except :
            continue
        if len(test[i].unique())<5:
            to_dummy.append(i)
        else :
            del test[i]
            #print(i)
test = pd.get_dummies(test,columns=to_dummy)

to_dummy =[]
for i in train.columns:
    if train[i].dtype==object:
        try :
            train[i]=train[i].astype(float)
        except :
            continue
        if len(train[i].unique())<5:
            to_dummy.append(i)
        else :
            del train[i]
            #print(i)
train = pd.get_dummies(train,columns=to_dummy)


# In[8]:


train.fillna(0,inplace=True)
test.fillna(0,inplace=True)


# In[9]:


col_dict = {}
for i in train.columns:
    if '>' in i:
        col_dict[i] = i.replace('>','').replace('<','')
train.rename(columns=col_dict,inplace = True)
test.rename(columns=col_dict, inplace =True)


# In[10]:


for i in train.columns:
    if train[i].dtype==object:
        del train[i]
for i in test.columns:
    if test[i].dtype==object:
        del test[i]
        
for i in train.columns:
    if i not in test.columns:
        del train[i]
        
for j in test.columns:
    if j not in train.columns:
        del test[j]


# In[ ]:


import numpy as np

from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve, average_precision_score
from sklearn.model_selection import KFold
from lightgbm import LGBMClassifier

import matplotlib.pyplot as plt
import seaborn as sns
import gc


def train_model(data_, test_, y_, folds_):

    oof_preds = np.zeros(data_.shape[0])
    sub_preds = np.zeros(test_.shape[0])
    
    feature_importance_df = pd.DataFrame()
    
    feats = [f for f in data_.columns if f not in ['SK_ID_CURR']]
    
    for n_fold, (trn_idx, val_idx) in enumerate(folds_.split(data_)):
        trn_x, trn_y = data_[feats].iloc[trn_idx], y_.iloc[trn_idx]
        val_x, val_y = data_[feats].iloc[val_idx], y_.iloc[val_idx]
        
        clf = LGBMClassifier(
            n_estimators=6000,
            learning_rate=0.02,
            num_leaves=40,
            colsample_bytree=.8,
            subsample=.8,
            max_depth=12,
            reg_alpha=.1,
            reg_lambda=.1,
            min_split_gain=.008,
            min_child_weight=2,
            min_child_samples=35,
            silent=-1,
            verbose=-1,
        )
        
        clf.fit(trn_x, trn_y, 
                eval_set= [(trn_x, trn_y), (val_x, val_y)], 
                eval_metric='auc', verbose=500, early_stopping_rounds=300  #30
               )
        
        oof_preds[val_idx] = clf.predict_proba(val_x, num_iteration=clf.best_iteration_)[:, 1]
        sub_preds += clf.predict_proba(test_[feats], num_iteration=clf.best_iteration_)[:, 1] / folds_.n_splits
        
        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = feats
        fold_importance_df["importance"] = clf.feature_importances_
        fold_importance_df["fold"] = n_fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        
        print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(val_y, oof_preds[val_idx])))
        del clf, trn_x, trn_y, val_x, val_y
        gc.collect()
        
    print('Full AUC score %.6f' % roc_auc_score(y, oof_preds)) 
    
    test_['TARGET'] = sub_preds

    return oof_preds, test_[['SK_ID_CURR', 'TARGET']], feature_importance_df
    

def display_importances(feature_importance_df_):
    # Plot feature importances
    cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(
        by="importance", ascending=False)[:50].index
    
    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]
    
    plt.figure(figsize=(8,10))
    sns.barplot(x="importance", y="feature", 
                data=best_features.sort_values(by="importance", ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    #plt.tight_layout()
    plt.savefig('lgbm_importances.png')


def display_roc_curve(y_, oof_preds_, folds_idx_):
    # Plot ROC curves
    plt.figure(figsize=(6,6))
    scores = [] 
    for n_fold, (_, val_idx) in enumerate(folds_idx_):  
        # Plot the roc curve
        fpr, tpr, thresholds = roc_curve(y_.iloc[val_idx], oof_preds_[val_idx])
        score = roc_auc_score(y_.iloc[val_idx], oof_preds_[val_idx])
        scores.append(score)
        plt.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC fold %d (AUC = %0.4f)' % (n_fold + 1, score))
    
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Luck', alpha=.8)
    fpr, tpr, thresholds = roc_curve(y_, oof_preds_)
    score = roc_auc_score(y_, oof_preds_)
    plt.plot(fpr, tpr, color='b',
             label='Avg ROC (AUC = %0.4f $\pm$ %0.4f)' % (score, np.std(scores)),
             lw=2, alpha=.8)
    
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('LightGBM ROC Curve')
    plt.legend(loc="lower right")
    #tight_layoutplt.tight_layout()
    
    plt.savefig('roc_curve.png')


def display_precision_recall(y_, oof_preds_, folds_idx_):
    # Plot ROC curves
    plt.figure(figsize=(6,6))
    
    scores = [] 
    for n_fold, (_, val_idx) in enumerate(folds_idx_):  
        # Plot the roc curve
        fpr, tpr, thresholds = roc_curve(y_.iloc[val_idx], oof_preds_[val_idx])
        score = average_precision_score(y_.iloc[val_idx], oof_preds_[val_idx])
        scores.append(score)
        plt.plot(fpr, tpr, lw=1, alpha=0.3, label='AP fold %d (AUC = %0.4f)' % (n_fold + 1, score))
    
    precision, recall, thresholds = precision_recall_curve(y_, oof_preds_)
    score = average_precision_score(y_, oof_preds_)
    plt.plot(precision, recall, color='b',
             label='Avg ROC (AUC = %0.4f $\pm$ %0.4f)' % (score, np.std(scores)),
             lw=2, alpha=.8)
    
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('LightGBM Recall / Precision')
    plt.legend(loc="best")
    #plt.tight_layout()
    
    plt.savefig('recall_precision_curve.png')


# In[ ]:


if __name__ == '__main__':
    gc.enable()
    # Build model inputs
    #data, test, y = build_model_input()
    # Create Folds
    folds = KFold(n_splits=5, shuffle=True, random_state=123)
    # Train model and get oof and test predictions
    oof_preds, test_preds, importances = train_model(train, test, y, folds)
    # Save test predictions
    test_preds.to_csv('first_automated_submission.csv', index=False)
    # Display a few graphs
    folds_idx = [(trn_idx, val_idx) for trn_idx, val_idx in folds.split(train)]
    display_importances(feature_importance_df_=importances)
    display_roc_curve(y_=y, oof_preds_=oof_preds, folds_idx_=folds_idx)
    display_precision_recall(y_=y, oof_preds_=oof_preds, folds_idx_=folds_idx)


# In[ ]:





# In[ ]:




