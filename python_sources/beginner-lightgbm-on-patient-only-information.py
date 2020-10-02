#!/usr/bin/env python
# coding: utf-8

# # Hi everyone !

# This first starter notebook is intended to give you an insight of building a model with only patient-level information (no image processing for the moment !). This simple LightGBM baseline can also be used as a more general pipeline for other projects :)

# # Please upvote this kernel if you found it interesting !

# # Start with preparing the patient-only data

# In[ ]:


import numpy as np 
import pandas as pd
import os
from sklearn.metrics import f1_score
import seaborn as sns
from scipy.misc import derivative
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, KFold
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score,confusion_matrix,roc_curve,roc_auc_score,classification_report


# In[ ]:


train = pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/train.csv')
test = pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/test.csv')
submission = pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/sample_submission.csv')


# ## Defining useful variables

# In[ ]:


train.columns, test.columns


# We will make use only of columns that are present both in train and test. It would be interesting, as future imporvement, to try and predict the column diagnosis in order to use it as an instrumental variable and help predict the target.

# In[ ]:


patient_only_cols = ['patient_id', 'sex', 'age_approx', 'anatom_site_general_challenge'] 
patient_only_train, patient_only_test = train[patient_only_cols+['target']].drop_duplicates(inplace=False), test[patient_only_cols].drop_duplicates(inplace=False)


# In[ ]:


patient_only_train.head()


# ## Pre-processing categorical variables

# LightGBM requires to specify, and transform to integers, all categorical variables. Note : we don't care about missing values, LGBM handles them quite well :)

# In[ ]:


categoricals = ['sex', 'anatom_site_general_challenge']


# What values do we have in the `sex` column ?

# In[ ]:


set(patient_only_train.sex.values.tolist())


# Matching them to integers :

# In[ ]:


matching_sex = {'female':1, 'male':0}


# What values do we have in the `anatom_site_general_challenge` column ?

# In[ ]:


set(patient_only_train.anatom_site_general_challenge.values.tolist())


# Matching them to integers :

# In[ ]:


matching_anatom = {'head/neck':0,
 'lower extremity':1,
 'oral/genital':2,
 'palms/soles':3,
 'torso':4,
 'upper extremity':5}


# Mapping those columns to their integer matched values :

# In[ ]:


patient_only_train.replace(to_replace={'anatom_site_general_challenge':matching_anatom, 'sex':matching_sex}, inplace=True)
patient_only_test.replace(to_replace={'anatom_site_general_challenge':matching_anatom, 'sex':matching_sex}, inplace=True)


# Finally, let's define the columns we will use for prediction :

# In[ ]:


Cols = ['sex', 'age_approx', 'anatom_site_general_challenge']
patient_only_train[Cols] = patient_only_train[Cols].astype('int32', errors='ignore')
patient_only_test[Cols] = patient_only_test[Cols].astype('int32', errors='ignore')
patient_only_train.head()


# # LightGBM Pipeline

# Even if this competition's metric is the AUC, I always find it better to optimize on the mean average precision, especially when classes are extremely unbalanced. Mean Average Precision is more sensible to improvements of respective scoring at the top of the ranking than at the bottom, contrary to AUC which is indifferent.

# In[ ]:


def MeanAveragePrecision(y_pred, y_true):
    y_true = y_true.get_label()
    df = pd.DataFrame({'true': y_true, 'pred_probas': y_pred})
    n = df.shape[0]
    df.sort_values(by='pred_probas', ascending=False, inplace=True)
    df['loss'] = df['true'].cumsum()/list(range(1, n+1))
    df = df.loc[df['true']==1, 'loss']
    return "MeanAveragePrecision", max(0, df.mean(axis=0)), True


# In[ ]:


def DataSetLgbm(Data,trn_idx,val_idx,target,features, categorical_features=""):
    trn_data=lgb.Dataset(Data.iloc[trn_idx][features], label=Data[target].iloc[trn_idx], categorical_feature=categorical_features)
    val_data=lgb.Dataset(Data.iloc[val_idx][features], label=Data[target].iloc[val_idx], categorical_feature=categorical_features)
    return trn_data,val_data

def TrainSimpleLgbm(Params,DataTrain,trn_idx,val_idx,target,features, categorical_features=""): 
    trn_data,val_data=DataSetLgbm(DataTrain,trn_idx,val_idx,target,features, categorical_features=categorical_features)
    clf=lgb.train(Params, trn_data, 30000, valid_sets = [trn_data, val_data],
                verbose_eval=100,feval = MeanAveragePrecision, early_stopping_rounds = 500)
    return clf

def mean_average_p(y_true, y_pred_p):
    df = pd.DataFrame({'true': y_true, 'pred_probas': y_pred_p})
    n = df.shape[0]
    df.sort_values(by='pred_probas', ascending=False, inplace=True)
    df['loss'] = df['true'].cumsum()/list(range(1, n+1))
    df = df.loc[df['true']==1, 'loss']
    return max(0, df.mean(axis=0))


# We will introduce to set of parameters for 2 different lgbms. Feel free to tamper with those and makeyour own experiments !

# In[ ]:


ParN1 = {
    'bagging_freq': 1,
    'bagging_fraction': 0.95,
    'boost_from_average':'true',
    'boost': 'gbdt',
    'feature_fraction': 0.5,
    'learning_rate': 0.04,
    'max_depth': -1,
    'metric':'auc',
    'is_unbalance':'true',
    'min_data_in_leaf':80,
    'lambda_l1' :1,
    'lambda_l2':1,
    'num_leaves': 2000,
    'colsample_bytree': 0.9,
    'tree_learner': 'serial',
    'objective': 'cross_entropy',
    'verbosity': 1}

ParN2 = {
    'bagging_freq': 20,
    'bagging_fraction': 0.9,
    'boost_from_average':'true',
    'boost': 'gbdt',
    'feature_fraction': 0.9,
    'learning_rate': 0.04,
    'max_depth': -1,
    'metric':'auc',
    'is_unbalance':'true',
    'lambda_l1' :10,
    'lambda_l2':10,
    'num_leaves': 7,
    'colsample_bytree': 0.7,
    'tree_learner': 'serial',
    'objective': 'cross_entropy',
    'verbosity': 1}


# In[ ]:


test_preds = []
folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=20_01_1998)
for fold,(trn_idx , val_idx) in enumerate(folds.split(patient_only_train[Cols],patient_only_train['target'])):
    print(f'********************* Fitting on Fold {fold+1} ... ******************')
    clf1=TrainSimpleLgbm(ParN1,patient_only_train,trn_idx,val_idx,"target",Cols, categoricals)
    clf2=TrainSimpleLgbm(ParN2,patient_only_train,trn_idx,val_idx,"target",Cols, categoricals)
    
    pred_oof1 = clf1.predict(patient_only_train.iloc[val_idx][Cols], num_iteration=clf1.best_iteration)
    pred_test1 = clf1.predict(patient_only_test[Cols], num_iteration=clf1.best_iteration)
    test_preds.append(pred_test1)
    pred_oof2 = clf2.predict(patient_only_train.iloc[val_idx][Cols], num_iteration=clf2.best_iteration)
    pred_test2 = clf1.predict(patient_only_test[Cols], num_iteration=clf2.best_iteration)
    test_preds.append(pred_test2)
    mean_pred_oof=0.5*pred_oof1+0.5*pred_oof2
    
    m1=mean_average_p(patient_only_train["target"].iloc[val_idx], pred_oof1)
    m2=mean_average_p(patient_only_train["target"].iloc[val_idx], pred_oof2)
    m3=mean_average_p(patient_only_train["target"].iloc[val_idx], mean_pred_oof)
    
    print(f' Mean Average M1 : {m1}  , M2 : {m2}   M3 : {m3}')
    
    
    pred_oof1=(pred_oof1>=0.5).astype(int)
    pred_oof2=(pred_oof2>=0.5).astype(int)
    mean_pred_oof=(mean_pred_oof>=0.5).astype(int)
    
    print("*************  CR Param 1 *************************")
    print(classification_report(patient_only_train["target"].iloc[val_idx],pred_oof1))
    
    print("*************  CR Param 2 *************************")
    print(classification_report(patient_only_train["target"].iloc[val_idx],pred_oof2))
    
    print("*************  CR Mean *************************")
    print(classification_report(patient_only_train["target"].iloc[val_idx],mean_pred_oof))
   


# Let's take a look at the feature importance. The most important feature seem to be, by far, the body part in which the melanoma is located.

# In[ ]:


feature_importance = pd.DataFrame({'Value':clf1.feature_importance(),'Feature':Cols})
plt.figure()
sns.barplot(x="Value", y="Feature", data=feature_importance.sort_values(by="Value", ascending=False))
plt.title('Features Importance')
plt.tight_layout()
plt.show()


# If we're interested in what part of the body it's actually the most dangerous to get a melanoma, let's make a brief analysis :

# In[ ]:


train.groupby(['anatom_site_general_challenge'])['target'].mean().sort_values(ascending=False)


# It seems that it is on the head or neck that you get the most malignant melanoma, with 3 times as much chances as when it is located on palms or soles !

# # Submission

# We had kept the `patient_id` column only to be able to perform the future merge of our results. Let's do it in two steps : first, adding our predictions to the `patient_only_test` database where we have `patient_id`, then merging it with the `test` database to match with the `image_name`, and finally using it to fill the `sample_submission` file.

# In[ ]:


patient_only_test['target'] = np.stack(test_preds, axis=1).mean(axis=1)


# In[ ]:


sample_submission = submission[['image_name']].merge(test[['image_name', 'patient_id']].merge(patient_only_test[['patient_id', 'target']], how='outer', on='patient_id'), how='left', on='image_name').drop(columns='patient_id', inplace=False).drop_duplicates(subset='image_name', inplace=False)


# In[ ]:


sample_submission.isna().describe()


# Great, the merging has performed well, we have no missing values !

# In[ ]:


sample_submission.to_csv('sample_submission.csv', index=False)

