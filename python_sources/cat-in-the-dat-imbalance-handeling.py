#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install imblearn')


# In[ ]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split, KFold, cross_val_score, RepeatedKFold
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, ExtraTreesRegressor
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import BorderlineSMOTE, SMOTENC, SMOTE, ADASYN
from imblearn.under_sampling import ClusterCentroids
from category_encoders import  LeaveOneOutEncoder, BinaryEncoder, TargetEncoder
import time
import logging

sample_submission = pd.read_csv("../input/cat-in-the-dat-ii/sample_submission.csv")
test = pd.read_csv("../input/cat-in-the-dat-ii/test.csv")
train = pd.read_csv("../input/cat-in-the-dat-ii/train.csv")


# In[ ]:


def replace_nans(dataframe):
    for each in dataframe.columns:
        if each == 'id':
            continue
        if dataframe[each].dtype != 'object' or dataframe[each].dtype != 'datetime64':
            dataframe.loc[:, each] = dataframe.fillna(dataframe[each].mode()[0])
        else:
            dataframe.loc[:, each] = dataframe.fillna('UNKNOWN')
    
    return dataframe


# In[ ]:


def encoder(dataframe, columns, enc_type='bin'):

	if enc_type == 'bin':
		for col in columns:
			unique = dataframe[col].unique()
			dataframe.loc[:, col] = 			dataframe[col].apply(lambda x: 1 if x==unique[0] else (0 if x==unique[1] else None))
	if enc_type == 'ord':
		encoder = OrdinalEncoder(dtype=np.int16)
		for col in columns:
			dataframe.loc[:, col] = encoder.fit_transform(np.array(dataframe[col]).reshape(-1,1))


	return dataframe


# In[ ]:


def fitter(clf, X_train, X_test, y_train, y_test):
    print('training ', clf)
    y_train = np.array([[target] for target in y_train])
    y_test = np.array([[target] for target in y_test])
    clf.fit(X_train, y_train)
    try:
        print('score:', clf.score(clf, X_test, y_test))
    except Exception:
        print(clf.best_score_)
    
    return clf


# In[ ]:


def generate_samples(X, y, cat_features=None):
    
    #smote_nc = SMOTENC(categorical_features=cat_features, n_jobs=-1)
    smote = BorderlineSMOTE(sampling_strategy='all', n_jobs=-1)
    X, y = smote.fit_sample(X, y)
    return X, y


# In[ ]:


def main_2():
    data = train
    data = replace_nans(data)
    submission_data = replace_nans(test)
    nom_cols = ['nom_0', 'nom_1', 'nom_2']
    ord_cols = ['ord_3', 'ord_4', 'ord_5']
    bin_cols = ['bin_3', 'bin_4']
    
    ord_encoder = OrdinalEncoder()
    for enc in ord_cols+nom_cols:
        data[enc] = ord_encoder.fit_transform(np.array(data[enc]).reshape(-1,1))
        submission_data[enc] = ord_encoder.fit_transform(np.array(submission_data[enc]).reshape(-1,1))
    
    
    for enc in ['nom_3','nom_4']:
        enc1 = pd.get_dummies(data[enc], prefix=enc)
        data.drop(columns=enc, inplace=True)
        data = pd.concat([data, enc1], axis=1)
        
    for enc in ['nom_3','nom_4']:
        enc1 = pd.get_dummies(submission_data[enc], prefix=enc)
        submission_data.drop(columns=enc, inplace=True)
        submission_data = pd.concat([submission_data, enc1], axis=1)
           
    
    target = data['target']
    print('class 0:', len([x for x in target if x == 0]))
    print('class 1:', len([x for x in target if x == 1]))
    data = data.drop('target', axis=1)
    loo_enc = LeaveOneOutEncoder(cols=['nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9'], return_df=True)
    loo_enc.fit(data, target)
    data = loo_enc.transform(data)
    submission_data = loo_enc.transform(submission_data)
    
    data = encoder(data, ['ord_1', 'ord_2'], enc_type='ord')
    data = encoder(data, bin_cols, enc_type='bin')
    submission_data = encoder(submission_data, ['ord_1', 'ord_2'], enc_type='ord')
    submission_data = encoder(submission_data, bin_cols, enc_type='bin')
    time_features = ['day', 'month']
    
    for feature in time_features:
        data[feature+'_sin'] = np.sin((2*np.pi*data[feature])/max(data[feature]))
        data[feature+'_cos'] = np.cos((2*np.pi*data[feature])/max(data[feature]))

    data.drop(time_features, axis=1, inplace=True)
    
    for feature in time_features:
        submission_data[feature+'_sin'] = np.sin((2*np.pi*submission_data[feature])/max(submission_data[feature]))
        submission_data[feature+'_cos'] = np.cos((2*np.pi*submission_data[feature])/max(submission_data[feature]))

    submission_data.drop(time_features, axis=1, inplace=True)
    features = nom_cols+bin_cols+['nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9']
    features = [i for i, col in enumerate(data.columns) if col in features]
    
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2)
    
    X_train, y_train = generate_samples(X_train, y_train, features)
    
    print(X_train, X_train.shape)
    print(y_train, y_train.shape)
    
    clf_3 = GradientBoostingClassifier(n_estimators=500, verbose=1, learning_rate=0.05, max_depth=7)
    clf_3.fit(X_train, y_train)
    try:
        print(clf_3.score(X_test, y_test))
    except:
        pass
    
    #rkf = RepeatedKFold(n_splits=5, n_repeats=3)
    """
    for train_index, test_index in rkf.split(data):
        X_train, X_test = data.values[train_index], data.values[test_index]
        y_train, y_test = target.values[train_index], target.values[test_index]
        clf_3.fit(X_train, y_train)
        print(clf_3.score(X_test, y_test))
    """
    predictions = clf_3.predict_proba(submission_data.values)
    predictions = [x[1] for x in predictions]
    print(predictions)
    submission_data = pd.read_csv("../input/cat-in-the-dat-ii/test.csv")
    submission_data['target'] = predictions
    submission_data = pd.concat([submission_data['id'], submission_data['target']], axis=1)
    submission_data.to_csv('submission.csv', index=False)


# In[ ]:


main_2()


# In[ ]:




