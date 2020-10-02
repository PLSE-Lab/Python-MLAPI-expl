#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tqdm import tqdm_notebook as tqdm
import warnings
warnings.filterwarnings("ignore")
from sklearn.feature_selection import VarianceThreshold
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
from sklearn.mixture import GaussianMixture
from sklearn.covariance import GraphicalLasso
import os
print(os.listdir("../input"))


# In[ ]:


train_df = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# In[ ]:


class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models, weights = None):
        self.models = models
        self.weights = weights
        assert len(self.models) == len(self.weights),  ('Len models != len weights')
        assert abs(np.sum(self.weights) - 1) < 0.01,  ('weight sum != 1')
    # we define clones of the original models to fit the data in
    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]
        
        # Train cloned base models
        for model in self.models_:
                model.fit(X, y)
        return self
    
    #Now we do the predictions for cloned models and average them
    def predict(self, X):
        predictions = np.column_stack([
            model.predict_proba(X)[:,1] for model in self.models_
        ])
        if self.weights:
            return np.sum(predictions * self.weights, axis=1)
        else:
            return np.mean(predictions, axis=1)  


# In[ ]:


def get_predicts_with_validate(model, X, Y, n_folds = 3, X_test=[]):
    kf = StratifiedKFold(n_splits = n_folds, shuffle = True, random_state = 42)
    scores = list()
    #kf.get_n_splits(X,Y)
    out_of_fold_predictions = np.zeros((X.shape[0]))
    predicts_list = list()
    for train_index, test_index in kf.split(X,Y):
        X_train, X_valid = X.iloc[train_index], X.iloc[test_index]
        Y_train, Y_valid = Y.iloc[train_index], Y.iloc[test_index]
        model.fit(X_train,Y_train)
        predicts = model.predict(X_valid)
        out_of_fold_predictions[test_index] = predicts
        scores.append(roc_auc_score(Y_valid, predicts))
        if len(X_test) >0:
            test_predicts = model.predict(X_test)
            predicts_list.append(test_predicts)
    return out_of_fold_predictions, scores, np.mean(predicts_list, axis=0)


# In[ ]:


# Func for get mean and preccision init per cluster (2 cluster per class)



def get_mean_cov(x,y):
    model = GraphicalLasso()
    ones = (y==1).astype(bool)
    x2 = x[ones]
    try:
        #Some times fall down, because bad clustering, and we give second chance for clustering
        kmeans_ = KMeans(3)
        clusters = kmeans_.fit_predict(x2)
        p1_list = list()
        m1_list = list()
        for i in range(3):
            model.fit(x2[np.where(clusters == i)[0]])
            p1_list.append(model.precision_)
            m1_list.append(model.location_)

        onesb = (y==0).astype(bool)
        x2b = x[onesb]
        kmeans_ = KMeans(3)
        clusters = kmeans_.fit_predict(x2b)
        for i in range(3):
            model.fit(x2b[np.where(clusters == i)[0]])
            p1_list.append(model.precision_)
            m1_list.append(model.location_)
        n_clusters = 3
    except:
        print('!!!')
        kmeans_ = KMeans(2)
        clusters = kmeans_.fit_predict(x2)
        p1_list = list()
        m1_list = list()
        for i in range(2):
            model.fit(x2[np.where(clusters == i)[0]])
            p1_list.append(model.precision_)
            m1_list.append(model.location_)

        onesb = (y==0).astype(bool)
        x2b = x[onesb]
        kmeans_ = KMeans(2)
        clusters = kmeans_.fit_predict(x2b)
        for i in range(2):
            model.fit(x2b[np.where(clusters == i)[0]])
            p1_list.append(model.precision_)
            m1_list.append(model.location_)
        n_clusters = 2
    ms = np.stack(m1_list)
    ps = np.stack(p1_list)
    return ms,ps, n_clusters


# In[ ]:


# QDA for drop flip and pseudo labeling
QDA = QuadraticDiscriminantAnalysis(reg_param=0.5)


# In[ ]:


# QDA for drop flips
cols = [c for c in train_df.columns if c not in ['id', 'target']]
cols.remove('wheezy-copper-turtle-magic')
preds = np.zeros(len(train_df))
# BUILD 512 SEPARATE MODELS
for i in tqdm(range(512)):
    # ONLY TRAIN WITH DATA WHERE WHEEZY EQUALS I
    train2 = train_df[train_df['wheezy-copper-turtle-magic']==i]
    idx1 = train2.index;
    train2.reset_index(drop=True,inplace=True)
    # FEATURE SELECTION (USE APPROX 40 OF 255 FEATURES)
    sel = VarianceThreshold(threshold=1.5).fit(train2[cols])
    train3 = sel.transform(train2[cols])
    models = [QDA]
    averaged_models = AveragingModels(models=models, weights=[1])
    out_of_fold_predictions, scores,_ = get_predicts_with_validate(averaged_models, pd.DataFrame(train3), train2['target'], 11)
    preds[idx1] = out_of_fold_predictions
# PRINT CV AUC
auc = roc_auc_score(train_df['target'], preds)
print('QDA scores CV =',round(auc,5))


# In[ ]:


# Drop flips
train_df['preds'] = preds
print(len(train_df[((train_df['target']==0) & (train_df['preds']>=0.95)) | ((train_df['target']==1) & (train_df['preds']<=0.05))]))
train_df = train_df[((train_df['target']==0) & (train_df['preds']<=0.95)) | ((train_df['target']==1) & (train_df['preds']>=0.05))]
train_df.drop(['preds'], axis=1, inplace=True)
train_df.reset_index(inplace=True,drop=True)


# In[ ]:


# QDA for Pseudo labeling
cols = [c for c in train_df.columns if c not in ['id', 'target']]
cols.remove('wheezy-copper-turtle-magic')
preds = np.zeros(len(train_df))
test_preds = np.zeros(len(test))

preds_gm = np.zeros(len(train_df))
test_preds_gm = np.zeros(len(test))
# BUILD 512 SEPARATE MODELS
for i in tqdm(range(512)):
    # ONLY TRAIN WITH DATA WHERE WHEEZY EQUALS I
    train2 = train_df[train_df['wheezy-copper-turtle-magic']==i]
    test2 = test[test['wheezy-copper-turtle-magic']==i]
    idx1 = train2.index; idx2 = test2.index
    train2.reset_index(drop=True,inplace=True)
    # FEATURE SELECTION (USE APPROX 40 OF 255 FEATURES)
    sel = VarianceThreshold(threshold=1.5).fit(train2[cols])
    train3 = sel.transform(train2[cols])
    test3 = sel.transform(test2[cols])
    models = [QDA]
    averaged_models = AveragingModels(models=models, weights=[1])
    out_of_fold_predictions, scores, test_predicts = get_predicts_with_validate(averaged_models, pd.DataFrame(train3), train2['target'], 11, test3)
    preds[idx1] = out_of_fold_predictions
    test_preds[idx2] = test_predicts
    
preds = preds 
test_preds = test_preds
# PRINT CV AUC
auc = roc_auc_score(train_df['target'], preds)
print('QDA scores CV =',round(auc,5))


# In[ ]:


# get len of feature pseudo labeling data
test['target'] = test_preds
print(len(test[(test['target']<=0.01) | (test['target']>=0.99)]))


# In[ ]:


preds = np.zeros(len(train_df))
test_preds_ = np.zeros(len(test))
preds_gm = np.zeros(len(train_df))
test_preds_gm = np.zeros(len(test))
# BUILD 512 SEPARATE MODELS
for i in tqdm(range(512)):
    # ONLY TRAIN WITH DATA WHERE WHEEZY EQUALS I
    train2 = train_df[train_df['wheezy-copper-turtle-magic']==i]
    train2p = train2.copy(); idx1 = train2.index 
    test2 = test[test['wheezy-copper-turtle-magic']==i]
    idx1 = train2.index; idx2 = test2.index
    #train2.reset_index(drop=True,inplace=True)
    max_predicts_count = len(train2p)
     # ADD PSEUDO LABELED DATA
    test2p = test2[ (test2['target']<=0.01) | (test2['target']>=0.99) ].copy()
    test2p.loc[ test2p['target']>=0.5, 'target' ] = 1
    test2p.loc[ test2p['target']<0.5, 'target' ] = 0 
    train2p = pd.concat([train2p,test2p],axis=0)
    train2p.reset_index(drop=True,inplace=True)
    train2.reset_index(drop=True,inplace=True)
    
    # FEATURE SELECTION (USE APPROX 40 OF 255 FEATURES)
    sel = VarianceThreshold(threshold=1.5).fit(train2p[cols])     
    train3p = sel.transform(train2p[cols])
    train3 = sel.transform(train2[cols])
    test3 = sel.transform(test2[cols])
    
    # Get means and precisions for Gaussian Mixture (4 - components [ two for zeroes class and two for ones class])
    ms, ps, n_clusters = get_mean_cov(train3p,train2p['target'].values)
    gm = GaussianMixture(n_components=n_clusters * 2, init_params='random', covariance_type='full', tol=0.001,reg_covar=0.001, max_iter=100, n_init=1,means_init=ms, precisions_init=ps)
    gm.fit(np.concatenate([train3,test3],axis = 0))
    predicts = gm.predict_proba(train3)
    preds_gm[idx1] = np.sum(predicts[:,:n_clusters], axis=1)
    predicts = gm.predict_proba(test3)
    test_preds_[idx2] = np.sum(predicts[:,:n_clusters], axis=1)
auc = roc_auc_score(train_df['target'], preds_gm)
print('QDA scores CV =',round(auc,5))


# In[ ]:


sub = pd.read_csv('../input/sample_submission.csv')
sub['target'] = test_preds_
sub.to_csv('submission.csv',index=False)

