#!/usr/bin/env python
# coding: utf-8

# * This is a starter script to get you started with pipelines
# * It is nowhere near optimized and doesn't reward the download and submit crowd so you will have to do some work for a reward!
# * The pipeline tries to classify into galaxy or intergalactic and then tries to predict the actual classes 

# In[ ]:


import gc
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import log_loss
from scipy.stats import skew, kurtosis

from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


def get_inputs(data, metadata):
    data['flux_ratio_sq'] = np.power(data['flux'] / data['flux_err'], 2.0)
    data['flux_by_flux_ratio_sq'] = data['flux'] * data['flux_ratio_sq']
    mjddelta = train[['object_id','detected','mjd']]
    mjddelta = mjddelta[mjddelta.detected==1].groupby('object_id').agg({'mjd': ['min', 'max']})
    mjddelta['delta'] = mjddelta[mjddelta.columns[1]]-mjddelta[mjddelta.columns[0]]
    mjddelta = mjddelta['delta'].reset_index(drop=False)
    metadata = metadata.merge(mjddelta,on='object_id')
    
    
    aggdata = data.groupby(['object_id','passband']).agg({'mjd': ['min', 'max', 'size'],
                                             'flux': ['min', 'max', 'mean', 'median', 'std','skew'],
                                             'flux_err': ['min', 'max', 'mean', 'median', 'std','skew'],
                                             'flux_by_flux_ratio_sq': ['sum'],    
                                             'flux_ratio_sq': ['sum'],                      
                                             'detected': ['mean','std']}).reset_index(drop=False)
    
    cols = ['_'.join(str(s).strip() for s in col if s) if len(col)==2 else col for col in aggdata.columns ]
    aggdata.columns = cols
    aggdata = aggdata.merge(metadata,on='object_id',how='left')
    aggdata.insert(1,'delta_passband', aggdata.mjd_max-aggdata.mjd_min)
    aggdata.drop(['mjd_min','mjd_max'],inplace=True,axis=1)
    aggdata['flux_diff'] = aggdata['flux_max'] - aggdata['flux_min']
    aggdata['flux_dif2'] = (aggdata['flux_max'] - aggdata['flux_min']) / aggdata['flux_mean']
    aggdata['flux_w_mean'] = aggdata['flux_by_flux_ratio_sq_sum'] / aggdata['flux_ratio_sq_sum']
    aggdata['flux_dif3'] = (aggdata['flux_max'] - aggdata['flux_min']) / aggdata['flux_w_mean']
    return aggdata

class GalaxyClassifier(BaseEstimator, TransformerMixin):
    def __init__(self, lgb_params=None, cv=5):
        self.lgb_params = lgb_params
        self.cv = cv
        self.clfs = []
    
    def fit(self, X, y=None):
        importances = pd.DataFrame()
        self.features = list(set(X.columns).difference(set(['object_id','hostgal_specz'])))
        target = (X.hostgal_specz==0).astype(int)
        folds = KFold(n_splits=self.cv, shuffle=True, random_state=1)
        
        oof_preds = np.zeros((len(X)))
        for fold_, (trn_, val_) in enumerate(folds.split(target)):
            trn_x, trn_y = X.iloc[trn_][self.features], target.iloc[trn_]
            val_x, val_y = X.iloc[val_][self.features], target.iloc[val_]

            clf = lgb.LGBMClassifier(**self.lgb_params)
            clf.fit(
                trn_x, trn_y,
                eval_set=[(trn_x, trn_y), (val_x, val_y)],
                eval_metric='binary_logloss',
                verbose=1000,
                early_stopping_rounds=50
            )
            imp_df = pd.DataFrame()
            imp_df['feature'] = self.features
            imp_df['gain'] = clf.feature_importances_
            imp_df['fold'] = fold_ + 1
            importances = pd.concat([importances, imp_df], sort=False)
            preds = clf.predict_proba(val_x, num_iteration=clf.best_iteration_)[:,1]
            oof_preds[val_] = preds
            print(log_loss(val_y, clf.predict_proba(val_x, num_iteration=clf.best_iteration_)))
            self.clfs.append(clf)
        mean_gain = importances[['gain', 'feature']].groupby('feature').mean()
        importances['mean_gain'] = importances['feature'].map(mean_gain['gain'])
        plt.figure(figsize=(8, 12))
        sns.barplot(x='gain', y='feature', data=importances.sort_values('mean_gain', ascending=False))
        plt.tight_layout()
        plt.show()
        return self
    
    def fit_transform(self, X, y):
        self.fit(X,y)
        target = (X.hostgal_specz==0).astype(int)
        oof_preds = np.zeros((len(X)))
                
        target = (X.hostgal_specz==0).astype(int)
        folds = KFold(n_splits=self.cv, shuffle=True, random_state=1)
        for fold_, (trn_, val_) in enumerate(folds.split(target)):
            trn_x, trn_y = X.iloc[trn_][self.features], target.iloc[trn_]
            val_x, val_y = X.iloc[val_][self.features], target.iloc[val_]
            clf = self.clfs[fold_]
            preds = clf.predict_proba(val_x, num_iteration=clf.best_iteration_)[:,1]
            oof_preds[val_] = preds
        
        X['hostgal_specz_pred'] = oof_preds
        mnx = X[['object_id','hostgal_specz_pred']].groupby('object_id').hostgal_specz_pred.mean().reset_index(drop=False)
        mnx.columns = ['object_id','hostgal_specz_pred']
        X.drop('hostgal_specz_pred',inplace=True,axis=1)
        X = X.merge(mnx,on='object_id',how='left')
        X.loc[X.hostgal_specz==0,'hostgal_specz_pred'] = 1.0
        return X
        
    def transform(self, X):
        target = (X.hostgal_specz==0).astype(int)
        oof_preds = np.zeros((len(X)))
        for clf in self.clfs:
            oof_preds += (clf.predict_proba(X[self.features], num_iteration=clf.best_iteration_)[:,1])/self.cv
        
        X['hostgal_specz_pred'] = oof_preds
        mnx = X[['object_id','hostgal_specz_pred']].groupby('object_id').hostgal_specz_pred.mean().reset_index(drop=False)
        mnx.columns = ['object_id','hostgal_specz_pred']
        X.drop('hostgal_specz_pred',inplace=True,axis=1)
        X = X.merge(mnx,on='object_id',how='left')
        X.loc[X.hostgal_specz==0,'hostgal_specz_pred'] = 1.0
        return X

class RealClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, lgb_params=None, cv=5):
        self.lgb_params = lgb_params
        self.cv = cv
        self.galaxyclfs = []
        self.intergalacticclfs = []
        self.standard_classes = [6, 15, 16, 42, 52, 53, 62, 64, 65, 67, 88, 90, 92, 95]
        self.standard_weight = {6: 1, 15: 2, 16: 1, 42: 1, 52: 1, 53: 1, 62: 1, 64: 2, 65: 1, 67: 1, 88: 1, 90: 1, 92: 1, 95: 1}
        
    def lgb_multi_weighted_logloss(self,y_true, y_preds):
        """
        @author olivier https://www.kaggle.com/ogrellier
        multi logloss for PLAsTiCC challenge
        """
        # class_weights taken from Giba's topic : https://www.kaggle.com/titericz
        # https://www.kaggle.com/c/PLAsTiCC-2018/discussion/67194
        # with Kyle Boone's post https://www.kaggle.com/kyleboone
        classes = list(np.unique(y_true))
        classes = [self.standard_classes[x] for x in classes]
        class_weight = {i:self.standard_weight[i] for i in classes if self.standard_weight[i] is not None}


        if len(np.unique(y_true)) > 14:
            classes.append(99)
            class_weight[99] = 2
        y_p = y_preds.reshape(y_true.shape[0], len(classes), order='F')

        # Trasform y_true in dummies
        y_ohe = pd.get_dummies(y_true)
        # Normalize rows and limit y_preds to 1e-15, 1-1e-15
        y_p = np.clip(a=y_p, a_min=1e-15, a_max=1 - 1e-15)
        # Transform to log
        y_p_log = np.log(y_p)
        # Get the log for ones, .values is used to drop the index of DataFrames
        # Exclude class 99 for now, since there is no class99 in the training set
        # we gave a special process for that class
        y_log_ones = np.sum(y_ohe.values * y_p_log, axis=0)
        # Get the number of positives for each class
        nb_pos = y_ohe.sum(axis=0).values.astype(float)
        # Weight average and divide by the number of positives
        class_arr = np.array([class_weight[k] for k in sorted(class_weight.keys())])
        y_w = y_log_ones * class_arr / nb_pos

        loss = - np.sum(y_w) / np.sum(class_arr)
        return 'wloss', loss, False
    
    def fitGalactic(self, X, y):
        importances = pd.DataFrame()
        folds = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=1)

        X = X.loc[X.hostgal_specz==0].copy()
        y = y.iloc[X.index].copy()
        
        oof_preds = np.zeros((len(X), np.unique(y).shape[0]))
        lgb_params = self.lgb_params 
        lgb_params.update({'num_class': len(y.unique())})
        for fold_, (trn_, val_) in enumerate(folds.split(X.index, y)):
            trn_x, trn_y = X.iloc[trn_][self.features], y.iloc[trn_]
            val_x, val_y = X.iloc[val_][self.features], y.iloc[val_]

            clf = lgb.LGBMClassifier(**lgb_params)
            clf.fit(
                trn_x, trn_y,
                eval_set=[(trn_x, trn_y), (val_x, val_y)],
                eval_metric=self.lgb_multi_weighted_logloss,
                verbose=1000,
                early_stopping_rounds=50
            )
            imp_df = pd.DataFrame()
            imp_df['feature'] = self.features
            imp_df['gain'] = clf.feature_importances_
            imp_df['fold'] = fold_ + 1
            importances = pd.concat([importances, imp_df], sort=False)
            self.galaxyclfs.append(clf)
        mean_gain = importances[['gain', 'feature']].groupby('feature').mean()
        importances['mean_gain'] = importances['feature'].map(mean_gain['gain'])
        plt.figure(figsize=(8, 12))
        sns.barplot(x='gain', y='feature', data=importances.sort_values('mean_gain', ascending=False))
        plt.tight_layout()
        plt.show()
        return self
    
    def fitInterGalactic(self, X, y):
        importances = pd.DataFrame()
        folds = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=1)
       
        X = X.loc[X.hostgal_specz!=0].copy()
        y = y.iloc[X.index].copy()
        
        
        
        oof_preds = np.zeros((len(X), np.unique(y).shape[0]))
        lgb_params = self.lgb_params 
        lgb_params.update({'num_class': len(y.unique())})
        for fold_, (trn_, val_) in enumerate(folds.split(X.index, y)):
            trn_x, trn_y = X.iloc[trn_][self.features], y.iloc[trn_]
            val_x, val_y = X.iloc[val_][self.features], y.iloc[val_]

            clf = lgb.LGBMClassifier(**lgb_params)
            clf.fit(
                trn_x, trn_y,
                eval_set=[(trn_x, trn_y), (val_x, val_y)],
                eval_metric=self.lgb_multi_weighted_logloss,
                verbose=1000,
                early_stopping_rounds=50
            )
            imp_df = pd.DataFrame()
            imp_df['feature'] = self.features
            imp_df['gain'] = clf.feature_importances_
            imp_df['fold'] = fold_ + 1
            importances = pd.concat([importances, imp_df], sort=False)
            self.intergalacticclfs.append(clf)
        mean_gain = importances[['gain', 'feature']].groupby('feature').mean()
        importances['mean_gain'] = importances['feature'].map(mean_gain['gain'])
        plt.figure(figsize=(8, 12))
        sns.barplot(x='gain', y='feature', data=importances.sort_values('mean_gain', ascending=False))
        plt.tight_layout()
        plt.show()
        return self

      
    def fit(self, X, y):
        X = X.copy()
        self.features = list(set(X.columns).difference(set(['object_id','target','hostgal_specz'])))
        self.fitGalactic(X, y)   
        self.fitInterGalactic(X, y)   
        return self
    
    
    def fit_transform(self, X, y):
        self.fit(X,y)
        oof_preds = np.zeros((len(X),len(self.galaxyclfs[0].classes_)))
        
        folds = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=1)
        for fold_, (trn_, val_) in enumerate(folds.split(X.index, y)):
            trn_x, trn_y = X.iloc[trn_][self.features], y.iloc[trn_]
            val_x, val_y = X.iloc[val_][self.features], y.iloc[val_]
            clf = self.galaxyclfs[fold_]
            preds = clf.predict_proba(val_x, num_iteration=clf.best_iteration_)
            oof_preds[val_] = preds
        
        oof_preds1 = np.zeros((len(X),len(self.intergalacticclfs[0].classes_)))
        folds = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=1)
        for fold_, (trn_, val_) in enumerate(folds.split(X.index, y)):
            trn_x, trn_y = X.iloc[trn_][self.features], y.iloc[trn_]
            val_x, val_y = X.iloc[val_][self.features], y.iloc[val_]
            clf = self.intergalacticclfs[fold_]
            preds = clf.predict_proba(val_x, num_iteration=clf.best_iteration_)
            oof_preds1[val_] = preds
        
        galaxy = pd.DataFrame(oof_preds)
        galaxy.columns = ['class_'+str(i) for i in self.galaxyclfs[0].classes_]
        intergalactic = pd.DataFrame(oof_preds1)
        intergalactic.columns = ['class_'+str(i) for i in self.intergalacticclfs[0].classes_]
        alltargets = list(set(list(galaxy.columns)+list(intergalactic.columns)))
        x = pd.DataFrame()
        x['object_id'] = X.object_id.ravel()
        for c in alltargets:
            if(c in galaxy):
                x[c] = galaxy[c].values*(X['hostgal_specz_pred'])
            if(c in intergalactic):
                x[c] = intergalactic[c].values*(1-X['hostgal_specz_pred'].values)
        x['class_99'] = (1- x[x.columns[1:]].sum(axis=1)).clip(0,1)
        x = x.groupby(['object_id']).mean().reset_index(drop=False)
        return x.fillna(0)
    
    
    def transform(self, X):
        X = X.copy()
        oof_preds = np.zeros((len(X),len(self.galaxyclfs[0].classes_)))
        
        for clf in self.galaxyclfs:
            oof_preds += (clf.predict_proba(X[self.features], num_iteration=clf.best_iteration_))/self.cv
        
        oof_preds1 = np.zeros((len(X),len(self.intergalacticclfs[0].classes_)))
        
        for clf in self.intergalacticclfs:
            oof_preds1 += (clf.predict_proba(X[self.features], num_iteration=clf.best_iteration_))/self.cv
               
        galaxy = pd.DataFrame(oof_preds)
        galaxy.columns = ['class_'+str(i) for i in self.galaxyclfs[0].classes_]
        intergalactic = pd.DataFrame(oof_preds1)
        intergalactic.columns = ['class_'+str(i) for i in self.intergalacticclfs[0].classes_]
        alltargets = list(set(list(galaxy.columns)+list(intergalactic.columns)))
        x = pd.DataFrame()
        x['object_id'] = X.object_id.ravel()
        for c in alltargets:
            if(c in galaxy):
                x[c] = galaxy[c].values*(X['hostgal_specz_pred'])
            if(c in intergalactic):
                x[c] = intergalactic[c].values*(1-X['hostgal_specz_pred'].values)
        x['class_99'] = (1- x[x.columns[1:]].sum(axis=1)).clip(0,1)
        x = x.groupby(['object_id']).mean().reset_index(drop=False)
        return x.fillna(0)
    
def multi_weighted_logloss(y_true, y_preds):
    """
    @author olivier https://www.kaggle.com/ogrellier
    multi logloss for PLAsTiCC challenge
    """
    # class_weights taken from Giba's topic : https://www.kaggle.com/titericz
    # https://www.kaggle.com/c/PLAsTiCC-2018/discussion/67194
    # with Kyle Boone's post https://www.kaggle.com/kyleboone
    classes = [6, 15, 16, 42, 52, 53, 62, 64, 65, 67, 88, 90, 92, 95]
    class_weight = {6: 1, 15: 2, 16: 1, 42: 1, 52: 1, 53: 1, 62: 1, 64: 2, 65: 1, 67: 1, 88: 1, 90: 1, 92: 1, 95: 1}
    if len(np.unique(y_true)) > 14:
        classes.append(99)
        class_weight[99] = 2
    y_p = y_preds
    # Trasform y_true in dummies
    y_ohe = pd.get_dummies(y_true)
    # Normalize rows and limit y_preds to 1e-15, 1-1e-15
    y_p = np.clip(a=y_p, a_min=1e-15, a_max=1 - 1e-15)
    # Transform to log
    y_p_log = np.log(y_p)
    # Get the log for ones, .values is used to drop the index of DataFrames
    # Exclude class 99 for now, since there is no class99 in the training set
    # we gave a special process for that class
    y_log_ones = np.sum(y_ohe.values * y_p_log, axis=0)
    # Get the number of positives for each class
    nb_pos = y_ohe.sum(axis=0).values.astype(float)
    # Weight average and divide by the number of positives
    class_arr = np.array([class_weight[k] for k in sorted(class_weight.keys())])
    y_w = y_log_ones * class_arr / nb_pos

    loss = - np.sum(y_w) / np.sum(class_arr)
    return loss


# In[ ]:


train = pd.read_csv('../input/training_set.csv')
meta_train = pd.read_csv('../input/training_set_metadata.csv')
traindata = get_inputs(train,meta_train)


# In[ ]:


traindata.columns


# In[ ]:


cols = list(set(traindata.columns).difference(set(['target'])))
print(cols)
X = traindata[cols].copy()
y = traindata.target


# In[ ]:


lgb_params = {
            'objective': 'binary',
            'boosting': 'gbdt',
            'learning_rate': 0.1 ,
            'silent': 1,
            'verbose': 0,
            'subsample': .9,
            'colsample_bytree': .7,
            'metric' : 'binary_logloss',
            'max_depth': 3,
            'min_child_weight':10
        }

al_lgb_params = {
            'objective': 'multiclass',
            'metric': 'multi_logloss',
            'learning_rate': 0.1,
            'subsample': .9,
            'colsample_bytree': .7,
            'reg_lambda': 1.0,
            'min_split_gain': 0.01,
            'n_estimators': 1000,
            'silent': 1,
            'verbose': 0,
            'max_depth': 3,
            'min_child_weight':10}


pipe = Pipeline([('gl', GalaxyClassifier(lgb_params=lgb_params, cv=5)),
                 ('al', RealClassifier(lgb_params=al_lgb_params, cv=5))])
predictions = pipe.fit_transform(X,y)


# In[ ]:


predictions.head()


# In[ ]:


cols = pd.get_dummies(meta_train.target).columns.values
cols = ['class_'+str(c) for c in cols]


# In[ ]:


log_loss(pd.get_dummies(meta_train.target), predictions[cols])


# In[ ]:


multi_weighted_logloss(meta_train.target,predictions[cols])


# In[ ]:


model = TSNE(n_components=2, perplexity=30,random_state=0)
tsnedata = model.fit_transform(predictions[predictions.columns[1:]])
cm = plt.cm.get_cmap('RdYlBu')
fig, axes = plt.subplots(1, 1, figsize=(15, 15))
sc = axes.scatter(tsnedata[:,0], tsnedata[:,1], alpha=1, c=(meta_train.target.values), cmap=cm, s=1)
cbar = fig.colorbar(sc, ax=axes)
cbar.set_label('Target')
_ = axes.set_title("Clustering colored by target")


# In[ ]:


# meta_test = pd.read_csv('../input/test_set_metadata.csv')
# print(meta_test.shape)
# chunks = 5000000
# remain_df = None
# test_all_predictions = None
# for i_c, df in enumerate(pd.read_csv('../input/test_set.csv', chunksize=chunks, iterator=True)):
#     print(i_c,df.shape)
#     unique_ids = np.unique(df['object_id'])
#     new_remain_df = df.loc[df['object_id'] == unique_ids[-1]].copy()
#     if remain_df is None:
#         df = df.loc[df['object_id'].isin(unique_ids[:-1])].copy()
#     else:
#         df = pd.concat([remain_df, df.loc[df['object_id'].isin(unique_ids[:-1])]], axis=0)
#     # Create remaining samples df
#     remain_df = new_remain_df

#     df = get_inputs(df, meta_test)
#     preds_df = pipe.transform(df)
#     if i_c == 0:
#         preds_df.to_csv('predictions.csv', header=True, index=False, float_format='%.6f')
#     else:
#         preds_df.to_csv('predictions.csv', header=False, mode='a', index=False, float_format='%.6f')

#     del preds_df
#     gc.collect()

