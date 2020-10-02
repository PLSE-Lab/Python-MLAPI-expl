#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import gc
import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE, RFECV, VarianceThreshold
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score


# In[ ]:


class ModelBase(object):
    def get_model(self):
        pass


# In[ ]:


class SVMModel(ModelBase):
    _model = None
    def get_model(self):
        self._model = SVC(
            probability=True,
            kernel='poly',
            degree=4,
            gamma='auto')

    def fit(self, x, y):
        self._model.fit(x, y)

    def predict(self, test):
        return np.array(self._model.predict_proba(test)[:, 1])

    def release(self):
        del self._model


# In[ ]:


class FoldedTrainer(object):
    # Model related
    _model = None
    _model_kwargs = None
    _train_kwargs = None
    _num_folds = None
    
    # Scores and data
    _oof = None
    _preds = None
    _aucs = None
    _train_data = None
    _test_data = None
    _feature_cols = None

    def __init__(self, model, model_kwargs=dict(), train_kwargs=dict(), num_folds=7):
        self._model = model
        self._model_kwargs = model_kwargs
        self._train_kwargs = train_kwargs
        self._num_folds = num_folds

    def train_single_case(self, case_index):
        train = self._train_data[self._train_data['wheezy-copper-turtle-magic'] == case_index]
        test = self._test_data[self._test_data['wheezy-copper-turtle-magic'] == case_index]
        idx_train = train.index
        idx_test = test.index
        train.reset_index(drop=True,inplace=True)
        test.reset_index(drop=True,inplace=True)
        oof = np.zeros(len(train))
        pred = np.zeros(len(test))
        all_features = pd.concat([train[self._feature_cols], test[self._feature_cols]])
        sel = VarianceThreshold(threshold=1.5).fit(all_features)
        train_sel = sel.transform(train[self._feature_cols])
        test_sel = sel.transform(test[self._feature_cols])
    
        skf = StratifiedKFold(n_splits=self._num_folds, random_state=42)
        for train_index, valid_index in skf.split(train_sel, train['target']):
            self._model.get_model(**self._model_kwargs)
            self._model.fit(
                x=train_sel[train_index, :],
                y=train.loc[train_index]["target"],
                **self._train_kwargs
            )
            pred_oof = self._model.predict(train_sel[valid_index, :])
            self._oof[idx_train[valid_index]] = pred_oof
            oof[valid_index] = pred_oof
            pred_test = self._model.predict(test_sel) / self._num_folds
            self._preds[idx_test] += pred_test
            pred += pred_test
            self._model.release()
    
        auc = roc_auc_score(train['target'], oof)
        pred_df = pd.DataFrame({
            "id": test["id"].tolist(),
            "target": list(pred)
        })
        return auc, pred_df
    
    def load_data(self, train_df, test_df):
        self._train_data = train_df
        self._test_data = test_df
        return None
    
    def train(self, train_df, test_df):
        self._train_data = train_df
        self._test_data = test_df
        self._feature_cols = [x for x in train_df.columns if x not in ["id", "target", "wheezy-copper-turtle-magic"]]
        self._aucs = []
        self._oof = np.zeros(len(train_df))
        self._preds = np.zeros(len(test_df))
        tmp_preds = []
        case_pool = sorted(list(set(train_df['wheezy-copper-turtle-magic'])))
        for case_index in case_pool:
            auc, preds_df = self.train_single_case(case_index=case_index)
            print(case_index, auc)
            self._aucs.append(auc)
            tmp_preds.append(preds_df)
        print(roc_auc_score(self._train_data["target"], self._oof))
        return tmp_preds

    def get_results(self):
        return self._aucs, self._preds


# In[ ]:


model = SVMModel()
trainer = FoldedTrainer(model=model, num_folds=4, train_kwargs=dict(), model_kwargs=dict())


# In[ ]:


train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')


# In[ ]:


tmp_preds = trainer.train(train_df, test_df)


# In[ ]:


aucs, preds = trainer.get_results()
preds2 = pd.concat(tmp_preds)


# In[ ]:


sub = pd.read_csv('../input/sample_submission.csv')
sub['target'] = preds
sub.to_csv('submission1.csv',index=False)
del sub


# In[ ]:


sub = pd.read_csv('../input/sample_submission.csv')
sub = sub[["id"]]
sub = sub.merge(preds2, on="id", how="left")
sub.to_csv('submission2.csv',index=False)
del sub


# In[ ]:




