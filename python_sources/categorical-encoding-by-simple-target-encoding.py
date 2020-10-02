#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


get_ipython().system('pip install git+https://gitlab.com/nyker510/vivid')


# In[ ]:


train_df = pd.read_csv('/kaggle/input/cat-in-the-dat-ii/train.csv')
test_df = pd.read_csv('/kaggle/input/cat-in-the-dat-ii/test.csv')
                       


# In[ ]:


y = train_df['target'].values


# In[ ]:


del train_df['target']


# In[ ]:


train_df.head()


# In[ ]:


for c in train_df.columns:
    vc = train_df[c].value_counts()
#     print(vc)

    if c == 'id':
        continue

    test_set = set(test_df[c].unique())
    train_set = set(train_df[c].unique())
    print(train_set - test_set)
    
    n_intersection = len(train_set & test_set)
    print(f'{c} intersection: {n_intersection / len(train_set)}')


# In[ ]:


train_df.columns


# In[ ]:


train_df['ord_0'].unique()

test_df['ord_0'].unique()


# In[ ]:


set(train_df[c].unique()) -


# In[ ]:


from vivid.featureset.encodings import OneHotEncodingAtom


# In[ ]:


from vivid.utils import get_logger, timer


# In[ ]:


logger = get_logger(__name__)


# In[ ]:


from sklearn.model_selection import GroupKFold, StratifiedKFold

class TargetEncodingTransformer:
    def __init__(self, cv=10):
        self.cv = cv
        self.mapping_ = None

    def fit(self, X, y):
        fold = StratifiedKFold(self.cv)
        if isinstance(X, pd.Series):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values

        oof = np.zeros_like(y, dtype=np.float)
#         logger.info(f'Cardinarity: {len(np.unique(X))}')
        logger.info(X.shape)
        s = pd.Series(X)
        self.mapping_ = pd.DataFrame(y).groupby(X).mean()[0].to_dict()
#         print(self.mapping_)

        for idx1, idx2 in fold.split(X, y):
            vc = pd.DataFrame(y[idx1]).groupby(X[idx1]).mean()[0].to_dict()
            val = s[idx2].map(vc).fillna(np.mean(y[idx1]))
            oof[idx2] = val

        return oof

    def transform(self, X):
        retval = pd.Series(X).map(self.mapping_)
        return retval


# In[ ]:


from vivid.featureset import AbstractAtom


# In[ ]:


class TargetEncodingAtom(AbstractAtom):
    use_columns = ['nom_0', 'nom_1',
       'nom_2', 'nom_3', 'nom_4', 'nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9',
       'ord_0', 'ord_1', 'ord_2', 'ord_3', 'ord_4', 'ord_5', 'day', 'month']

    def __init__(self):
        self.mapping = {}

    def call(self, input_df: pd.DataFrame, y):
        output_df = pd.DataFrame()
        for col in self.use_columns:
            name = f'{col}'
            encoder = self.mapping.get(name, None)

            if encoder is not None:
                x = encoder.transform(input_df[col].values)
            else:
                encoder = TargetEncodingTransformer()
                x = encoder.fit(input_df[col], y)
                self.mapping[name] = encoder
            output_df[name] = x

        return output_df.add_prefix('TE_')

    def fit(self, input_df: pd.DataFrame, y) -> pd.DataFrame:
        self.mapping = {}
        self.call(input_df, y)
        return self

    def transform(self, input_df):
        return self.call(input_df, y=None)


# In[ ]:


TargetEncodingAtom().generate(train_df, y)


# In[ ]:


from collections import OrderedDict

class CatOneHotEoncodingAtom(OneHotEncodingAtom):
    use_columns = ['bin_0', 'bin_1', 'bin_2', 'bin_3']
    
    def fit(self, input_df: pd.DataFrame, y=None):
        self.mapping_ = OrderedDict()
        for c in self.use_columns:
            cat = input_df[c].dropna().unique()
            self.mapping_[c] = cat
        return self


# In[ ]:


CatOneHotEoncodingAtom().generate(train_df, y)


# In[ ]:


from vivid.out_of_fold.boosting import LGBMClassifierOutOfFold, XGBoostClassifierOutOfFold
from vivid.out_of_fold.boosting.block import create_boosting_seed_blocks
from vivid.out_of_fold.linear import LogisticOutOfFold
from vivid.out_of_fold.ensumble import RFClassifierFeatureOutOfFold
from vivid.out_of_fold.base import BaseOutOfFoldFeature
from vivid.core import MergeFeature, EnsembleFeature
from vivid.out_of_fold.base import BaseOutOfFoldFeature

from sklearn.ensemble import ExtraTreesClassifier
from rgf.sklearn import RGFClassifier
from sklearn.linear_model import LogisticRegression


# In[ ]:


class KaggleKernelMixin:
    pred_ = None
    def save_best_models(self, best_models):
        pass
    
    def predict(self, input_df):
        if self.pred_:
            logger.info(f'{self}: load from cache')
            return self.pred_
        self.pred_ = super().predict(input_df)
        return self.pred_
    
class FillnaMixin:
    def call(self, df_source, y=None, test=False):
        if not test:
            self.mappings_ = df_source.mean()
        df_source = df_source.fillna(self.mappings_)
        return super().call(df_source, y, test)

class CustomLGBM(KaggleKernelMixin, LGBMClassifierOutOfFold):
    initial_params = {
        'n_estimators': 10000,
        'objective': 'binary',
        'feature_fraction': .9,
        'learning_rate': .05,
        'max_depth': 5,
        'num_leaves': 17
    }
    
class LogisticOptuna(FillnaMixin, KaggleKernelMixin, LogisticOutOfFold):
    initial_params = {
        'input_scaling': 'standard'
    }

class XGB(KaggleKernelMixin, XGBoostClassifierOutOfFold):
    pass

class SimpleLGBM(KaggleKernelMixin, LGBMClassifierOutOfFold):
    initial_params = {
        'n_estimators': 10000,
        'learning_rate': .05,
        'reg_lambda': 1.,
        'reg_alpha': 1.,
        'feature_fraction': .7,
        'max_depth': 3,
    }
    
class RF(FillnaMixin, KaggleKernelMixin, RFClassifierFeatureOutOfFold):
    initial_params = {'n_estimators': 125, 'max_features': 0.2, 'max_depth': 25, 'min_samples_leaf': 4, 'n_jobs': -1}


# In[ ]:


from vivid.featureset.molecules import create_molecule, MoleculeFeature


# In[ ]:


class ExtraTree(FillnaMixin, KaggleKernelMixin, BaseOutOfFoldFeature):
    model_class = ExtraTreesClassifier
    initial_params = {'n_estimators': 100, 'max_features': 0.5, 'max_depth': 18, 'min_samples_leaf': 4, 'n_jobs': -1}
    
class RGF(FillnaMixin, KaggleKernelMixin, BaseOutOfFoldFeature):
    model_class = RGFClassifier
    initial_params = {'algorithm': 'RGF_Sib', 'loss': 'Log'}
    
class Logistic(FillnaMixin, KaggleKernelMixin, BaseOutOfFoldFeature):
    model_class = LogisticRegression
    init_params = { 'input_scaling': 'standard' }


# In[ ]:


basic_molecule = create_molecule([
    CatOneHotEoncodingAtom(),
    TargetEncodingAtom()
], name='basic')


# In[ ]:


entry_point = MoleculeFeature(basic_molecule, root_dir='/kaggle/working/')


# In[ ]:


single_models = [
    Logistic(parent=entry_point, name='logistic'),
    CustomLGBM(parent=entry_point, name='lgbm'),
    XGB(parent=entry_point, name='xgb'),
    RF(parent=entry_point, name='rf'),
]


# In[ ]:


merge = MergeFeature(single_models[:], name='merged', root_dir=entry_point.root_dir)


# In[ ]:


stacking_models = [
    Logistic(parent=merge, name='stack_logistic'),
    CustomLGBM(parent=merge, name='stack_lgbm'),
    EnsembleFeature(single_models[:], name='ensumble', root_dir=entry_point.root_dir)
]


# In[ ]:


for m in stacking_models:
    m.fit(train_df, y)


# In[ ]:


final_model = stacking_models[-1]


# In[ ]:


pred = final_model.predict(test_df).values[:, 0]


# In[ ]:


sub_df = pd.read_csv('/kaggle/input/cat-in-the-dat-ii/sample_submission.csv')


# In[ ]:


sub_df['target'] = pred


# In[ ]:


sub_df.to_csv('/kaggle/working/submission.csv', index=False)

