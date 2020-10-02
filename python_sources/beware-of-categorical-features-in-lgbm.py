#!/usr/bin/env python
# coding: utf-8

# # How to understand feature importance of categorical features reported by LightGBM?
# LightGBM allows one to specify directly categorical features and handles those internally in a smart way, that might out-perform OHE. Originally, *I was puzzled about feature importance reported for such categorical features*. After  iterating in comments, and learning more about feature importance reported, it seems that:
# 
# - **the default implementation is not very useful**, as there are several types of importance and importance values do not behave according to intuitive expectation. See [this blog post](https://towardsdatascience.com/interpretable-machine-learning-with-xgboost-9ec80d148d27) for clear motivation and introduction into SHAP;
# - it is beneficial to use [SHAP package in python](https://github.com/slundberg/shap) to produce stable feature-importance evaluation.
# 
# It all started with abnormally high importance reported for `ORGANIZATION_TYPE` in [an earlier version of my modified fork](https://www.kaggle.com/mlisovyi/modular-good-fun-with-ligthgbm?scriptVersionId=3888846) of  [olivier's](https://www.kaggle.com/ogrellier) very popular [Good_fun_with_LigthGBM kernel](https://www.kaggle.com/ogrellier/good-fun-with-ligthgbm). After some investigation I realized that the problem was due to missing OHE of categorical features (beacuse categorical feature were stores as `categories` instead of `objects`). I fixed that, `ORGANIZATION_TYPE` got OHE-transformed and disappeared from tops of  important features. 
# 
# Then I started to looking into how to use internal handling of categorical features in LightGBM. It turns out that the **sklearn API of LightGBM actually has those enabled by default**, in a sense that by default it tries to guess which features are categorical, if you provided a `pd.DataFrame` as input (because it has `feature_name='auto', categorical_feature='auto'` as the defaults in the `lgb.LGBMModel.fit()` method). And it makes that guess assuming that all features of type `category` have to be treated with the internal categorical treatment (i.e. following [this procedure from the docs](https://github.com/Microsoft/LightGBM/blob/master/docs/Advanced-Topics.rst#categorical-feature-support)). It turns out that in such case LightGBM reports unexpectedly high importance  in some cases.
# 
# Below is a minimalistic example to reproduce this behaviour and an illustrastion of SHAP usage.

# In[ ]:


import numpy as np 
import pandas as pd 

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

import gc
gc.enable()

PATH = '../input/'


# ## Read in the basic 'application' data

# In[ ]:


application_train = pd.read_csv(PATH+'application_train.csv')

y = application_train['TARGET']
X = application_train.drop(['TARGET', 'SK_ID_CURR'], axis=1)

del application_train
gc.collect()


# Transform categorical features into the appropriate type that is expected by LightGBM.

# In[ ]:


for c in X.columns:
    col_type = X[c].dtype
    if col_type == 'object' or col_type.name == 'category':
        X[c] = X[c].astype('category')


# Printout types of features in the dataset

# In[ ]:


X.info()


# # Model fitting
# We will use LightGBM classifier (i.e. sklearn API)
# ### Split the full sample into train/test (80/20)

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=314, stratify=y)


# ### Use test subset for early stopping criterion 
# This allows us to avoid overtraining and we do not need to optimise the number of trees

# In[ ]:


fit_params={"early_stopping_rounds":10, 
            "eval_metric" : 'auc', 
            "eval_set" : [(X_test,y_test)],
            'eval_names': ['valid'],
            'verbose': 100,
            'feature_name': 'auto', # that's actually the default
            'categorical_feature': 'auto' # that's actually the default
           }


# ### Create a model. 
# Parameters are rough guesstimates. they are not supposed to be the best optimal choice.[](http://)

# In[ ]:


import lightgbm as lgb
#n_estimators is set to a "large value". The actual number of trees build will depend on early stopping and 1000 define only the absolute maximum
clf = lgb.LGBMClassifier(num_leaves= 15, max_depth=-1, 
                         random_state=314, 
                         silent=True, 
                         metric='None', 
                         n_jobs=4, 
                         n_estimators=1000,
                         colsample_bytree=0.9,
                         subsample=0.9,
                         learning_rate=0.1)


# ## Train the model
# We do training with the 0.8 subset of the dataset and 0.2 subset for early stopping. 

# In[ ]:


#force larger number of max trees and smaller learning rate
clf.fit(X_train, y_train, **fit_params)


# ### Plot feature importance

# In[ ]:


feat_imp = pd.Series(clf.feature_importances_, index=X.columns)
feat_imp.nlargest(30).plot(kind='barh', figsize=(8,10))


# Upsss. `ORGANIZATION_TYPE` pops up as the most *important*. But do not celerbate- if you train the same model on the same data with OHE for categorical features you will get the same ROC AUC (and a similar importance for the `EXT_SOURCE_x` features as on this plot), i.e. most likely just importance of `ORGANIZATION_TYPE` is reported wrong, unless i misunderstand something. Any feedback will be helpful for me to make the next step in LightGBM usage.

# In[ ]:


class LGBMClassifier_GainFE(lgb.LGBMClassifier):
    @property
    def feature_importances_(self):
        if self._n_features is None:
            raise LGBMNotFittedError('No feature_importances found. Need to call fit beforehand.')
        return self.booster_.feature_importance(importance_type='gain')
    


# In[ ]:


clf2 = LGBMClassifier_GainFE(num_leaves= 15, max_depth=-1, 
                         random_state=314, 
                         silent=True, 
                         metric='None', 
                         n_jobs=4, 
                         n_estimators=1000,
                         colsample_bytree=0.9,
                         subsample=0.9,
                         learning_rate=0.1)


# In[ ]:


clf2.fit(X_train, y_train, **fit_params)


# In[ ]:


feat_imp = pd.Series(clf2.feature_importances_, index=X.columns)
feat_imp.nlargest(30).plot(kind='barh', figsize=(8,10))


# ## Let's try SHAP
# following comment by [@cast42](https://www.kaggle.com/cast42) and his kernel [here]()

# In[ ]:


import shap
shap.initjs()


# In[ ]:


shap_values = shap.TreeExplainer(clf.booster_).shap_values(X_train)


# In[ ]:


shap.summary_plot(shap_values, X_train)


# This agrees more with our intuitive expectation of which features show show up on top in importance ranking. Note, that categoric features do not show colour highlighting, as higher/lower value is not defined.

# In[ ]:




