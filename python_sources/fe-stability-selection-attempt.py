#!/usr/bin/env python
# coding: utf-8

# Attempt to select features for classification using "Stability Selection" algorythm introduced in https://stat.ethz.ch/~nicolai/stability.pdf.
# 
# The main idea of algorythm is iterative fitting of multiple models using bootstraped subsamples; `stable`(significant) features should occur in models more often than `unstable`(non significant).
# 
# Implementation of `StabilitySelection` is now deprecated in *scikit-learn* package and moved into *scikit-learn-contrib* project https://github.com/scikit-learn-contrib/stability-selection/tree/master/stability_selection.
# 
# Stability selection used here with default settings as 1-st step of feature selection pipeline and shrinks number of features from 300 to 85 (~ 4 times). As 2-nd step the `SequentialFeatureSelector` used here.

# In[ ]:


import pandas as pd
import numpy as np


# In[ ]:


df_train = pd.read_csv('../input/train.csv', index_col='id')
df_test = pd.read_csv('../input/test.csv', index_col='id')
df_train.shape, df_test.shape


# In[ ]:


target = 'target'

X_train = df_train.drop([target], axis=1).values
y_train = df_train[target]


# In[ ]:


from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from stability_selection import StabilitySelection


# In[ ]:


Cs = np.logspace(-5, 5, 21)

pipe = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(solver='lbfgs', verbose=1, n_jobs=1, random_state=42))
])

ss = StabilitySelection(base_estimator=pipe, 
                        lambda_name='clf__C', lambda_grid=Cs, 
                        n_jobs=-1, verbose=1, random_state=42)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'ss.fit(X_train, y_train)')


# In[ ]:


ss.stability_scores_[0]


# In[ ]:


ss


# In[ ]:


from stability_selection import plot_stability_path


# In[ ]:


fig, ax = plot_stability_path(ss, figsize=(12,8))
ax.set_xscale('log')


# In[ ]:


ss_indices = np.where(ss.get_support() == True)[0]
ss_indices


# In[ ]:


len(ss_indices)


# In[ ]:


X_train = ss.transform(X_train)
X_train.shape


# After selecting stable features let's try to select 5..10 most significant of them using `SequentialFeatureSelector` from *mlxtend* package

# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from mlxtend.feature_selection import SequentialFeatureSelector


# In[ ]:


pipe = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(solver='lbfgs', verbose=1, n_jobs=1, random_state=42))
])

cv = RepeatedStratifiedKFold(2, 50, random_state=42)

sfs = SequentialFeatureSelector(pipe, k_features=(5,10), floating=True,
                                scoring='roc_auc', cv=cv, verbose=1, n_jobs=-1)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'sfs.fit(X_train, y_train)')


# In[ ]:


for idx in sfs.k_feature_idx_:
    print(ss_indices[idx])


# In[ ]:


len(sfs.k_feature_names_)


# In[ ]:


import matplotlib.pyplot as plt


# In[ ]:


scores = [v['avg_score'] for k, v in sfs.subsets_.items()]
plt.plot(scores)


# In[ ]:


X_train = sfs.transform(X_train)
X_train.shape


# Let's find the best value of regularization penalty hyperparameter with selected features

# In[ ]:


from sklearn.model_selection import GridSearchCV


# In[ ]:


Cs = np.logspace(-5, 5, 21)

params = {
    'clf__C': Cs
}

pipe = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(solver='lbfgs', verbose=1, n_jobs=1, random_state=42))
])

cv = RepeatedStratifiedKFold(2, 100, random_state=42)

grid = GridSearchCV(estimator=pipe, param_grid=params, cv=cv,
                    return_train_score=True,
                    scoring='roc_auc', verbose=1, n_jobs=-1)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'grid.fit(X_train, y_train)')


# In[ ]:


np.max(grid.cv_results_['mean_test_score'])


# In[ ]:


pipe = grid.best_estimator_
pipe


# In[ ]:


import matplotlib.pyplot as plt


# In[ ]:


plt.semilogx(Cs, grid.cv_results_['mean_train_score'], label='train')
plt.semilogx(Cs, grid.cv_results_['mean_test_score'], label='test')
plt.xlabel('C')
plt.ylabel('ROC-AUC')
plt.grid()
plt.legend()


# Fitting model on all data and forecasting

# In[ ]:


X_test = df_test.values
for t in [ss, sfs]:
    X_test = t.transform(X_test)


# In[ ]:


pipe.fit(X_train, y_train)


# In[ ]:


y_pred = pipe.predict_proba(X_test)[:, 1]


# In[ ]:


plt.hist(y_pred, bins=50)
pass


# In[ ]:


df_subm = pd.read_csv('../input/sample_submission.csv', index_col='id')


# In[ ]:


df_subm[target] = y_pred


# In[ ]:


df_subm.head()


# In[ ]:


df_subm.to_csv('submission.csv')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




