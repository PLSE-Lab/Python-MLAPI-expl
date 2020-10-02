#!/usr/bin/env python
# coding: utf-8

# 
# ## Description
# 
# UPDATE: In this version of the kernel we will try to test the idea of selecting features using LOFO. For more details about LOFO please see Ahmet Erdem's kernel available [at this link](https://www.kaggle.com/divrikwicky/instantgratification-lofo-feature-importance). The feature selection step is going to slow down the training process, so this new version will run longer than 1 minute. If you want to see the original kernel that runs less than a minute please refer to Version 1 of this kernel. 
# 
# The original kernel scores 0.99610 on the LB. Unfortunately, we won't be able to use this result as a baseline for comparison because we won't be able to submit our work to LB: in order for LOFO to work, an external package, `lofo-importance`, must be loaded but the usage of external packages is banned by the competion rules. However, it is possible to compute the cross-validation score for the QDA model without LOFO. As a matter of fact, I have already done it in a different kernel: [link](https://www.kaggle.com/graf10a/tuning-512-separate-qda-models) (see the "Repeat Using the Standard Parameters" section). The result was a CV score of 0.96629.  Let's see if selecting features with LOFO can improve this baseline. 
# 
# SPOILER: Basically, the resutl is very inconclusive -- the combined AUC went up from 0.96629 to 0.96727, the fold-average AUC went down from 0.96628 to 0.96213, and the standard deviation increased from 9e-05 to 0.0097. It would be nice to submit it to the LB to see how well it performs.

# ## Setting things up
# ### Loading Libraries

# In[ ]:


import warnings
warnings.filterwarnings('ignore')

import numpy as np 
import pandas as pd 
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler


from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


# ### Loading Data

# In[ ]:


get_ipython().run_cell_magic('time', '', "\ntrain = pd.read_csv('../input/train.csv')\ntest = pd.read_csv('../input/test.csv')\n\ntrain['wheezy-copper-turtle-magic'] = train['wheezy-copper-turtle-magic'].astype('category')\ntest['wheezy-copper-turtle-magic'] = test['wheezy-copper-turtle-magic'].astype('category')")


# ### Computing LOFO Importance
# 
# Here is the adapted code from [Ahmet's notebook](https://www.kaggle.com/divrikwicky/instantgratification-lofo-feature-importance):

# In[ ]:


from lofo import LOFOImportance, FLOFOImportance, plot_importance
from tqdm import tqdm_notebook

def get_model():
    return Pipeline([('scaler', StandardScaler()),
                    ('qda', QuadraticDiscriminantAnalysis(reg_param=0.111))
                   ])

features = [c for c in train.columns if c not in ['id', 'target', 'wheezy-copper-turtle-magic']]


def get_lofo_importance(wctm_num):
    sub_df = train[train['wheezy-copper-turtle-magic'] == wctm_num]
    sub_features = [f for f in features if sub_df[f].std() > 1.5]
    lofo_imp = LOFOImportance(sub_df, target="target",
                              features=sub_features, 
                              cv=StratifiedKFold(n_splits=4, random_state=42, shuffle=True), scoring="roc_auc",
                              model=get_model(), n_jobs=4)
    return lofo_imp.get_importance()

features_to_remove = []
potential_gain = []

n_models=512
for i in tqdm_notebook(range(n_models)):
    imp = get_lofo_importance(i)
    features_to_remove.append(imp["feature"].values[-1])
    potential_gain.append(-imp["importance_mean"].values[-1])
    
print("Potential gain (AUC):", np.round(np.mean(potential_gain), 5))


# ## Building the QDA Classifier with LOFO

# ### Preparing Things for Cross-Validation

# In[ ]:


clf_name='QDA'

NFOLDS=25
RS=42

oof=np.zeros(len(train))
preds=np.zeros(len(test))


# ### Training the Classifiers on All Data

# In[ ]:


get_ipython().run_cell_magic('time', '', "\nprint(f'Cross-validation for the {clf_name} classifier:')\n\ndefault_cols = [c for c in train.columns if c not in ['id', 'target', 'wheezy-copper-turtle-magic']]\n\n# BUILD 512 SEPARATE NON-LINEAR MODELS\nfor i in range(512):  \n    \n    # EXTRACT SUBSET OF DATASET WHERE WHEEZY-MAGIC EQUALS i     \n    X = train[train['wheezy-copper-turtle-magic']==i].copy()\n    Y = X.pop('target').values\n    X_test = test[test['wheezy-copper-turtle-magic']==i].copy()\n    idx_train = X.index \n    idx_test = X_test.index\n    X.reset_index(drop=True,inplace=True)\n\n    #cols = [c for c in X.columns if c not in ['id', 'wheezy-copper-turtle-magic']]\n    cols = [c for c in default_cols if c != features_to_remove[i]]\n    X = X[cols].values             # numpy.ndarray\n    X_test = X_test[cols].values   # numpy.ndarray\n\n    # FEATURE SELECTION (USE APPROX 40 OF 255 FEATURES)\n    vt = VarianceThreshold(threshold=1.5).fit(X)\n    X = vt.transform(X)            # numpy.ndarray\n    X_test = vt.transform(X_test)  # numpy.ndarray   \n\n    # STRATIFIED K FOLD\n    auc_all_folds=np.array([])\n    folds = StratifiedKFold(n_splits=NFOLDS, random_state=RS)\n\n    for fold_num, (train_index, val_index) in enumerate(folds.split(X, Y), 1):\n\n        X_train, Y_train = X[train_index, :], Y[train_index]\n        X_val, Y_val = X[val_index, :], Y[val_index]\n\n        pipe = Pipeline([('scaler', StandardScaler()),\n                         (clf_name, QuadraticDiscriminantAnalysis(reg_param=0.111)),\n                       ])  \n\n        pipe.fit(X_train, Y_train)\n\n        oof[idx_train[val_index]] = pipe.predict_proba(X_val)[:,1]\n        preds[idx_test] += pipe.predict_proba(X_test)[:,1]/NFOLDS\n\n        auc = roc_auc_score(Y_val, oof[idx_train[val_index]])\n        auc_all_folds = np.append(auc_all_folds, auc)\n            \n# PRINT CROSS-VALIDATION AUC FOR THE CLASSFIER\nauc_combo = roc_auc_score(train['target'].values, oof)\nauc_folds_average = np.mean(auc_all_folds)\nstd = np.std(auc_all_folds)/np.sqrt(NFOLDS)\n\nprint(f'The combined CV score is {round(auc_combo,5)}.')    \nprint(f'The folds average CV score is {round(auc_folds_average,5)}.')\nprint(f'The standard deviation is {round(std, 5)}.')")


# ## Creating the Submission File
# 
# All done! At this point we are ready to make our submission file! (We won't be able to submit it but let's make it anyway.)

# In[ ]:


sub = pd.read_csv('../input/sample_submission.csv')
sub['target'] = preds
sub.to_csv('submission.csv',index=False)


# In[ ]:


sub.shape


# In[ ]:


sub.head()

