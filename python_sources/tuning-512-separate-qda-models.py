#!/usr/bin/env python
# coding: utf-8

# UPDATED!!!
# 
# ## The Purpose of This Notebook
# 
# In earlier versions of the notebook we were investigating the question of parameter tuning for the QDA model. Now, after the power of pseudolabeling was finally revealed the question becomes: What would be the best values of the QDA parameters with pseudolabeling? This notebook does not answer this question completely but it gives you the right tool to do this investigation on your own. 
# 
# ## Short Description of the Method
# 
# Our method is very simple: we generate 100 random values for the QDA parameter `reg_param` and then train 100 different models: one model for each value of `reg_param`. Then the value that maximizes ROC AUC score is identified and the results are visualized in the form of the AUC vs `reg_parm` scatter plot.
# 
# Here is the list of our assumptions:
# 
# * If you remember how pseudolableing with QDA works (which was very well explained in [Roman's](https://www.kaggle.com/nroman/i-m-overfitting-and-i-know-it) and [Chris's](https://www.kaggle.com/cdeotte/psuedo-labeling-qda-0-969) kernels) then you remember that we need to train QDA twice: one time without pseudolableing and the other with pseudolabeling. In what follows we will assume for simplicity that for both of these trainings the same value of `reg_param` is used. It is not difficult to modify the code below if you want to explore non-equal values of the parameters. 
# 
# * We also use `lowest=0.01`, `highest=0.99`, the same values as in Chris's notebook. Those can be easily adjusted as well. 
# 
# * In this notebook, we will do 100 trials with 5-fold cross-validation each. This can be easily adjusted by modifying the values of the parameter `NTRIALS` and `NFODS` below.
# 
# Enjoy!

# ## Preparatory Work

# ### Loading Libraries

# In[ ]:


import time
import warnings
import multiprocessing
from pathlib import Path
warnings.filterwarnings('ignore')

import numpy as np 
import pandas as pd 
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import VarianceThreshold
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


# ### Loading Data

# In[ ]:


get_ipython().run_cell_magic('time', '', "\npath=Path('../input')\n\ndef load_data(data):\n    return pd.read_csv(data)\n\nwith multiprocessing.Pool() as pool:\n    train, test, sub = pool.map(load_data, [path/'train.csv', \n                                            path/'test.csv', \n                                            path/'sample_submission.csv'])")


# ### Preparing Things for Cross-Validation

# Defining the optional parameters.

# In[ ]:


NFOLDS=5
NTRIALS=100
RS=42
debug=0

lowest=0.01
highest=0.99


# Checking and handling the debuging mode (low values of `magic_max` and `NFOLDS` save a lot of time; the latter breaks cross-validation):

# In[ ]:


if debug:
    magic_max=2
    magic_min=0
    NFOLDS=2
    NTRIALS=2
else:
    magic_max=train['wheezy-copper-turtle-magic'].max()
    magic_min=train['wheezy-copper-turtle-magic'].min()


# Define the preprocessing function applying variance threshold to data grouped by the values of the `wheezy-copper-turtle-magic` variable.

# In[ ]:


def preprocess(clfs=['QDA'], train=train, test=test, magic_min=magic_min, magic_max=magic_max):
    
    prepr = {}
    
    #PREPROCESS 512 SEPARATE MODELS
    for i in range(magic_min, magic_max+1):

        # EXTRACT SUBSET OF DATASET WHERE WHEEZY-MAGIC EQUALS i     
        X = train[train['wheezy-copper-turtle-magic']==i].copy()
        Y = X.pop('target').values
        X_test = test[test['wheezy-copper-turtle-magic']==i].copy()
        idx_train = X.index 
        idx_test = X_test.index
        X.reset_index(drop=True,inplace=True)

        cols = [c for c in X.columns if c not in ['id', 'wheezy-copper-turtle-magic']]

        l=len(X)
        X_all = pd.concat([X[cols], X_test[cols]], ignore_index=True)

        X_vt = VarianceThreshold(threshold=1.5).fit_transform(X_all)              # np.ndarray
        
        prepr['vt_' + str(i)] = X_vt        
        prepr['train_size_' + str(i)] = l
        prepr['idx_train_' + str(i)] = idx_train
        prepr['idx_test_' + str(i)] = idx_test
        prepr['target_' + str(i)] = Y
        
    return prepr


# In[ ]:


get_ipython().run_cell_magic('time', '', '\ndata = preprocess()')


# In[ ]:


def get_data(i, data):
    
    l = data['train_size_' + str(i)]    

    X_all = data['vt_' + str(i)]                

    X = X_all[:l, :]
    X_test = X_all[l:, :]

    Y = data['target_' + str(i)]

    idx_train = data['idx_train_' + str(i)]
    idx_test = data['idx_test_' + str(i)]
    
    return X, X_test, Y, idx_train, idx_test


# In[ ]:


def pseudolabeling(X_train, X_test, Y_train, Y_pseudo, 
                   idx_test, lowest=lowest, highest=highest, test=test):
    
    assert len(test) == len(Y_pseudo), "The length of test does not match that of Y_pseudo!"
    
    #SELECT ONLY THE PSEUDOLABLES CORRESPONDING TO THE CURRENT VALUES OF 'wheezy-copper-turtle-magic'
    Y_aug = Y_pseudo[idx_test]
    
    assert len(Y_aug) == len(X_test), "The length of Y_aug does not match that of X_test!"

    Y_aug[Y_aug > highest] = 1
    Y_aug[Y_aug < lowest] = 0
    
    mask = (Y_aug == 1) | (Y_aug == 0)
    
    Y_useful = Y_aug[mask]
    X_test_useful = X_test[mask]
    
    X_train_aug = np.vstack((X_train, X_test_useful))
    Y_train_aug = np.vstack((Y_train.reshape(-1, 1), Y_useful.reshape(-1, 1)))
    
    return X_train_aug, Y_train_aug


# In[ ]:


def train_classifier(clf_name, clfs, data=data, train=train, test=test, 
                     debug=debug, NFOLDS=NFOLDS, RS=RS, Y_pseudo=None,
                     magic_min=magic_min, magic_max=magic_max,
                     lowest=lowest, highest=highest, verbose=1):
    
    auc_all = np.array([])
    oof = np.zeros(len(train))
    preds = np.zeros(len(test))    
    
    #TRAIN 512 SEPARATE MODELS
    for i in range(magic_min, magic_max+1):
        
        X, X_test, Y, idx_train, idx_test = get_data(i=i, data=data)      
   
        # STRATIFIED K FOLD    
        folds = StratifiedKFold(n_splits=NFOLDS, random_state=RS)
        
        auc_folds = np.array([])
        
        for train_index, val_index in folds.split(X, Y):     

            X_train, Y_train = X[train_index, :], Y[train_index]
            X_val, Y_val = X[val_index, :], Y[val_index]
            
            if Y_pseudo is not None:
                X_train_aug, Y_train_aug = pseudolabeling(X_train, X_test, 
                                                          Y_train, Y_pseudo, idx_test, 
                                                          lowest=lowest, highest=highest, 
                                                          test=test)
                clfs[clf_name].fit(X_train_aug, Y_train_aug)                
            else:
                clfs[clf_name].fit(X_train, Y_train)

            oof[idx_train[val_index]] = clfs[clf_name].predict_proba(X_val)[:,1]
            preds[idx_test] += clfs[clf_name].predict_proba(X_test)[:,1]/NFOLDS

            auc = roc_auc_score(Y_val, oof[idx_train[val_index]])
            auc_folds = np.append(auc_folds, auc)
                 
        auc_all = np.append(auc_all, np.mean(auc_folds))
        
    auc_combo = roc_auc_score(train['target'].values, oof)
    auc_av = np.mean(auc_all)
    std = np.std(auc_all)/(np.sqrt(NFOLDS)*np.sqrt(magic_max+1))
    
    if verbose:    
        # PRINT VALIDATION CV AUC FOR THE CLASSFIER
        print(f'The result summary for the {clf_name} classifier:')
        print(f'The combined CV score is {round(auc_combo, 5)}.')    
        print(f'The folds average CV score is {round(auc_av, 5)}.')
        print(f'The standard deviation is {round(std, 5)}.\n')
    
    return preds, auc_combo


# ## Parameter Search
# 
# ### Trying Different Values of the Parameters

# In[ ]:


get_ipython().run_cell_magic('time', '', '\nresults = {}\nresults[\'rp\']=np.array([])\nresults[\'auc\']=np.array([])\n        \nnp.random.seed(RS)\n\nfor j in range(NTRIALS):\n\n    rp=10**(-2*np.random.rand()) # sampling values between 0.01 and 1\n       \n    # KEY: NAME, VALUE: [CLASSIFIER, DO_RANKING]\n    clfs_init={\'QDA\': QuadraticDiscriminantAnalysis(reg_param=rp)}\n\n    clfs={\'QDA\': QuadraticDiscriminantAnalysis(reg_param=rp)}\n\n    Y_pseudo, _ = train_classifier(\'QDA\', clfs=clfs_init, verbose=0)\n\n    _, auc = train_classifier(\'QDA\', clfs=clfs, Y_pseudo=Y_pseudo, verbose=0)\n        \n    results[\'rp\']=np.append(results[\'rp\'], rp)\n    results[\'auc\']=np.append(results[\'auc\'], auc)\n        \n    print(f"Trial number {j}: AUC = {round(auc, 5)}, rp={round(rp, 5)}.\\n")   ')


# ### Summary of the Results

# In[ ]:


auc_max = np.max(results['auc'])
i_max = np.argmax(results['auc'])
rp_best = results['rp'][i_max]

print(f"The highest AUC achived is {round(auc_max, 5)} for rp={round(rp_best, 5)}.")

auc_min = np.min(results['auc'])
i_min = np.argmin(results['auc'])

print(f"The lowest AUC achived is {round(auc_min, 5)} for rp={round(results['rp'][i_min], 5)}.")

#CHECK IF THE BEST VALUE IS ON THE BOUNDARY
print(f"The smallest value of `reg_param` that was explored during the search is {round(np.min(results['rp']), 5)}.")
print(f"The larges value of `reg_param` that was explored during the search is {round(np.max(results['rp']), 5)}.")


# ### Visualizing the Results of the Search

# In[ ]:


import matplotlib.pyplot as plt

plt.scatter(results['rp'], results['auc'], s=4)
plt.xlabel('reg_param')
plt.ylabel('ROC AUC')


# ## Submission
# 
# ### Training the Classifier with the Best Parameters

# In[ ]:


clfs_best = {'QDA': QuadraticDiscriminantAnalysis(reg_param=rp_best)}

preds_best, auc_best = train_classifier('QDA', clfs=clfs_best, Y_pseudo=Y_pseudo, verbose=0)

print(f"AUC: {auc_best}")


# ### Creating the Submission File

# In[ ]:


sub['target'] = preds_best
sub.to_csv('submission.csv',index=False)

