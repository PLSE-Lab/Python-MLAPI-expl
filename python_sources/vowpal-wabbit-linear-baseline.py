#!/usr/bin/env python
# coding: utf-8

# # Linear model using Vowpal Wabbit
# 
# <img src="https://cdn.dribbble.com/users/261617/screenshots/3146111/vw-dribbble.png" alt="drawing" width="400"/>
# 
# ## What is Vowpal Wabbit?
# This is a package to perform fast training of linear models. It is a very sophysticated tool, that allows to use many different advanced algorithms. I encourage you to check out their [Wiki on github](https://github.com/VowpalWabbit/vowpal_wabbit/wiki)
# 
# For the baseline the two key issues are:
# 
# - it is **fast**. In fact, the package allows online learning, i.e. sequential processing of training examples one-by-one, similar to what SGDClassifier/SGDRegressor models in sklearn aim to achieve. But in command-line mode one truelly reads only a single line  from an input file into memory, thus one can train a model on a dataset that does not fit into memory.
# - it **applies hashing on text features**. 
#    - This means that we do not need to run much of pre-processing and can let the machine to do the learning. That's what is implemented in this baseline- we directly feed the message text into the training removing punctuation and stopwords only (the latter is not needed in fact). 
#    - This also means that the text features are stored in a more compact form that the naive OHE (=BoW) representation, thus memory footprint is reduced.
#    
# In the following kernel the sklearn API of VW is used. This allows to use the same data and the same methods to be used in VW as well as in other ML tools. A small technical note: the dataset had to be slightly processed, as the VW internal functions can not properly handle text inputs in a DataFrame.
# 
# The ideas of using class weighting for disbalance problem, threshold optimisation and ngrams come from https://www.kaggle.com/hippskill/vowpal-wabbit-starter-pack (check it out- it is a solid piece of work)

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.simplefilter(action='ignore', category=Warning)

import os
print(os.listdir("../input"))


# Read in the data

# In[ ]:


df_trn = pd.read_csv('../input/train.csv')
df_tst = pd.read_csv('../input/test.csv')


# Print data stats

# In[ ]:


print('Train and test shapes are: {}, {}'.format(df_trn.shape, df_tst.shape))
print('Train and test memory footprint: {:.2f} MB, {:.2f} MB'
      .format(df_trn.memory_usage(deep=True).sum()/ 1024**2,
              df_tst.memory_usage(deep=True).sum()/ 1024**2)
     )
w_pos = df_trn['target'].sum()/df_trn.shape[0]
print('Fraction of positive target (insencere) = {:.4f}'.format(w_pos))


# Display a couple of first entries

# In[ ]:


df_trn.head()


# Stop words

# In[ ]:


from nltk.corpus import stopwords
stops = set(stopwords.words("english"))


# Extract the training features and the target variable

# In[ ]:


import string
def remove_punctuation(s):
    s = ''.join([i for i in s if i not in frozenset(string.punctuation)])
    return s

X_trn = (df_trn['question_text']
         .apply(remove_punctuation)
         .apply(lambda x: ' '.join([w.lower() for w in x.split(' ') if w.lower() not in stops]))
        )
X_trn2 = (df_trn['question_text']
         .apply(remove_punctuation)
         #.apply(lambda x: ' '.join([w.lower() for w in x.split(' ') if w.lower() not in stops]))
        )
X_tst = (df_tst['question_text']
         .apply(remove_punctuation)
         #.apply(lambda x: ' '.join([w.lower() for w in x.split(' ') if w.lower() not in stops]))
        )
y_trn = df_trn['target']

del df_trn, df_tst


# In[ ]:


X_trn.head()


# Helper functions for Vowpal Wabbit

# In[ ]:


import vowpalwabbit as vw
from vowpalwabbit.sklearn_vw import VWClassifier

# VW uses 1/-1 target variables for classification instead of 1/0, so we need to apply mapping
def convert_labels_sklearn_to_vw(y_sklearn):
    return y_sklearn.map({1:1, 0:-1})

# The function to create VW-compatible inputs from the text features and the target
def to_vw(X, y=None, namespace='Name', w=None):
    labels = '1' if y is None else y.astype(str)
    if w is not None:
        labels = labels + ' ' + np.round(y.map({1: w, -1: 1}),5).astype(str)
    prefix = labels + ' |' + namespace + ' '
    if isinstance(X, pd.DataFrame):
        return prefix + X.apply(lambda x: ' '.join(x), axis=1)
    elif isinstance(X, pd.Series):
        return prefix + X


# Define the model that will be trained: our `VW_passes3` model will do 3 iterations (=passes) over the data using the prepared text format as the input. **The threshold to apply for 0/1 label assignment was tuned on the data to get the best F1 score**

# In[ ]:


mdl_inputs = {
# VW1x is analogous to the configuration from the kernel cited in the intro
#                 'VW1x': [VWClassifier(quiet=False, convert_to_vw=False, 
#                                      passes=1, link='logistic',
#                                      random_seed=314),
#                              {'pos_threshold':0.5, 
#                               'b':29, 'ngram':2, 'skips': 1, 
#                               'l1':3.4742122764e-09, 'l2':1.24232077629e-11},
#                              {},
#                              None,
#                              None,
#                              1./w_pos
#                             ],
                'VW1': [VWClassifier(quiet=False, convert_to_vw=False, 
                                     passes=3, link='logistic',
                                     random_seed=314),
                             {'pos_threshold':0.5},
                             {},
                             None,
                             None,
                             1./w_pos
                            ],
#                 'VW2': [VWClassifier(quiet=False, convert_to_vw=False, 
#                                      passes=5, link='logistic',
#                                      random_seed=314),
#                              {'pos_threshold':0.5},
#                              {},
#                              None,
#                              None,
#                              1./w_pos
#                             ],
#                 'VW3': [VWClassifier(quiet=False, convert_to_vw=False, 
#                                      passes=10, link='logistic',
#                                      random_seed=314),
#                              {'pos_threshold':0.5},
#                              {},
#                              None,
#                              None,
#                              1./w_pos
#                             ],
         }

# for i in [22]:
#     mdl_inputs['VW_passes3_thrs{}'.format(i)] = mdl_inputs['VW_passes3'].copy()
#     mdl_inputs['VW_passes3_thrs{}'.format(i)][1] = {'pos_threshold':i/100.}


# The function to do training in a cross-validation loop and evaluate performance of the model

# In[ ]:


from sklearn.model_selection import StratifiedKFold, KFold, GroupKFold
from sklearn.base import clone, ClassifierMixin, RegressorMixin

def train_single_model(clf_, X_, y_, random_state_=314, opt_parameters_={}, fit_params_={}):
    '''
    A wrapper to train a model with particular parameters
    '''
    c = clone(clf_)
    
    param_dict = {}
    if 'VW' in type(c).__name__:
        # we need to get ALL parameters, as the VW instance is destroyed on set_params
        param_dict = c.get_params()
        # the threshold is lost in the cloning
        param_dict['pos_threshold'] = clf_.pos_threshold
        param_dict.update(opt_parameters_)
        # the random_state is random_seed so far
        param_dict.update({'random_seed': random_state_})
        if hasattr(c, 'fit_'):
            # reset VW if it has already been trained
            c.get_vw().finish()
            c.vw_ = None 
    else:
        param_dict = opt_parameters_
        param_dict['random_state'] = random_state_
    # Set pre-configured parameters
    c.set_params(**param_dict)
    #print('Threshold = ',c.pos_threshold)
    
    return c.fit(X_, y_, **fit_params_)

def train_model_in_CV(model, X, y, metric, metric_args={},
                            model_name='xmodel',
                            seed=31416, n=5,
                            opt_parameters_={}, fit_params_={},
                            verbose=True,
                            groups=None, 
                            y_eval=None,
                            w_=1.):
    # the list of classifiers for voting ensable
    clfs = []
    # performance 
    perf_eval = {'score_i_oof': 0,
                 'score_i_ave': 0,
                 'score_i_std': 0,
                 'score_i': []
                }

    cv = KFold(n, shuffle=True, random_state=seed) #Stratified

    scores = []
    clfs = []

    for n_fold, (trn_idx, val_idx) in enumerate(cv.split(X, (y!=0).astype(np.int8), groups=groups)):
        X_trn, y_trn = X.iloc[trn_idx], y.iloc[trn_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        X_trn_vw = to_vw(X_trn, convert_labels_sklearn_to_vw(y_trn), w=w_).values
        X_val_vw = to_vw(X_val, convert_labels_sklearn_to_vw(y_val), w=w_).values

        #display(y_trn.head())
        clf = train_single_model(model, X_trn_vw, None, 314+n_fold, opt_parameters_, fit_params_)
        #plt.hist(clf.decision_function(X_val_vw), bins=50)
        
        if 'VW' in type(clf).__name__:
            x_thres = np.linspace(0.05, 0.95, num=37)
            y_f1    = []
            for thres in x_thres:
                # predict on the validation sample
                y_pred_tmp = (clf.decision_function(X_val_vw) > thres).astype(int)
                y_f1.append(metric(y_val, y_pred_tmp, **metric_args))
            i_opt = np.argmax(y_f1)

            clf.pos_threshold = x_thres[i_opt]
            #print('Optimal threshold = {:.4f}'.format(clf.pos_threshold))
        
        # predict on the validation sample
        y_pred_tmp = (clf.decision_function(X_val_vw) > clf.pos_threshold).astype(int)
        #store evaluated metric
        scores.append(metric(y_val, y_pred_tmp, **metric_args))
        
        # store the model
        clfs.append(('{}{}'.format(model_name,n_fold), clf))
        
        #cleanup
        del X_trn, y_trn, X_val, y_val, y_pred_tmp, X_trn_vw, X_val_vw

    #plt.show()
    perf_eval['score_i_oof'] = 0
    perf_eval['score_i'] = scores            
    perf_eval['score_i_ave'] = np.mean(scores)
    perf_eval['score_i_std'] = np.std(scores)

    return clfs, perf_eval, None

def print_perf_clf(name, perf_eval, fmt='.4f'):
    print('Performance of the model:')    
    print('Mean(Val) score inner {} Classifier: {:{fmt}}+-{:{fmt}}'.format(name, 
                                                                       perf_eval['score_i_ave'],
                                                                       perf_eval['score_i_std'],
                                                                       fmt=fmt
                                                                     ))
    print('Min/max scores on folds: {:{fmt}} / {:{fmt}}'.format(np.min(perf_eval['score_i']),
                                                            np.max(perf_eval['score_i']),
                                                            fmt=fmt
                                                           ))
    print('OOF score inner {} Classifier: {:{fmt}}'.format(name, perf_eval['score_i_oof'], fmt=fmt))
    print('Scores in individual folds: [{}]'
          .format(' '.join(['{:{fmt}}'.format(c, fmt=fmt) 
                            for c in perf_eval['score_i']
                           ])
                 )
         )


# Actual training of the model

# --------------- VW1 -----------
# Performance of the model:
# Mean(Val) score inner VW1 Classifier: 0.4984+-0.0122
# Min/max scores on folds: 0.4783 / 0.5154
# OOF score inner VW1 Classifier: 0.0000
# Scores in individual folds: [0.4952 0.4978 0.4783 0.5154 0.5054]
# --------------- VW1_l1_1em7 -----------
# Performance of the model:
# Mean(Val) score inner VW1_l1_1em7 Classifier: 0.4987+-0.0123
# Min/max scores on folds: 0.4787 / 0.5161
# OOF score inner VW1_l1_1em7 Classifier: 0.0000
# Scores in individual folds: [0.4956 0.4981 0.4787 0.5161 0.5050]
# --------------- VW1_l1_1em5 -----------
# Performance of the model:
# Mean(Val) score inner VW1_l1_1em5 Classifier: 0.5009+-0.0104
# Min/max scores on folds: 0.4853 / 0.5149
# OOF score inner VW1_l1_1em5 Classifier: 0.0000
# Scores in individual folds: [0.4983 0.4853 0.4964 0.5097 0.5149]
# --------------- VW1_l2_1em7 -----------
# Performance of the model:
# Mean(Val) score inner VW1_l2_1em7 Classifier: 0.4985+-0.0121
# Min/max scores on folds: 0.4787 / 0.5154
# OOF score inner VW1_l2_1em7 Classifier: 0.0000
# Scores in individual folds: [0.4952 0.4978 0.4787 0.5154 0.5054]
# --------------- VW1_l2_1em5 -----------
# Performance of the model:
# Mean(Val) score inner VW1_l2_1em5 Classifier: 0.5037+-0.0108
# Min/max scores on folds: 0.4875 / 0.5200
# OOF score inner VW1_l2_1em5 Classifier: 0.0000
# Scores in individual folds: [0.5030 0.4990 0.4875 0.5200 0.5094]
# --------------- VW2 -----------
# Performance of the model:
# Mean(Val) score inner VW2 Classifier: 0.4430+-0.0058
# Min/max scores on folds: 0.4373 / 0.4536
# OOF score inner VW2 Classifier: 0.0000
# Scores in individual folds: [0.4390 0.4373 0.4444 0.4406 0.4536]
# --------------- VW2_l1_1em7 -----------
# Performance of the model:
# Mean(Val) score inner VW2_l1_1em7 Classifier: 0.4461+-0.0059
# Min/max scores on folds: 0.4401 / 0.4569
# OOF score inner VW2_l1_1em7 Classifier: 0.0000
# Scores in individual folds: [0.4401 0.4413 0.4450 0.4469 0.4569]
# --------------- VW2_l1_1em5 -----------
# Performance of the model:
# Mean(Val) score inner VW2_l1_1em5 Classifier: 0.4768+-0.0102
# Min/max scores on folds: 0.4640 / 0.4936
# OOF score inner VW2_l1_1em5 Classifier: 0.0000
# Scores in individual folds: [0.4701 0.4744 0.4640 0.4820 0.4936]
# --------------- VW2_l2_1em7 -----------
# Performance of the model:
# Mean(Val) score inner VW2_l2_1em7 Classifier: 0.4432+-0.0070
# Min/max scores on folds: 0.4361 / 0.4555
# OOF score inner VW2_l2_1em7 Classifier: 0.0000
# Scores in individual folds: [0.4361 0.4371 0.4420 0.4452 0.4555]
# --------------- VW2_l2_1em5 -----------
# Performance of the model:
# Mean(Val) score inner VW2_l2_1em5 Classifier: 0.4501+-0.0124
# Min/max scores on folds: 0.4330 / 0.4707
# OOF score inner VW2_l2_1em5 Classifier: 0.0000
# Scores in individual folds: [0.4330 0.4495 0.4439 0.4532 0.4707]

# --------------- VW1_l1_1em4 -----------
# Performance of the model:
# Mean(Val) score inner VW1_l1_1em4 Classifier: 0.3987+-0.0173
# Min/max scores on folds: 0.3705 / 0.4185
# OOF score inner VW1_l1_1em4 Classifier: 0.0000
# Scores in individual folds: [0.4185 0.3911 0.3705 0.3993 0.4142]
# --------------- VW1_l1_1em3 -----------
# Performance of the model:
# Mean(Val) score inner VW1_l1_1em3 Classifier: 0.1171+-0.0024
# Min/max scores on folds: 0.1132 / 0.1203
# OOF score inner VW1_l1_1em3 Classifier: 0.0000
# Scores in individual folds: [0.1203 0.1169 0.1132 0.1185 0.1164]
# --------------- VW1_l2_1em4 -----------
# Performance of the model:
# Mean(Val) score inner VW1_l2_1em4 Classifier: 0.4889+-0.0128
# Min/max scores on folds: 0.4659 / 0.5017
# OOF score inner VW1_l2_1em4 Classifier: 0.0000
# Scores in individual folds: [0.4848 0.4981 0.4659 0.5017 0.4939]
# --------------- VW1_l2_1em3 -----------
# Performance of the model:
# Mean(Val) score inner VW1_l2_1em3 Classifier: 0.3869+-0.0157
# Min/max scores on folds: 0.3606 / 0.4039
# OOF score inner VW1_l2_1em3 Classifier: 0.0000
# Scores in individual folds: [0.3950 0.3777 0.3606 0.3972 0.4039]
# --------------- VW2_l1_1em4 -----------
# Performance of the model:
# Mean(Val) score inner VW2_l1_1em4 Classifier: 0.1171+-0.0024
# Min/max scores on folds: 0.1132 / 0.1203
# OOF score inner VW2_l1_1em4 Classifier: 0.0000
# Scores in individual folds: [0.1203 0.1169 0.1132 0.1185 0.1164]
# --------------- VW2_l1_1em3 -----------
# Performance of the model:
# Mean(Val) score inner VW2_l1_1em3 Classifier: 0.1171+-0.0024
# Min/max scores on folds: 0.1132 / 0.1203
# OOF score inner VW2_l1_1em3 Classifier: 0.0000
# Scores in individual folds: [0.1203 0.1169 0.1132 0.1185 0.1164]
# --------------- VW2_l2_1em4 -----------
# Performance of the model:
# Mean(Val) score inner VW2_l2_1em4 Classifier: 0.4441+-0.0153
# Min/max scores on folds: 0.4260 / 0.4705
# OOF score inner VW2_l2_1em4 Classifier: 0.0000
# Scores in individual folds: [0.4367 0.4501 0.4260 0.4370 0.4705]
# --------------- VW2_l2_1em3 -----------
# Performance of the model:
# Mean(Val) score inner VW2_l2_1em3 Classifier: 0.2886+-0.0157
# Min/max scores on folds: 0.2693 / 0.3157
# OOF score inner VW2_l2_1em3 Classifier: 0.0000
# Scores in individual folds: [0.2866 0.3157 0.2783 0.2932 0.2693]
# 

# In[ ]:


get_ipython().run_cell_magic('time', '', "from sklearn.metrics import f1_score\n\nmdls = {}\nresults = {}\ny_oofs = {}\nfor name, (mdl, mdl_pars, fit_pars, y_, g_, w_) in mdl_inputs.items():\n    print('--------------- {} -----------'.format(name))\n    mdl_, perf_eval_, y_oof_ = train_model_in_CV(mdl, X_trn2.iloc[:],\n                                                  y_trn.iloc[:], f1_score, \n                                                  metric_args={},\n                                                  model_name=name, \n                                                  opt_parameters_=mdl_pars,\n                                                  fit_params_=fit_pars, \n                                                  n=5,\n                                                  verbose=500, \n                                                  groups=g_, \n                                                  y_eval=None if 'LGBMRanker' not in type(mdl).__name__ else y_rnk_eval,\n                                                  w_=w_\n                                                )\n    results[name] = perf_eval_\n    mdls[name] = mdl_\n    y_oofs[name] = y_oof_\n    print_perf_clf(name, perf_eval_)")


# ## Prepare submission

# Models, that were trained on k sets of k-1 folds are averaged before application of the decision threshold

# In[ ]:


get_ipython().run_cell_magic('time', '', 'y_subs= {}\nX_tst_vw = to_vw(X_tst, None).values\nfor c in mdl_inputs:\n    mdls_= mdls[c]\n    y_sub = np.zeros(X_tst_vw.shape[0])\n    for mdl_ in mdls_:\n        y_sub += mdl_[1].decision_function(X_tst_vw)\n    y_sub /= len(mdls_)\n    \n    y_subs[c] = y_sub')


# In[ ]:


df_sub = pd.read_csv('../input/sample_submission.csv')


# In[ ]:


s= 'VW1'#VW_passes3_w
df_sub['prediction'] = (y_subs[s] > np.median([mdl_[1].pos_threshold for mdl_ in mdls[s]])).astype(int)
df_sub.to_csv('submission.csv', index=False)


# In[ ]:


get_ipython().system('head submission.csv')


# In[ ]:




