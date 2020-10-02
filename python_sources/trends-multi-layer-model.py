#!/usr/bin/env python
# coding: utf-8

# ### Parameters

# In[ ]:


SEED = 123      # modifiable seed
CLF_SS = 1      # sub-sample model types for faster run
TARGETS = -1    # which target (0-4) to predict; -1 for all


# In[ ]:





# In[ ]:





# In[ ]:





# ### Imports

# In[ ]:


import numpy as np  
import pandas as pd 
import pickle


# In[ ]:


import multiprocessing
from joblib import Parallel, delayed


# In[ ]:


from collections import Counter
import datetime as datetime


# In[ ]:


import gc
import psutil
import sys


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
from pylab import rcParams
rcParams['figure.figsize'] = (15,5.5)

pd.options.display.max_rows = 150


# In[ ]:


start = datetime.datetime.now()

if SEED < 0:
    np.random.seed(datetime.datetime.now().microsecond)
else:
    np.random.seed(SEED)


# In[ ]:





# ### Data Loading

# In[ ]:


path = '/kaggle/input/trends-assessment-prediction/'

loading =  pd.read_csv(path+ '/' + 'loading.csv').set_index('Id')
fnc =  pd.read_csv(path+ '/' + 'fnc.csv').set_index('Id')
assert len(loading) == len(fnc)


# In[ ]:


y_data =  pd.read_csv(path+ '/' + 'train_scores.csv').set_index('Id')

data = pd.concat((loading, fnc,  ), axis = 'columns')  
test_data = data[~data.index.isin(y_data.index)]

X = data.loc[y_data.index] 
y = y_data 
groups = np.random.randint(0, 5, len(y))


# In[ ]:





# ### Model Setup

# In[ ]:


from sklearn.model_selection import RandomizedSearchCV, RepeatedKFold, KFold, ShuffleSplit
from sklearn.svm import SVR, NuSVR
from sklearn.linear_model import ElasticNet, Ridge, Lasso
from sklearn.model_selection import ParameterSampler
from sklearn.metrics import make_scorer, mean_absolute_error
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler, FunctionTransformer 


# In[ ]:


nusvr_params = {
    'kernel': [  'rbf',  ] , 
    'C': [ 1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 70, 100, 140, 200, 300  ],
    'gamma': [ 'scale'], 
    'nu': [   0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1] }

def trainNuSVR(x, y, groups, cv = 0, n_jobs = -1, **kwargs):
    clf = NuSVR(cache_size=1000, tol = 1e-5)
    params = nusvr_params        
    return trainModel(x, y, groups, clf, params, cv, n_jobs,  **kwargs)


# In[ ]:





# In[ ]:


enet_params = { 'alpha': [  1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 3e-2, 0.1, 0.3,   ],
                'l1_ratio': [ 0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.95, 0.97, 0.98, 0.99, 1,   ]}

def trainENet(x, y, groups, cv = 0, n_jobs = -1, **kwargs):
    clf = ElasticNet(normalize = True, selection = 'random', max_iter = 10000, tol = 1e-5 )
    return trainModel(x, y, groups, clf, enet_params, cv, n_jobs, **kwargs)


# In[ ]:





# In[ ]:


def fnae(y_true, y_pred):
    valid = ~np.isnan(y_true)
    y_true = y_true[valid]
    y_pred = y_pred[valid]
    return np.sum(np.abs(y_true - y_pred))/np.sum(y_true)

fnae_scorer = make_scorer(fnae, greater_is_better = False)


# In[ ]:


def trainModel(x, y, groups, clf, params, cv = 0, n_jobs = None, 
                   verbose=0, splits=None, **kwargs):
    if n_jobs is None:
        n_jobs = -1    

    n_iter = 30    
        
    folds = ShuffleSplit(n_splits = 10, train_size = 0.75, test_size = 0.20)
    clf = RandomizedSearchCV(clf, params, cv = folds, n_iter = n_iter, 
                            verbose = 1, n_jobs = n_jobs, scoring = fnae_scorer)
    
    f = clf.fit(x, y, groups)
    
    print(pd.DataFrame(clf.cv_results_['mean_test_score'])); print();  
    best = clf.best_estimator_;  print(best)
    print("Best Score: {}".format(np.round(clf.best_score_,4)))
    
    return best


# In[ ]:





# In[ ]:


def cleanX(X, target):
    X = X.copy()
    
    for col in fnc.columns:
        X[col] = X[col] / 300
       
    return X;


# In[ ]:





# In[ ]:


def runBag(n = 3, model_type = trainENet, data = None, **kwargs):
    start_time = datetime.datetime.now(); 
    
    X, y, groups = data

    valid = ~y.isnull()
    X = X[valid]; y = y[valid]; groups = groups[valid]
    
    if 'target' in kwargs:
        X = cleanX(X, kwargs['target'])
    
    group_list = [*dict.fromkeys(groups)]   
    group_list.sort()
    
    clfs = []; preds = []; ys=[]; datestack = []
    for group in group_list:
        g = gc.collect()
        x_holdout = X[groups == group]
        y_holdout = y[groups == group]
        x_train = X[groups != group]
        y_train = y[groups != group]
        
        groups_train = groups[groups != group]

        model = model_type 
        clf = model(x_train, y_train, groups_train, **kwargs) 
        clfs.append(clf)

        predicted = clf.predict(x_holdout)
        print("{}: {:.4f}".format(group,
              fnae(y_holdout, predicted)  ) )
        
        preds.append(predicted)
        ys.append(y_holdout)
    
    y_pred = np.concatenate(preds)
    y_ho = np.concatenate(ys) 

    end_time = datetime.datetime.now(); 
    print("\nModel Bag Time: {}\n".format(str(end_time - start_time).split('.', 2)[0] ))
    return clfs


# In[ ]:





# In[ ]:





# In[ ]:


def trainBaseClfs(clfs, clf_names, data, target = None, **kwargs):
    start_time = datetime.datetime.now(); 
    
    X, y, groups = data
    
    X = cleanX(X, target)
    
    group_list = [*dict.fromkeys(groups)]   
    group_list.sort()
    
    X_ordered = []; y_ordered = []; groups_ordered =[]  
    all_base_clfs = []; base_preds = [[] for i in range(0, 5 * len(clfs))]; 
    for group in group_list:
        print("Training Fold {} of {}:".format(group, len(group_list)))
        np.random.seed(SEED)
        
        x_holdout = X[groups == group]
        y_holdout = y[groups == group]
        x_train = X[groups != group]
        y_train = y[groups != group]

        y_idx = ALL_TARGETS.index(target)
        
        X_ordered.append(x_holdout)
        y_ordered.append(y_holdout)
        groups_ordered.append(groups[groups == group])
        
        base_clfs = []
        for idx, clf in enumerate(clfs):
            base_clfs.append(clone(clf))
        
        def train_model(model, X, y):
            ss = (~pd.DataFrame(y).isnull().any(axis=1))
            model.fit(X[ss], y[ss]); return model
        
        base_clfs = Parallel(n_jobs=4)(delayed(train_model)(model, x_train, y_train[y_var]) for model in base_clfs)
        all_base_clfs.append(base_clfs)
        
        def predict_model(model, X):
            o = model.predict(X); return o    
        preds = Parallel(n_jobs=4)(delayed(predict_model)(model, x_holdout) for model in base_clfs)
        
        
        pidx = 0; clf_pred_names = []
        for idx, clf in enumerate(base_clfs):   
            print("{:.4f} for {}".format( 
                      fnae(y_holdout[target], preds[idx]), clf_names[idx]  ) )
            base_preds[pidx].append(preds[idx]); pidx+=1;
            clf_pred_names.append(clf_names[idx])
            
        print("\nTime Elapsed: {}\n".format(str(datetime.datetime.now() - start_time).split('.', 2)[0] ))

    base_preds = base_preds[:len(clf_pred_names)]
    for idx in range(0, len(base_preds)):
        base_preds[idx] = np.concatenate(base_preds[idx])

    
    print("\Base Classifier Train Time: {}\n".format(str(datetime.datetime.now() - start_time).split('.', 2)[0] ))
    return (all_base_clfs, base_preds, clf_pred_names, 
        pd.concat(X_ordered), pd.concat(y_ordered), np.concatenate(groups_ordered))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


def Lassos():
    clfs = []; clf_names = []
    lassos =  [1e-5, 3e-5, 1e-4,  3e-4,  0.001, 0.003,  0.01,  0.03,  0.1,  0.3,  1, ]
    for l in lassos:
        clfs.append(Lasso(alpha = l,  selection = 'random', max_iter = 5000, tol = 1e-5))
        clf_names.append('Lasso alpha={}'.format(l))
        if CLF_SS > 1:
            clfs.append(clfs[-1]); clf_names.append(clf_names[-1])
 
    return clfs, clf_names


# In[ ]:


def Ridges():
    clfs = []; clf_names = []
    ridges =  [3e-5,  1e-4,  2e-4, 5e-4, 0.001, 0.002, 0.005,  0.01,  0.03,  0.1,  0.3,  1,  3,  10,    ]
    for r in ridges:
        clfs.append(Ridge(alpha = r, max_iter = 5000, tol = 1e-5))
        clf_names.append('Ridge alpha={}'.format(r))
        if CLF_SS > 1:
            clfs.append(clfs[-1]); clf_names.append(clf_names[-1])

    return clfs, clf_names


# In[ ]:


def SVRs():
    clfs = []; clf_names = []
    svrs =  ([0.2, 1, 7, 50], [1, 3, 7]) 
    for c in svrs[0]:
        for e in svrs[1]:
            clfs.append(SVR(C = c, epsilon = e, cache_size=1000, max_iter = 5000, tol = 1e-5))
            clf_names.append('SVR C={}, epsilon={}'.format(c,e))
            
    return clfs, clf_names


# In[ ]:


def ENets():
    clfs = []; clf_names = []
    enets = ([3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2  ], [ 0, 0.05, 0.1, 0.5, 0.9, 0.95, 0.98, 1]) 
    for a in enets[0]:
        for l in enets[1]:
            clfs.append(ElasticNet(alpha = a, l1_ratio = l,
                         normalize = False, selection = 'random', 
                         max_iter = 5000, tol = 1e-5))
            clf_names.append('Enet alpha={}, l1_ratio={}'.format(a,l))
 
    for a in enets[0]:
        for l in enets[1]:
            clfs.append(ElasticNet(alpha = a, l1_ratio = l,
                         normalize = True, selection = 'random', 
                         max_iter = 5000, tol = 1e-5))
            clf_names.append('Enet-n alpha={}, l1_ratio={}'.format(a,l))
            
    return clfs, clf_names


# In[ ]:





# In[ ]:


def getBaseClfs(y_var):
    idx = ALL_TARGETS.index(y_var)

    clfs = []
    clf_names = []
    
    model_sets =  [SVRs(), ENets(), Lassos(), Ridges()]
    for model_set in model_sets:
        clfs.extend(model_set[0])
        clf_names.extend(model_set[1])
   

    return clfs[::CLF_SS], clf_names[::CLF_SS];


# In[ ]:





# 

# In[ ]:


ALL_TARGETS = y.columns.to_list()  
if isinstance(TARGETS, list):
    targets = [ALL_TARGETS[i] for i in TARGETS]
elif TARGETS is not None and TARGETS >= 0:
    targets = ALL_TARGETS[TARGETS: TARGETS + 1]
else:
    targets = ALL_TARGETS
# print(targets)


# In[ ]:


def metaFilter(X):
    return X[[c for c in X.columns if c not in data.columns or c in loading.columns ]] 


# In[ ]:





# ### Train Models

# In[ ]:


all_clfs = []; all_raw_base_clfs = []; all_base_clfs = []; scalers = []
for idx, y_var in enumerate(targets):
    print('---Training Models for {}---\n'.format(y_var))
       
    
    # train base classifiers
    raw_base_clfs, base_clf_names = getBaseClfs(y_var)
    all_raw_base_clfs.append((raw_base_clfs, base_clf_names))
    
    base_clfs, base_clf_preds, base_clf_names, Xe, ye, ge =                     trainBaseClfs(raw_base_clfs, base_clf_names, 
                                  data = (X, y, groups), 
                                  target=y_var, )
    Xe = pd.concat( (Xe, pd.DataFrame( dict(zip(base_clf_names, base_clf_preds)), index=Xe.index) ),
                     axis = 'columns')
    
    all_base_clfs.append((base_clfs, base_clf_preds, base_clf_names, Xe, ye, ge ))
    
    
    # train meta model
    
    if y_var == 'age':
        s = FunctionTransformer()
        meta_model = trainNuSVR
    else:
        s = StandardScaler()
        meta_model = trainENet
     
    s.fit(metaFilter(Xe))
    scalers.append(s)
    
    all_clfs.append( runBag(data = (s.transform(metaFilter(Xe)), ye[y_var], ge), # target=y_var,
                                   model_type = meta_model) )
    # run


# In[ ]:





# ### Prediction Code

# In[ ]:


def predictBag(X, y, groups, clfs, target = None):
    start_time = datetime.datetime.now(); 

    valid = ~y.isnull()
    X = X[valid]; y = y[valid]; groups = groups[valid]
    
    if target is not None:
        X = cleanX(X, target)
    
    group_list = [*dict.fromkeys(groups)]   
    group_list.sort()

    preds = []; ys=[]; datestack = []
    for idx, group in enumerate(group_list):
        g = gc.collect()
        x_holdout = X[groups == group]
        y_holdout = y[groups == group]
  
        y_pred = clfs[idx].predict(x_holdout)    
        preds.append(y_pred)
        ys.append(y_holdout)
    
        print("{}: {:.4f}".format(group,
              fnae(y_holdout, y_pred) ) )
        
    y_pred = np.concatenate(preds)
    y_true = np.concatenate(ys) 
    
    print("\Bag Prediction Time: {}\n".format(str(datetime.datetime.now() - start_time).split('.', 2)[0] ))
    return y_pred, y_true


# In[ ]:


def predictAll(X_test, all_base_clfs, all_clfs):
    start_time = datetime.datetime.now(); 
        
    def predict_model(model, X):
        o = model.predict(X)
        return o    
    
    all_preds = pd.DataFrame(columns = targets, index=X_test.index)
    for tidx, y_var in enumerate(targets): # loop over targets
        print(y_var)
        Xi = cleanX(X_test, y_var)
        base_clfs = all_base_clfs[tidx][0]
         

        preds = []; 
        for g_idx, g_clfs in enumerate(base_clfs): # loop over groups
            print(g_idx)
            preds.append(Parallel(n_jobs=4)(delayed(predict_model)(model, Xi) for model in g_clfs))
        print("\Base Classifier Prediction Time: {}\n".format(str(datetime.datetime.now() - start_time).split('.', 2)[0] ))


        c_preds = []; sub_preds = np.zeros((len(preds), len(Xi)))
        for c_idx in range(0, len(preds[0])):  
            if len(preds[0][c_idx].shape) > 1: 
                for t_idx in range(0, preds[0][c_idx].shape[1]):
                    for g_idx, this_pred_group in enumerate(preds):  
                        sub_preds[g_idx, :] = this_pred_group[c_idx][:, t_idx]
                    c_preds.append(np.mean( sub_preds, axis = 0))  
            else:
                for g_idx, this_pred_group in enumerate(preds): 
                    sub_preds[g_idx, :] = this_pred_group[c_idx]
                c_preds.append(np.mean( sub_preds, axis = 0)) 

        Xf = pd.concat( (Xi, pd.DataFrame( dict(zip(all_base_clfs[tidx][2], c_preds)), index=Xi.index) ),
                     axis = 'columns')
        print("\nTime Elapsed: {}\n".format(str(datetime.datetime.now() - start_time).split('.', 2)[0] ))
 

        s = scalers[tidx]
        print('\nrunning stacker')
        pred = Parallel(n_jobs=4)(delayed(predict_model)(model, s.transform(metaFilter(Xf))) 
                                                       for model in all_clfs[tidx])
        sub_preds = np.zeros((len(all_clfs[tidx]), len(Xi)))
        for g_idx, clf in enumerate(all_clfs[tidx]):
            sub_preds[g_idx, :] = pred[g_idx]
        all_preds[y_var] = np.mean(sub_preds, axis = 0)


    end_time = datetime.datetime.now(); 
    print("\Prediction Time: {}\n".format(str(end_time - start_time).split('.', 2)[0] ))
    return all_preds, Xf


# In[ ]:





# ### Show Scores by Fold

# In[ ]:


y_preds = pd.DataFrame(index = X.index)
y_trues = y_preds.copy()
scores = pd.DataFrame(index = targets, columns = ['score'])
for idx, y_var in enumerate(targets):
    print(y_var)
    s = scalers[idx]
    y_pred, y_true =  predictBag(s.transform(metaFilter(all_base_clfs[idx][3])), 
                                 all_base_clfs[idx][4][y_var], all_base_clfs[idx][5], all_clfs[idx] ) 
    score = fnae(y_true, y_pred)
    print('{}: {:.4f}\n\n'.format(y_var, score))
    scores.loc[y_var] = score

scores.round(4) # MSCORE


# In[ ]:





# In[ ]:





# ### Show Overall Score

# In[ ]:


try:
    weights = pd.DataFrame( index = ALL_TARGETS, data = [.3, .175, .175, .175, .175] )
    overall_score = np.sum(scores * weights.values).iloc[0]
    age_score = np.mean(scores.iloc[:1]).iloc[0]
    other_scores = np.mean(scores.iloc[1:]).iloc[0]

    print(np.round(scores,4))
    print("\nOverall Score: {:.4f}".format(overall_score))

    print("   {:.4f}:  {:.4f} / {:.4f}   {}".format(overall_score, age_score, other_scores, 
                          [ np.round(s, 4) for s in scores.score] ))

except:
    pass


# In[ ]:





# ### Build Submission

# In[ ]:


y_oos, Xf = predictAll(test_data, all_base_clfs, all_clfs) 

y_oos = y_oos.reset_index().melt(id_vars = 'Id', value_name = 'Predicted')
y_oos.Id = y_oos.Id.astype(str) + '_' + y_oos.variable
y_oos.drop(columns = 'variable', inplace=True)

y_oos.to_csv('submission.csv', index=False)


# In[ ]:





# ### Show Final Submission

# In[ ]:


y_oos


# In[ ]:




