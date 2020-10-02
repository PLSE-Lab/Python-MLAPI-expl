#!/usr/bin/env python
# coding: utf-8

# # Numerai - prediction on encrypted stock 
# ## Data science tournament that powers the Numerai hedge fund
# 
# 
# ![](https://miro.medium.com/max/3304/1*jUPqj_1UoOCiKl4I85TiZw.png)
# 
# This kernel is made of helping get a start on the numerai tournament competiions.
# Official rules and data description on the numerai website [here](https://numer.ai/tournament).
# 
# 
# 
# <span style="color:red">Whats is Numerai???</span>
# 
# <img src="https://qph.fs.quoracdn.net/main-qimg-9cb24ba49c62c1ecda33c99bcee1d057.webp" width="800" height="500">
# 
# 
# 
# **Your task is to train a model to make predictions on the out-of-sample tournament_data. This dataset includes validation and test hold out sets, as well as live features of current stock market.**
# 
# Submissions scored by the following rubric:
# - measurements by rank_correlation between your predictions and the true targets
# - each day (for 4 weeks) submissions gets an updated correlation score
# - no scores on Sundays or Mondays
# - <span style="color:red">send submissions before Monday 14:30 UTC</span>
# 
# 
# **<span style="color:red">You can stake your model to start earning daily payouts but the kaggler is not responsible for any results from this kernel.</span>**
# 
# **Usefull sources: **
# 
# [**<span style="color:purple">NUMER.AI </span>**](https://numer.ai)
# 
# [**<span style="color:blue">EXCHANGE BTC TO NMR </span>**](https://greencoin.online/exchange/BTC-NMR/)
# 
# [**<span style="color:purple">BTC and OTHER </span>**](https://coinatmradar.com/bitcoin-atm-near-me/)
# 
# [**<span style="color:purple">ERASUREQUANT </span>**](https://erasurequant.com/)
# 

# ### Manage the datasets with <span style="color:red">pd.HDFStore</span>

# In[ ]:


get_ipython().run_cell_magic('time', '', 'import pandas as pd \nimport numpy as np\nimport gc; gc.enable()\n        \nimport pathlib as pt\n\nDATA_STORE = \'/kaggle/working/data_kaz.h5\'\nPATH_TRAIN = \'/kaggle/input/numerai-round-190/numerai_training_data_t.csv\'\nPATH_TEST = \'/kaggle/input/numerai-round-190/numerai_tournament_data.csv\'\nTARGET = \'/kaggle/input/numerai-round-190/target.csv\'\nERAS = \'/kaggle/input/numerai-round-190/eras_t.csv\'\n\ntry:\n    \n    path = pt.Path(DATA_STORE)\n    \n    with pd.HDFStore(path) as store: \n        train = store[\'train\']; tournament = store[\'tournament\'];\n        target = store[\'TARGET\']; eras = store[\'ERAS\'];\n    print(\'[INFO] read data...\')\n\nexcept (ValueError, KeyError):\n    \n    path = pt.Path(DATA_STORE)\n    \n    print(\'[INFO] create data storage...\')\n    with pd.HDFStore(path) as store: store.put(\'train\',pd.read_csv(PATH_TRAIN, header=0)\\\n                                        .apply(lambda x: \n                                                           x.astype(np.float16).round(2) if x.dtype==np.float64\n                                        else x),\n                                               compression=\'gzip\', complevel=20, format = \'table\')\n        \n    with pd.HDFStore(path) as store: store.put(\'tournament\',pd.read_csv(PATH_TEST, header=0).iloc[0:500000]\\\n                                        .apply(lambda x: \n                                                           x.astype(np.float16).round(2) if x.dtype==np.float64\n                                        else x),\n                                               compression=\'gzip\',complevel=20, format = \'table\')\n        \n    with pd.HDFStore(path) as store: store.put(\'tournament\',pd.read_csv(PATH_TEST, header=0).iloc[500001:1000000]\\\n                                        .apply(lambda x: \n                                                           x.astype(np.float16).round(2) if x.dtype==np.float64\n                                        else x),\n                                               compression=\'gzip\',complevel=20, append=True, format = \'table\')\n        \n    with pd.HDFStore(path) as store: store.put(\'tournament\',pd.read_csv(PATH_TEST, header=0).iloc[1000001:]\\\n                                        .apply(lambda x: \n                                                           x.astype(np.float16).round(2) if x.dtype==np.float64\n                                        else x),\n                                               compression=\'gzip\',complevel=20, append=True, format = \'table\')\n        \n    with pd.HDFStore(path) as store:\n        store.put(\'TARGET\',pd.read_csv(TARGET, header=None), format = \'table\')\n        store.put(\'ERAS\',pd.read_csv(ERAS, header=None), format = \'table\')\n\n    print(\'[INFO] read data...\')\n    with pd.HDFStore(path) as store: \n        train = store[\'train\'];tournament = store[\'tournament\'];\n        target = store[\'TARGET\'];eras = store[\'ERAS\'];\n\nnumerai_benchmark = pd.read_csv(\'/kaggle/input/numerai-round-190/example_predictions_target_kazutsugi.csv\')\n\n# preparing train and val data\nvalidation = tournament[tournament[\'data_type\'] == \'validation\']\ntarget_val = validation[\'target_kazutsugi\']\n\nerasv = validation.era.str.slice(3).astype(int)\nerast =  tournament[tournament[\'data_type\'] == \'test\'].era.str.slice(3).astype(int)\n\n# Transform the loaded CSV data into numpy arrays\nfeatures = [f for f in list(train) if "feature" in f]\n\ntrain = train[features]\nval = validation[features]\n\ntest = tournament[features]\nids = tournament[\'id\']\n\ndel tournament, validation\ngc.collect()\n\ntrain[\'era\']=eras\nval[\'era\']=erasv\n\nprint(train.shape, test.shape, val.shape)\nprint(\'loading datasets has done\')')


# #### Some helping functions 

# In[ ]:


from gplearn.functions import make_function
from gplearn.genetic import SymbolicClassifier
def th(x):
    return np.tanh(x)

gptanh = make_function(th, 'tanh', 1)
sample_wts = np.sqrt(np.array([x - 10.0 if x > 10.0 else 0 for x in target.values]) + 1.0)
function_set = ['add', 'sub', 'mul', 'div', 'inv', 'abs', 'neg', 'max', 'min', gptanh]
count = 1
est = SymbolicRegressor(population_size=2000,
                       generations=count,
                       tournament_size=50,  # consider 20, was 50
                       parsimony_coefficient=0.0001,  # oops: 0.0001?
                       function_set=function_set,init_depth=(6, 16),
                       metric='mean absolute error', verbose=1, random_state=42, n_jobs=-1, low_memory=True)
est.fit(train[features], target)
est.predict(val[features])


# In[ ]:


from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error

GENS = 500
MAE_THRESH = 2.5
MAX_NO_IMPROVE = 50
np.random.seed(666)
maes = []
gens = []
from gplearn.functions import make_function

def th(x):
    return np.tanh(x)

gptanh = make_function(th, 'tanh', 1)
sample_wts = np.sqrt(np.array([x - 10.0 if x > 10.0 else 0 for x in target.values]) + 1.0)
function_set = ['add', 'sub', 'mul', 'div', 'inv', 'abs', 'neg', 'max', 'min', gptanh]
folds = KFold(n_splits=3, shuffle=True, random_state=42)

for fold_, (trn_idx, val_idx) in enumerate(folds.split(train[features], target.values)):
    print('working fold %d' % fold_)
    X_tr, X_val = train[features].iloc[trn_idx], train[features].iloc[val_idx]
    y_tr, y_val = target.values[trn_idx].ravel(), target.values[val_idx].ravel()
    sample_wts_tr = sample_wts[trn_idx]
    np.random.seed(5591 + fold_)
    best = 1e10
    count = 1
    imp_count = 0
    best_mdl = None
    best_iter = 0
  
    gp = SymbolicRegressor(population_size=2000,
                       generations=count,
                       tournament_size=50,  # consider 20, was 50
                       parsimony_coefficient=0.0001,  # oops: 0.0001?
                       const_range=(-16, 16),  # consider +/-20, was 100
                       function_set=function_set,
                       # stopping_criteria=1.0,
                       # p_hoist_mutation=0.05,
                       # max_samples=.875,  # was in
                       # p_crossover=0.7,
                       # p_subtree_mutation=0.1,
                       # p_point_mutation=0.1,
                       init_depth=(6, 16),
                       warm_start=True,
                       metric='mean absolute error', verbose=1, random_state=42, n_jobs=-1, low_memory=True)

    for run in range(GENS):
        mdl = gp.fit(X_tr, y_tr, sample_weight=sample_wts_tr)
        pred = gp.predict(X_val)
        mae = np.sqrt(mean_absolute_error(y_val, pred))

    if mae < best and imp_count < MAX_NO_IMPROVE:
        best = mae
        count += 1
        gp.set_params(generations=count, warm_start=True)
        imp_count = 0
        best_iter = run
        if mae < MAE_THRESH:
            best_mdl = copy.deepcopy(mdl)
    elif imp_count < MAX_NO_IMPROVE:
        count += 1
        gp.set_params(generations=count, warm_start=True)
        imp_count += 1
    else:
        break

    print('GP MAE: %.4f, Run: %d, Best Run: %d, Fold: %d' % (mae, run, best_iter, fold_))

maes.append(best)
gens.append(run)
      
print('Finish - GP MAE: %.4f, Run: %d, Best Run: %d' % (mae, run, best_iter))

preds = best_mdl.predict(val[features])
print(preds[0:12])
predictions += preds / folds.n_splits


# In[ ]:


get_ipython().run_cell_magic('time', '', '\nBENCHMARK = 0.002\nBAND = 0.04\n\nTOURNAMENT_NAME = "kazutsugi"\nPREDICTION_NAME = f"prediction_{TOURNAMENT_NAME}"\nTARGET_NAME = f"target_{TOURNAMENT_NAME}"\ntrain[TARGET_NAME]=target.values\nval[\'target_kazutsugi\'] = target_val.values\n\n# The payout function\ndef payout(scores):\n    return ((scores - BENCHMARK)/BAND).clip(lower=-1, upper=1)\n\ndef score(df):\n    # method="first" breaks ties based on order in array\n    return np.corrcoef(df[\'target_kazutsugi\'], df[PREDICTION_NAME].rank(pct=True, method="first"))[0,1]\n\ndef evaluation_test(train,val):\n    # Check the per-era correlations on the training set\n    train_correlations = train.groupby(\'era\').apply(score)\n    print(f"On training the correlation has mean {train_correlations.mean()} and std {train_correlations.std()}")\n    print(f"On training the average per-era payout is {payout(train_correlations).mean()}")\n\n    # Check the per-era correlations on the validation set\n    validation_correlations = val.groupby(\'era\').apply(score)\n    print(f"On validation the correlation has mean {validation_correlations.mean()} and std {validation_correlations.std()}")\n    print(f"On validation the average per-era payout is {payout(validation_correlations).mean()}")\n    \ndef create_sub(results, ids):\n    import time\n    from time import gmtime, strftime\n    results_df = pd.DataFrame(data={\'probability_kazutsugi\': results})\n    strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime())\n    joined = pd.DataFrame(ids).join(results_df)\n    path = \'predictions_{:}\'.format(strftime("%Y-%m-%d_%Hh%Mm%Ss", time.gmtime())) + \'.csv\'\n    print("Writing predictions to " + path.strip())\n    joined.to_csv(path, float_format=\'%.15f\', index=False)\n    \n# multy threading module  \'------------------------------------------------------------------------------------------------------------------\'\nfrom multiprocessing.pool import ThreadPool\nfrom functools import partial\n\nclass parallelization:\n    \n    @staticmethod\n    def map_parallel(fn, lst):\n        with ThreadPool(processes=3) as pool:\n            return pool.map(fn, lst)\n\n    @staticmethod\n    def compute_(est, X, y=None):\n        if y is not None:\n            \n            return est.fit(X, y)\n        else:\n            return est.predict(X)\n\n    def run_compile(self, models, X_tr, y_tr=None):\n        if y_tr is None:\n            return self.map_parallel(partial(self.compute_, X=X_tr), models)\n        else:\n            return self.map_parallel(partial(self.compute_, X=X_tr, y=y_tr), models)')


# ### Simple multithreading stack regression models

# In[ ]:


get_ipython().run_cell_magic('time', '', '\nimport os\nos.environ[\'TF_CPP_MIN_LOG_LEVEL\'] = \'2\'\nos.environ[\'OMP_NUM_THREADS\'] = \'1\'\n\nfrom sklearn.pipeline import Pipeline\nfrom sklearn.ensemble import VotingRegressor\nfrom lightgbm import LGBMRegressor\nfrom sklearn.linear_model import Ridge, ElasticNet, Lasso\nimport warnings\nwarnings.filterwarnings(\'ignore\')\n\nfrom sklearn.decomposition import PCA \npca = PCA(n_components=.95)\npca.fit(train[features])\n\nreg1 = LGBMRegressor(n_jobs=1, colsample_bytree=0.1, learning_rate=0.06, \n                     n_estimators=600, max_depth=4,subsample=0.82,\n                     nthread=5)\n\nreg2 = LGBMRegressor(n_jobs=1, colsample_bytree=0.1, learning_rate=0.06, \n                     n_estimators=100, max_depth=5,subsample=0.82,\n                     nthread=5)\n\nreg3 = LGBMRegressor(n_jobs=1, colsample_bytree=0.1, learning_rate=0.07, \n                    n_estimators=400, num_leaves = 65,subsample=0.82,\n                    nthread=5)\n\nreg4 = ElasticNet(alpha=1e-05)\n\nreg5 = LGBMRegressor(n_jobs=1, colsample_bytree=0.1, learning_rate=0.05, \n                    n_estimators=400, max_depth=7,subsample=0.82,\n                    nthread=5)\n\nreg6 = LGBMRegressor(n_jobs=1, colsample_bytree=0.1, learning_rate=0.002, \n                    n_estimators=200, max_depth=7,subsample=0.82,\n                    nthread=5)\n\nmodels_list = [reg1,reg2,reg3,reg4,reg5,reg6]\n\nmodels_input = VotingRegressor((\n    (\'reg1\',reg1), (\'reg2\',reg2), (\'reg3\',reg3),\n    (\'reg4\',reg4), (\'reg5\',reg5), (\'reg6\',reg6)\n))\n\nmodel_reg = Pipeline([(\'reg\', models_input)])\n\nparalle = parallelization()\n\nfitedmodels = paralle.run_compile(model_reg,X_tr=pca.transform(train[features]), y_tr=target.values.ravel())\n# predict\npred_train_ = paralle.run_compile(fitedmodels,pca.transform(train[features]))[0]\npred_val_ = paralle.run_compile(fitedmodels,pca.transform(val[features]))[0]\npred_test_ = paralle.run_compile(fitedmodels,pca.transform(test[features]))[0]\n\n# FIRST LEVEL  \'--------------------------------------------------------------------------------------------------------------------\'\nfitedmodels = paralle.run_compile(models_list,pca.transform(train[features]), target.values.ravel())\nsx_train_ = paralle.run_compile(fitedmodels,pca.transform(train[features]))\nsx_test_ = paralle.run_compile(fitedmodels,pca.transform(test[features]))\nsx_val_ = paralle.run_compile(fitedmodels,pca.transform(val[features]))\n\nsx_train_ = np.vstack(sx_train_).T\nsx_test_ = np.vstack(sx_test_).T\nsx_val_ = np.vstack(sx_val_).T\n\n# SECOND LEVEL \'--------------------------------------------------------------------------------------------------------------------\'\nreg = LGBMRegressor(n_jobs=-1, colsample_bytree=0.1, learning_rate=0.01, \n                     n_estimators=400, max_depth=7,subsample=0.82,\n                     nthread=5)\n\nreg.fit(sx_train_,target.values.ravel())\n    \n# FINAL PREDCTION \'-----------------------------------------------------------------------------------------------------------------\'\ntrain[PREDICTION_NAME] = (reg.predict(sx_train_)*.1+pred_train_*.9)\nval[PREDICTION_NAME] = (reg.predict(sx_val_)*.1+pred_val_*.9)\ntest[PREDICTION_NAME] = (reg.predict(sx_test_)*.1+pred_test_*.9)\n\ngc.collect()\nevaluation_test(train,val)\n# save submission file\nssubm = pd.read_csv("../input/numerai-round-190/example_predictions_target_kazutsugi.csv")\ncreate_sub(test[PREDICTION_NAME], ssubm.id)')


# In[ ]:


function_set = ['add', 'sub', 'mul', 'div',
                'sqrt', 'log', 'abs', 'neg', 'inv',
                'max', 'min']
gp = SymbolicTransformer(generations=20, population_size=2000,
                         hall_of_fame=100, n_components=10,
                         function_set=function_set,
                         parsimony_coefficient=0.0005,
                         max_samples=0.9, verbose=1,
                         random_state=0, n_jobs=3)


# In[ ]:


import gplearn


# In[ ]:


est = SymbolicClassifier(parsimony_coefficient=.01,
                         feature_names=cancer.feature_names,
                         random_state=1)
est.fit(cancer.data[:400], cancer.target[:400])

