#!/usr/bin/env python
# coding: utf-8

# 

# In[ ]:


# Standard-ish includes
import kagglegym
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import xgboost as xgb
import math

import time

#from sklearn.preprocessing import PolynomialFeatures
#from sklearn.pipeline import make_pipeline
#from sklearn.linear_model import Ridge

from operator import itemgetter


# In[ ]:


localrun = False     # keep data structures that (might) get freed to save memory
usepublic = False    # Load the second half of training data, to get a better idea 
vmode = False        # Trains on whole data, exports a file with validation predictions 
all_features = False # Overrides feature list, for feature selection


# In[ ]:


# This is taken from Frans Slothoubers post on the contest discussion forum.
# https://www.kaggle.com/slothouber/two-sigma-financial-modeling/kagglegym-emulation

def r_score(y_true, y_pred, sample_weight=None, multioutput=None):
    r2 = r2_score(y_true, y_pred, sample_weight=sample_weight,
                  multioutput=multioutput)
    r = (np.sign(r2)*np.sqrt(np.abs(r2)))
    if r <= -1:
        return -1
    else:
        return r

# From the xgboost script (along with the param settings)
# https://www.kaggle.com/jacquespeeters/two-sigma-financial-modeling/xgboost-0-007-lb
    
# Functions for XGBOOST ########################################################
def xgb_obj_custom_r(y_pred, dtrain):
    y_true = dtrain.get_label()
    y_mean = np.mean(y_true)
    y_median = np.median(y_true)
    c1 = y_true
    #c1 = y_true - y_mean
    #c1 = y_true - y_median
    grad = 2*(y_pred-y_true)/(c1**2)
    hess = 2/(c1**2)
    return grad, hess

def xgb_eval_custom_r(y_pred, dtrain):
    #y_pred = np.clip(y_pred, -0.075, .075)
#    y_pred[y_pred > .075] = .075
#    y_pred[y_pred < -.075] = -.075
    y_true = dtrain.get_label()
    ybar = np.sum(y_true)/len(y_true)
    ssres = np.sum((y_true - y_pred) ** 2)
    sstot = np.sum((y_true - ybar)**2)
    r2 = 1 - ssres/sstot
    error = np.sign(r2) * np.absolute(r2)**0.5
    return 'error', error

env = kagglegym.make()
o = env.reset()

# Kagglegym emulator didn't provide these:
# excl = [env.ID_COL_NAME, env.SAMPLE_COL_NAME, env.TARGET_COL_NAME, env.TIME_COL_NAME]
excl = ['id', 'sample', 'y', 'timestamp']
basecols = [c for c in o.train.columns if c not in excl]


# In[ ]:


# Reduced columns
rcol_orig = ['Dtechnical_20', 'y_prev_pred_avg_diff', 'Dtechnical_21', 'technical_43_prev', 'technical_20', 'y_prev_pred_avgT0', 'y_prev_pred_mavg5', 'y_prev_pred_avgT1', 'fundamental_8_prev', 'Dtechnical_40', 'technical_7_prev', 'technical_7', 'fundamental_5', 'Dtechnical_30', 'technical_32_prev', 'technical_14_prev', 'fundamental_1', 'fundamental_43_prev', 'Dfundamental_22', 'Dtechnical_35', 'Dtechnical_6', 'Dtechnical_17', 'Dtechnical_27', 'Dfundamental_42', 'fundamental_1_prev', 'Dtechnical_0', 'technical_40', 'technical_40_prev', 'fundamental_36', 'Dfundamental_33', 'Dfundamental_48', 'technical_27_prev', 'fundamental_62_prev', 'fundamental_41_prev', 'Dfundamental_50', 'fundamental_48', 'derived_2_prev', 'Dtechnical_18', 'fundamental_35', 'Dfundamental_49', 'fundamental_26_prev', 'technical_28_prev', 'Dfundamental_63', 'fundamental_10_prev', 'fundamental_36_prev', 'fundamental_16', 'Dfundamental_8', 'fundamental_32', 'fundamental_40_prev', 'derived_0', 'Dfundamental_32', 'fundamental_17', 'Dtechnical_7', 'fundamental_25', 'technical_35', 'Dtechnical_19', 'technical_35_prev', 'fundamental_8', 'Dtechnical_32', 'Dfundamental_18', 'Dtechnical_37', 'fundamental_33_prev', 'Dtechnical_28', 'fundamental_46', 'Dfundamental_1', 'Dfundamental_45', 'fundamental_18', 'technical_12', 'technical_44', 'fundamental_22', 'Dtechnical_5', 'technical_17_prev', 'Dfundamental_25']
rcol = rcol_orig.copy()

if all_features:
    rcol = []
    for c in basecols:
        rcol.append(c)
        rcol.append(c + '_prev')
        rcol.append('D' + c)

# Features used to compute previous Y value
backy_fset = ['technical_13', 'technical_20', 'technical_13_prev', 'technical_20_prev', 'technical_30_prev', 'technical_30']
for f in backy_fset:
    if f not in rcol:
        rcol.append(f)
        
# convert column name to base column
def get_basecol(r):
    corename = r[1:] if r[0] == 'D' else r[0:]
    corename = corename.split('_')
    corename = corename[0] + '_' + corename[1]

    return corename

def get_basecols(rcol):
    duse = {}

    for r in rcol:
        if 'y' in r:
            continue
            
        duse[get_basecol(r)] = True
        
    return [k for k in duse.keys()]

basecols_touse = get_basecols(rcol)


# In[ ]:





# 

# In[ ]:





# In[ ]:





# 

# In[ ]:





# 

# In[ ]:





# 

# In[ ]:


if vmode:
    preds_xgb = model[0].predict(valid_xgb, ntree_limit=model[0].best_ntree_limit)
    preds_linear = model2.predict(prep_linear(xvalid))
    
    preds = (preds_xgb * 0.7) + (preds_linear * 0.3)
    #preds = preds_xgb
    
    rs = kagglegym.r_score(xvalid.y, preds)
    
    ID = 'expv-{0}.pkl'.format(int(rs * 10000000))
    print(rs, ID)
    
    #ID = 'subv-203172.pkl' # if actual submission
    
    output = xvalid[['id', 'timestamp', 'y']].copy()
    output['y_hat'] = preds
    output['y_hat_xgb'] = preds_xgb
    output['y_hat_linear'] = preds_linear
    
    output.to_pickle(ID)

# Not the final version I used, this just shows which features actually got picked up at all...
if all_features:
    m = model[0]

    fs = m.get_fscore()
    fsl = [(f,fs[f]) for f in fs.keys()]
    fsl = sorted(fsl, key=itemgetter(1), reverse=True)

    print('rcol =', [f[0] for f in fsl])


# The code below is meant to be run with local/usepublic/allfeatures all set to True.  It then multiplies normalized train+test and keeps everything above 0.  
# 
# It's inspired by the xgboost refresh updater kernel: https://www.kaggle.com/tks0123456789/two-sigma-financial-modeling/xgboost-refresh-updater-oob-feature-importance

# In[ ]:


def update_model(m, params_in, cols_to_use):

    params = params_in.copy()
    
    params.update({'process_type': 'update',
                       'updater'     : 'refresh',
                       'refresh_leaf': False})

    m_train = xgb.train(params, train_xgb, m.best_ntree_limit, xgb_model=m)
    m_test = xgb.train(params, public_xgb, m.best_ntree_limit, xgb_model=m)

    imp = pd.DataFrame(index=cols_to_use)
    imp['train'] = pd.Series(m_train.get_score(importance_type='gain'), index=cols_to_use)
#    imp['valid'] = pd.Series(m_valid.get_score(importance_type='gain'), index=cols_to_use)
    imp['test'] = pd.Series(m_test.get_score(importance_type='gain'), index=cols_to_use)

    imp = imp.fillna(0)
    
    return m_train, m_test, imp

m_train, m_test, imp_orig = update_model(model[0], params, cols_to_use)

imp = imp_orig.copy()

imp.train = imp.train / imp.train.max()
imp.test = imp.test / imp.test.max()

imp['train_test'] = imp.train * imp.test

imp = imp.sort_values('train_test', ascending=False)
impr = imp[imp.train_test > 0]

print(list(impr.index))

# features output from above
# ['Dtechnical_20', 'technical_7', 'technical_20', 'technical_7_prev', 'Dtechnical_21', 'fundamental_1', 'Dtechnical_40', 'fundamental_36', 'technical_14_prev', 'Dtechnical_30', 'fundamental_1_prev', 'fundamental_5', 'technical_35', 'Dtechnical_35', 'technical_35_prev', 'Dtechnical_19', 'technical_40', 'Dfundamental_49', 'fundamental_41_prev', 'fundamental_36_prev', 'technical_40_prev', 'fundamental_8', 'Dfundamental_48', 'Dtechnical_0', 'derived_2_prev', 'fundamental_46', 'Dfundamental_22', 'fundamental_43_prev', 'Dtechnical_17', 'Dfundamental_33', 'Dfundamental_50', 'technical_43_prev', 'Dfundamental_63', 'technical_27_prev', 'technical_17_prev', 'Dtechnical_18', 'fundamental_10_prev', 'Dtechnical_27', 'Dfundamental_18', 'fundamental_25', 'fundamental_18', 'Dfundamental_8', 'fundamental_35', 'fundamental_8_prev', 'technical_12', 'technical_32_prev', 'fundamental_17', 'Dtechnical_7', 'fundamental_26_prev', 'fundamental_33_prev', 'Dtechnical_32', 'derived_0', 'Dfundamental_32', 'Dfundamental_42', 'fundamental_16', 'fundamental_32', 'Dtechnical_28', 'technical_28_prev', 'fundamental_40_prev', 'fundamental_62_prev', 'technical_44', 'Dfundamental_25', 'fundamental_48']


# 

# In[ ]:




