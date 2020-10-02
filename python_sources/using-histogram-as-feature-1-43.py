#!/usr/bin/env python
# coding: utf-8

# Scripts and discussions have pointed out the fact that value was as important as features if not more. All this originated from Giba's **magic** script and post:
#  - [Script](https://www.kaggle.com/titericz/giba-countvectorizer-d-lb-1-43)
#  - [Post](https://www.kaggle.com/c/santander-value-prediction-challenge/discussion/61071)
# 
# Here is a very simple and wuick script that would use the samples histogram as features.
# 
# The script uses np.apply_along_axis that is lot quicker than pd.apply !

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from lightgbm import LGBMRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error


# Read data?

# In[ ]:


get_ipython().run_cell_magic('time', '', "data = pd.read_csv('../input/train.csv')\ntest = pd.read_csv('../input/test.csv')")


# Transform samples to histogram
# 
# It may sound like bins are created independently for each row when in fact not.
# 
# np.bincount  will create all integer bins up to the max value in the list, i.e. if max value is 5 it will create bins 0, 1, 2, 3, 4 and 5 columns. 
# 
# I set the number of bins to 30 to make sure all returned bins are contain data for 0s up to 29
# 
# 30 is above the max value in data and test. 
# 
# So although the bin process is independent, the resulting columns are the same across rows.
# 
# doc is [here](https://docs.scipy.org/doc/numpy/reference/generated/numpy.bincount.html)

# In[ ]:


get_ipython().run_cell_magic('time', '', "def to_hist_func(row):\n    return np.bincount(row, minlength=30)\n\nfeatures = [f for f in data.columns if f not in ['target', 'ID']]\n\nhist_data = np.apply_along_axis(\n    func1d=to_hist_func, \n    axis=1, \n    arr=(np.log1p(data[features])).astype(int)) ")


# In[ ]:


get_ipython().run_cell_magic('time', '', 'hist_test = np.apply_along_axis(\n    func1d=to_hist_func, \n    axis=1, \n    arr=(np.log1p(test[features])).astype(int)) ')


# Let's try to fit a model on this 

# In[ ]:


folds = KFold(n_splits=5, shuffle=True, random_state=1)
oof_preds = np.zeros(data.shape[0])
sub_preds = np.zeros(test.shape[0])

for n_fold, (trn_, val_) in enumerate(folds.split(hist_data)):
    reg = ExtraTreesRegressor(
        n_estimators=1000, 
        max_features=.8,                       
        max_depth=12, 
        min_samples_leaf=10, 
        random_state=3, 
        n_jobs=-1
    )
    # Fit Extra Trees
    reg.fit(hist_data[trn_], np.log1p(data['target'].iloc[trn_]))
    # Get OOF predictions
    oof_preds[val_] = reg.predict(hist_data[val_])
    # Update TEST predictions
    sub_preds += reg.predict(hist_test) / folds.n_splits
    # Display fold's score
    print('Fold %d scores : TRN %.4f TST %.4f'
          % (n_fold + 1,
             mean_squared_error(np.log1p(data['target'].iloc[trn_]),
                                reg.predict(hist_data[trn_])) ** .5,
             mean_squared_error(np.log1p(data['target'].iloc[val_]),
                                reg.predict(hist_data[val_])) ** .5))
          
print('Full OOF score : %.4f' % (mean_squared_error(np.log1p(data['target']), oof_preds) ** .5))


# In[ ]:


test['target'] = np.expm1(sub_preds)
test[['ID', 'target']].to_csv('histogram_predictions.csv', index=False)


# In[ ]:




