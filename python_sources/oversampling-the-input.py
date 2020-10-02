#!/usr/bin/env python
# coding: utf-8

# ## Oversampling Attempt 
# This is another attempt at trying different techniques that might help in this competition. 
# 
# I have oversampled the positive cases by 20k and then merged the oversampled cases with the original dataset. (Technique - SMOTE)
# 
# One particular thing that is happening with oversampling the positive cases is the model is overfitting drastically. 
# 
# For e.g. 
# * Lightgbm on normal dataset (no FE) give 90% .
# * But when you oversample the dataset with postive cases (even small oversampling ) causes big overfitting. 
# 
# The validation dataset is next to useless when oversampling the training cases, as it will never depict the test scenario. So I was not too reliant on validation dataset. 
# 
# But with 20% over sampling. 
# * AUC for training data - 100%
# * AUC for validation data - 96%
# * AUC for test data - 84%
# 
# Now what intrigues me is that it is difficult to overfit lightgbm model for this dataset. But with adding little impurities the model starts to overfit immediately. 
# My hypothesis is that the oversampled cases (produced through SMOTE (I guess using nn approach)) are linked with the input in such a way that it helps classify the training data. But since we cannot create the synthetic data for the test cases it underperforms. 
# 
# The code might be little assorted but I will try to put comments where ever necessary. I hope it might be helpful to some, who may make some sense out of it and probably help improve their LB
# 

# ### Kaggle wrote these lines for me :)

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# ### Loading the libraries 

# In[ ]:


import os
import shutil
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from datetime import datetime
pd.set_option('display.max_columns', 200)


# In[ ]:


import numpy as np
import pandas as pd
import sklearn
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report,accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from imblearn.combine import SMOTEENN,SMOTETomek
from sklearn.svm import OneClassSVM
from pylab import rcParams
rcParams['figure.figsize'] = 14, 8
RANDOM_SEED = 42
LABELS = ["Nobuy", "Buy"]
import plotly.plotly as py
import plotly.graph_objs as go
import plotly
import plotly.figure_factory as ff
from plotly.offline import init_notebook_mode, iplot


# ### Reading the input

# In[ ]:


print("Reading training data")
data = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# ### Extremely useful display function, that is borrowed from fast.ai modules, it helps analyse the data in full. Thanks to @jeremyhoward

# In[ ]:


def display_all(df):
    with pd.option_context("display.max_rows", 1000, "display.max_columns", 1000): 
        display(df)


# ### Removing the ID_code variable as I will not be using this in training and prediction

# In[ ]:


data = data.drop(['ID_code'],axis=1)
test_ids = test.ID_code.values
test = test.drop(['ID_code'],axis=1)


# ### This is the Crux of the code. 
# 
# I have imported the SMOTE function to oversample my target variable for positive cases. 
# I am taking 1/4 of non-positive cases , i.e. 40k and 20k postive cases. After oversampling I am expecting 40k nonpositive and 40k positive cases
# 
# Meaning I have introduced 20k addition positive datapoints just in hope that I am able to classify the positive cases better. 
# 
# The resampled cases are stored in X_resampled, y_resampled. 

# In[ ]:


get_ipython().run_cell_magic('time', '', 'from imblearn.over_sampling import SMOTE\nsub_non_fraud = data.loc[data.loc[:, \'target\'] == 0, :].sample(int(len(data.loc[:, \'target\'])/4 ))\n#sub_non_fraud = data.loc[data.loc[:, \'target\'] == 0, :]\ndata_resample = pd.concat([sub_non_fraud, data.loc[data.loc[:, \'target\'] == 1, :]])\nX = data_resample.drop([\'target\'],axis=1)\ny = data_resample.loc[:, "target"]\nsm = SMOTETomek()\nX_resampled, y_resampled = sm.fit_sample(X, y)')


# ### Verifiying whether the oversampling has happened correctly by checking the size

# In[ ]:


# size of X and y after SMOTE
print("Size of X", X_resampled.shape)
print("Size of y", y_resampled.shape)
print("Size of positive cases", y_resampled[y_resampled == 1].shape)


# Changing the resampled cases into pandas dataframe

# In[ ]:


from sklearn.model_selection import train_test_split
X_resampled = pd.DataFrame(X_resampled)
y_resampled = pd.DataFrame(y_resampled)


# ### Validating duplicates in resampled cases
# 
# No duplicate cases are created 

# In[ ]:


duplicateRowsDF = X_resampled[X_resampled.duplicated()]
duplicateRowsDF.shape


# ### Merging with the original dataset. 
# I would be merging my resampled dataset with the original dataset. creating 220k training + validation cases with 40k positive cases and 160k negative cases. 
# 
# To perform merge I am changing the variable name of the resampled cases so that merging happens in axis=0 

# In[ ]:


X_resampled = X_resampled.add_prefix('var_')


# ### Merging with original dataset
# 
# Some naive variable assignment in tem_var as I was doing lot of experimentation (negelect it)

# In[ ]:


temp_xvar = X_resampled
temp_yvar = y_resampled
X_resampled1 = pd.concat([data.drop(['target'],axis=1), temp_xvar])
y_resampled1 = pd.concat([data['target'],temp_yvar])


# In[ ]:


X_resampled1.shape


# ### Removing dupicates 
# 
# My resmapled dataset had 80k cases. and original dataset had 200k cases. 60k cases are duplicate that needs to be removed. 
# 
# Some non pythonic code :) to perform this deletion 

# In[ ]:


data1 = X_resampled1
data1['target'] = y_resampled1


# In[ ]:


data1.shape


# In[ ]:


data1.drop_duplicates(keep = 'first', inplace = True) 


# In[ ]:


data1.shape


# ### Resetting the index 

# In[ ]:


data1.reset_index(drop=True,inplace=True)


# data1.shape

# In[ ]:


del X_resampled
del y_resampled


# In[ ]:


X_resampled = data1.drop(['target'],axis=1)
y_resampled = data1['target']


# In[ ]:


X = X_resampled.values.astype(float)
y = np.array(y_resampled)
X_test = test.values.astype(float)


# In[ ]:



y = y.flatten()


# ### Training with kfold 5. 
# The below code has been taken from @abhishekthakur's framework for competition. 
# I guess everyone will be familiar with the code, but its the output that is interesting

# In[ ]:


import os
import shutil
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
import math
import gc
from sklearn.preprocessing import MinMaxScaler,StandardScaler
NFOLDS = 5
RANDOM_STATE = 42


# In[ ]:


import lightgbm as lgb
param = {
    'bagging_freq': 5,
    'bagging_fraction': 0.335,
    'boost_from_average':'false',
    'boost': 'gbdt',
    'feature_fraction': 0.041,
    'learning_rate': 0.1,
    'max_depth': 7,
    'metric':'auc',
    'min_data_in_leaf': 80,
    'min_sum_hessian_in_leaf': 10.0,
    'num_leaves': 13,
    'num_threads': 8,
    'tree_learner': 'serial',
    'objective': 'binary', 
    'verbosity': -1
#    'device': 'gpu',
#    'gpu_platform_id': 0,
 #   'gpu_device_id': 0
}


# In[ ]:


clfs = []
folds = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=RANDOM_STATE)
oof_preds = np.zeros((len(X_resampled), 1))
oof_preds_lgb = np.zeros((len(X_resampled), 1))
test_preds = np.zeros((len(test), 1))
test_preds_lgb = np.zeros((len(test), 1))
#del train, test
gc.collect()


# In[ ]:


val1_x= data.drop(['target'],axis=1).values.astype(float)
val1_y=data['target']


# ### Overfitting the training data. 
# 
# I am unable to understand why. 
# 
# I know it will overfit but so quickly and problem is that the validation set auc also keeps on improving. 
# 
# With lower learning rate and high iterations the kernel will run for day. Not ideal for person reliant on kaggle kernels for compute power :)

# In[ ]:


c = 0
for fold_, (trn_, val_) in enumerate(folds.split(y, y)):
    print("Current Fold: {}".format(fold_))
    trn_x, trn_y = X[trn_, :], y[trn_]
    val_x, val_y = X[val_, :], y[val_]
#    X_tr, y_tr = augment(trn_x, trn_y)
#    X_tr = pd.DataFrame(X_tr)
    trn_data = lgb.Dataset(trn_x, label=trn_y)
    val_data = lgb.Dataset(val_x, label=val_y)
    val1_data = lgb.Dataset(val1_x,label=val1_y)
    
    clf = lgb.train(param, trn_data, 100000, valid_sets = [trn_data, val_data,val1_data], verbose_eval=500, early_stopping_rounds = 4000)
 #   validation_data=([val_x1,val_x2],val_y)
 #   validation_data.shape
#    logger = Logger(patience=10, out_path='./', out_fn='cv_{}.h5'.format(c))
#    model.fit(trn_x,trn_y,batch_size=512,epochs=500,verbose=1,callbacks=[logger],validation_data=(val_x,val_y))
#    model.load_weights('cv_{}.h5'.format(c))
    val_pred = clf.predict(val_x,num_iteration=clf.best_iteration)
    test_fold_pred = clf.predict(X_test,num_iteration=clf.best_iteration)

    print("AUC = {}".format(metrics.roc_auc_score(val_y, val_pred)))
    oof_preds_lgb[val_, :] = val_pred.reshape((-1, 1))
    test_preds_lgb += test_fold_pred.reshape((-1, 1))
    del trn_x, trn_y , val_x,val_y
    gc.collect()
    
test_preds_lgb /= NFOLDS


# In[33]:


np.mean(test_preds_lgb)


# In[ ]:


roc_score = metrics.roc_auc_score(y, oof_preds_lgb.ravel())
print("Overall AUC = {}".format(roc_score))

print("Saving OOF predictions")
#oof_preds = pd.DataFrame(np.column_stack((train_ids, oof_preds.ravel())), columns=['ID_code', 'target'])
#oof_preds.to_csv('../kfolds/nn__{}.csv'.format( str(roc_score)), index=False)

print("Saving code to reproduce")
#shutil.copyfile('../model_source/nn__{}.py'.format( str(roc_score)))


#abc =  test_preds_lgb + test_preds_sq_lgb + test_preds_c_lgb +  test_preds_log_lgb
#abc = abc/4
print("Saving submission file")
sample = pd.read_csv('../input/sample_submission.csv')
sample.target = test_preds_lgb.astype(float)
sample.ID_code = test_ids
sample.to_csv('submission__smote_lgb_{}.csv'.format(str(roc_score)), index=False)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




