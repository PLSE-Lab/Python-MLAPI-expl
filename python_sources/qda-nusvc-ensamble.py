#!/usr/bin/env python
# coding: utf-8

# ### Description
# 
# This kernel shows how to get to LB 0.96610 with a single QDA (version 1 has the score) in less than 1 minute.

# ## Building the QDA Classifier

# ### Loading Libraries

# In[1]:


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

# In[2]:


get_ipython().run_cell_magic('time', '', "\ntrain = pd.read_csv('../input/train.csv')\ntest = pd.read_csv('../input/test.csv')")


# ### Training the Classifiers on All Data

# In[3]:


get_ipython().run_cell_magic('time', '', "preds = np.zeros(len(test))\n\ncols = [c for c in train.columns if c not in ['id', 'target', 'wheezy-copper-turtle-magic']]\n    \n# BUILD 512 SEPARATE NON-LINEAR MODELS\nfor i in range(512):\n    print(i)\n    # EXTRACT SUBSET OF DATASET WHERE WHEEZY-MAGIC EQUALS i\n    train2 = train[train['wheezy-copper-turtle-magic']==i].copy()\n    test2 = test[test['wheezy-copper-turtle-magic']==i].copy()\n    #idx1 = train2.index\n    idx2 = test2.index\n    train2.reset_index(drop=True,inplace=True)\n\n    # FEATURE SELECTION (USE APPROX 40 OF 255 FEATURES)\n    sel = VarianceThreshold(threshold=1.5).fit(train2[cols])\n    train3 = sel.transform(train2[cols])\n    test3 = sel.transform(test2[cols])\n\n    #BUILDING THE PIPELINE FOR THE CLASSIFIER\n    pipe = Pipeline([('scaler', StandardScaler()),\n                    ('qda', QuadraticDiscriminantAnalysis(reg_param=0.111))\n                   ])       \n\n\n    pipe.fit(train3,train2['target'])\n    preds[idx2] = pipe.predict_proba(test3)[:,1]")


# In[4]:


from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn import neural_network
from sklearn.svm import NuSVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# In[11]:


preds2 = np.zeros(len(test))
oof = np.zeros(len(train))
for i in range(512):
    train2 = train[train['wheezy-copper-turtle-magic']==i]
    test2 = test[test['wheezy-copper-turtle-magic']==i]
    idx1 = train2.index; idx2 = test2.index
    train2.reset_index(drop=True,inplace=True)

    sel = VarianceThreshold(threshold=1.5).fit(train2[cols])
    train3 = sel.transform(train2[cols])
    test3 = sel.transform(test2[cols])

    #BUILDING THE PIPELINE FOR THE CLASSIFIER
    pipe = Pipeline([('scaler', StandardScaler()),
                    ('nusvc', NuSVC(probability=True, kernel='poly', degree=4, gamma='auto', random_state=4, nu=0.6, coef0=0.08)) 
                   ])       
    
    pipe.fit(train3,train2['target'])
    preds2[idx2] = pipe.predict_proba(test3)[:,1]
        
    if i%15==0: print(i)


# All done! At this point we are ready to make our submission file!

# In[12]:


prediction = (preds + preds2)/2


# In[13]:


sub = pd.read_csv('../input/sample_submission.csv')
sub['target'] = prediction
sub.to_csv('submission.csv',index=False)


# In[14]:


sub.shape


# In[15]:


sub.head()

