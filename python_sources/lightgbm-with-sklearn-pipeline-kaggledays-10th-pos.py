#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler 
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from lightgbm import LGBMClassifier
from category_encoders import OneHotEncoder
from sklearn.model_selection import cross_val_predict
from warnings import filterwarnings
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
filterwarnings('ignore')
import os
import xgboost as xgb
print(os.listdir("../input"))


# 

# In[ ]:


train = pd.read_csv("../input/train_2.csv")
print("train shape", train.shape)
test = pd.read_csv("../input/test_2.csv")
print("test shape", test.shape)


# In[ ]:


target_column = "target"
id_column = "id"
categorical_cols = [c for c in test.columns if test[c].dtype in [np.object]]
numerical_cols = [c for c in test.columns if test[c].dtype in [np.float, np.int] and c not in [target_column, id_column]]
print("Number of features", len(categorical_cols)+len(numerical_cols))


# In recent sklearn they introduced ColumnTransformer which is a very compact way to define end-2-end solution.

# In[ ]:


# Lowered learning rate from 0.3 to 0.02.
# Set number of trees as 100 (this is the default value)
# Set L1 Regularization to 5. Default value is 0. 
# Increased max_bin to 512. Default is 255. 
classifier = make_pipeline(
    ColumnTransformer([
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(), categorical_cols),    
    ]),
    LGBMClassifier(n_jobs=-1,learning_rate=0.02,num_tree=100,lambda_l1=5,max_bin=512)
)


# In[ ]:


get_ipython().run_cell_magic('time', '', '# 3 fold CV; default option in cross_val_predict is Stratified k-Fold CV\noof_pred = cross_val_predict(classifier, \n                             train, \n                             train[target_column], \n                             cv=3,\n                             method="predict_proba")')


# In[ ]:



print("Cross validation AUC {:.4f}".format(roc_auc_score(train[target_column], oof_pred[:,1])))


# In[ ]:


# compute and print log-loss
from sklearn.metrics import log_loss
log_loss_value = log_loss(train[target_column], oof_pred, eps=1e-15, normalize=True, sample_weight=None, labels=None)
print("The log-loss value is {:.4f}".format(log_loss_value))


# In[ ]:


sub = pd.read_csv("../input/sample_submission.csv")
sub.head()


# In[ ]:


get_ipython().run_cell_magic('time', '', 'classifier.fit(train, train[target_column])\n# predict on test set and save to submission file\ntest_preds = classifier.predict_proba(test)[:,1]\nsub[target_column] = test_preds\nsub.to_csv("submission.csv", index=False)')

