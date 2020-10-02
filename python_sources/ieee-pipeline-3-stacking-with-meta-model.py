#!/usr/bin/env python
# coding: utf-8

# Input - Val & test preds from single models, validation set, sample_submission CSV
# - https://www.kaggle.com/priteshshrivastava/ieee-pipeline-2-a-model-a-catboost-feat-sel
# - https://www.kaggle.com/priteshshrivastava/ieee-pipeline-2-b-model-b-random-forest
# - https://www.kaggle.com/priteshshrivastava/ieee-pipeline-2-c-model-c-xgboost
# 
# Output - Competition submission file

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import linear_model
import os
import math
from sklearn.metrics import roc_auc_score

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


#predsA = pd.read_csv("")
#test_predsA = pd.read_csv("")

predsB = pd.read_csv("/kaggle/input/ieee-pipeline-2-b-model-b-random-forest/predsB.csv")
test_predsB = pd.read_csv("/kaggle/input/ieee-pipeline-2-b-model-b-random-forest/test_predsB.csv")

predsC = pd.read_csv("/kaggle/input/ieee-pipeline-2-c-model-c-xgboost/predsC.csv")
test_predsC = pd.read_csv("/kaggle/input/ieee-pipeline-2-c-model-c-xgboost/test_predsC.csv")

predsB.head()


# ### Defining function to calculate the evaluation metric

# In[ ]:


def auc(x,y): 
    return roc_auc_score(x,y)
def print_score(m):
    res = [auc(m.predict(train_X), train_y), auc(m.predict(val_X), val_y)]
    print(res)


# ### Form a new dataset for validation & test by stacking the predictions

# In[ ]:


stacked_predictions = np.column_stack((predsB, predsC))
stacked_predictions = pd.DataFrame({'predsB': stacked_predictions[:, 0],
                                    'predsC': stacked_predictions[:, 1]})

stacked_test_predictions = np.column_stack((test_predsB, test_predsC))
stacked_test_predictions = pd.DataFrame({'test_predsB': stacked_test_predictions[:, 0],
                                         'test_predsC': stacked_test_predictions[:, 1]})

stacked_predictions.head()


# ### Specify meta model & fit it on stacked validation set predictions

# In[ ]:


meta_model = linear_model.LogisticRegression()


# In[ ]:


val_y = pd.read_csv("/kaggle/input/ieee-pipeline-1-create-validation-set/val_y.csv")
val_y.head()


# In[ ]:


meta_model.fit(stacked_predictions, val_y)


# ### Use meta model to make preditions on the stacked predictions of test set

# In[ ]:


#final_predictions = meta_model.predict(stacked_test_predictions)  ## Or Lower AUC even though individual models score higher
final_predictions = meta_model.predict_proba(stacked_test_predictions)[:,1]   ## Probabilities generally imporove AUC


# ### Submit predictions

# In[ ]:


submission = pd.read_csv('/kaggle/input/ieee-fraud-detection/sample_submission.csv')
submission.head()


# In[ ]:


submission['isFraud'] = final_predictions 
submission.to_csv('meta_stacking_v1.csv', index=False)

