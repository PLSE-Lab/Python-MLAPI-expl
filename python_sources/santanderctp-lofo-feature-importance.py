#!/usr/bin/env python
# coding: utf-8

# # Leave One Feature Out (LOFO) Feature Importance
# 
# https://github.com/aerdem4/lofo-importance

# In[ ]:


get_ipython().system('pip install lofo-importance')


# In[ ]:


import pandas as pd
import numpy as np

train_df = pd.read_csv("../input/train.csv")
train_df.shape


# In[ ]:


from sklearn.model_selection import StratifiedKFold
from lightgbm import LGBMClassifier

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
model = LGBMClassifier(n_estimators=50, num_leaves=7, n_jobs=-1)


# In[ ]:


from lofo import LOFOImportance, plot_importance

features = train_df.columns[2:]

lofo_imp = LOFOImportance(train_df, features, "target", model=model, cv=skf, scoring="roc_auc")

importance_df = lofo_imp.get_importance()
importance_df.head()


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')

plot_importance(importance_df, figsize=(12, 32))


# In[ ]:




