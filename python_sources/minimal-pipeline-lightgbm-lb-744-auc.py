#!/usr/bin/env python
# coding: utf-8

# # Minimal pipeline with LightGBM (~.744 AUC on Public Leaderboard)
# 
# - Still trying to make it shorter. Please do let me know if you have any suggestions/improvements! 
# - No EDA, no custom/manual feature engineering (aside from generic numeric/categoricals handling in pipeline)
# - (pipeline goal is to be as generic as possible)
# - some warnings with LabelEnconder I coudnt fix yet, checking.
# - note: sklearn .20 has some  solutions for OneHot/Categorical Imputation, anyone knows ow to update the package Kaggle kernels?

# In[ ]:


import numpy as np
import pandas as pd
import warnings
import sklearn


# In[ ]:


target="TARGET"
submission_id_col="SK_ID_CURR"

seed_split=1 
test_size=1/3
seed_train=100

df_kaggle_train=pd.read_csv("../input/application_train.csv")
df_kaggle_test=pd.read_csv("../input/application_test.csv")


# In[ ]:


from sklearn.model_selection import train_test_split

# Split X,y
y= df_kaggle_train[target].values
df_kaggle_train.drop(columns=target,inplace=True)

# Split kaggle train, reserve internal hold out test set
X_train, X_test, y_train,y_test = train_test_split(df_kaggle_train,y, test_size=test_size, random_state=seed_split,stratify =y)


# In[ ]:


from sklearn.base import TransformerMixin

# Note: sklearn .20 has now SimpleImputer works for categorical also, workaround
class DataFrameImputer(TransformerMixin):

    def __init__(self, default_value="NA"):
        self.default_value = default_value
        
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return pd.DataFrame(X).fillna(self.default_value)


# In[ ]:


import sklearn.preprocessing as pp
from sklearn_pandas import DataFrameMapper

# Some workarounds for sklearn 19.1 (.20 use OneHot, SimpleImputer)
nums=[ ([c],pp.Imputer()) for c in X_train.select_dtypes(include=[np.number])]
cats=[ ([c],[DataFrameImputer(default_value="NA"), pp.LabelEncoder()]) for c in X_train.select_dtypes(include=["object"])]
mapper=DataFrameMapper(nums+cats)


# In[ ]:


from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier

pipeline=Pipeline([('featurize', mapper),("clf",LGBMClassifier(random_state=seed_train))])


# In[ ]:


with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    pipeline.fit(X_train,y_train)
    
    y_pred_train=pipeline.predict_proba(X_train)[:,1]
    y_pred_test=pipeline.predict_proba(X_test)[:,1]


# In[ ]:


from sklearn.metrics import roc_auc_score

print("train score",roc_auc_score(y_score=y_pred_train,y_true=y_train))
print("test score",roc_auc_score(y_score=y_pred_test,y_true=y_test))


# In[ ]:


# Full fit
full_pipeline=sklearn.clone(pipeline)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    full_pipeline.fit(df_kaggle_train,y)
    y_pred_submission=full_pipeline.predict_proba(df_kaggle_test)[:,1]


# In[ ]:


# Prepare submission
df_submission=pd.DataFrame({submission_id_col:df_kaggle_test[submission_id_col],target:y_pred_submission})
df_submission.head()


# In[ ]:


# Check predictions 
df_submission[target].hist()
print("y mean:",np.mean(y))
print("y submission mean:",df_submission[target].mean())


# In[ ]:


df_submission.to_csv(f"submission.csv",index=False)
print("Done!")

