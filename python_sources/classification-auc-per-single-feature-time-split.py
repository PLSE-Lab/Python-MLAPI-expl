#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd

import warnings
warnings.simplefilter("ignore")

import os


# # Read Data

# In[ ]:


target="HasDetections"
submission_id_col="MachineIdentifier"
seed_train=100
SAMPLE_ROWS=200000
random_state=42


# In[ ]:


df_kaggle_train = pd.read_hdf(
         '../input/save-hdf-1m-sample/train_sample.hdf',
         key="train_sample"
)


# In[ ]:


df_kaggle_train.shape


# In[ ]:


def add_timestamps(df):
    datedictAS = np.load('../input/timestamps/AvSigVersionTimestamps.npy')[()]
    df['DateAS'] = df['AvSigVersion'].map(datedictAS)  
    
   
add_timestamps(df_kaggle_train)


# In[ ]:


split_date="2018-09-20"
ix_time_train=df_kaggle_train.query(f"DateAS<'{split_date}'").index.values
ix_time_test=df_kaggle_train.query(f"DateAS>='{split_date}'").index.values
df_kaggle_train.drop(columns="DateAS",inplace=True)
ix_time_train.shape,ix_time_test.shape


# # Cross validate by feature

# In[ ]:


df_train=df_kaggle_train.loc[ix_time_train]
df_test=df_kaggle_train.loc[ix_time_test]


# In[ ]:


from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score

pipeline=Pipeline([
    ("clf",LGBMClassifier(random_state=seed_train,
                          n_jobs=1 ))])

list_results=[]
if __name__ == "__main__":
    for i_feature in df_kaggle_train.columns.values:
        if i_feature not in [submission_id_col,target]:
            from sklearn.model_selection import cross_val_score,RepeatedStratifiedKFold

            X_train=df_train[[i_feature]]
            y_train=df_train[target]
            
            X_test=df_test[[i_feature]]
            y_test=df_test[target]
            pipeline.fit(X_train,y_train.values)
            
            y_pred_test=pipeline.predict_proba(X_test)[:,1]
            time_split_score=roc_auc_score(y_score=y_pred_test,y_true=y_test)
            print(i_feature,time_split_score)
            
            list_results.append(pd.DataFrame({"feature":i_feature,"time_split_score":time_split_score},index=[0]))


# In[ ]:


df_results=pd.concat(list_results).sort_values("time_split_score",ascending=False).reset_index()
df_results


# # Save results for later reuse

# In[ ]:


df_results.to_csv("time_split_feature_results.csv",index=False)

