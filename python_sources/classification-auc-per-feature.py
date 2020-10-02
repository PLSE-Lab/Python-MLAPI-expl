#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd

import warnings
warnings.simplefilter("ignore")

import os
print(os.listdir("../input"))

pd.set_option("display.max_rows",1000)


# # Params

# In[ ]:


target="HasDetections"
submission_id_col="MachineIdentifier"

SEED_TRAIN=100
SEED_SAMPLE=100
SEED_CV=42
SEED_DEFAULT=42
SAMPLE_ROWS=200000

CV_SPLITS=5
CV_REPEATS=30

np.random.seed(SEED_DEFAULT)


# 
# # Read Data

# In[ ]:


df_kaggle_train = pd.read_hdf(
         '../input/save-hdf-train-1m-sample/train_sample.hdf',
         key="train_sample"
)


# In[ ]:


df_kaggle_train.shape


# # Eval single feature models (single feature model  vs random permuted feature)
# 

# In[ ]:


from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline

pipeline=LGBMClassifier(random_state=SEED_TRAIN,n_jobs=1 )

df_train=df_kaggle_train.sample(SAMPLE_ROWS,random_state=SEED_SAMPLE)

list_results=[]
if __name__ == "__main__":
    for i_feature in df_kaggle_train.columns.values:
        if i_feature not in [submission_id_col,target]:
            from sklearn.model_selection import cross_val_score,RepeatedStratifiedKFold

            rskf = RepeatedStratifiedKFold(n_splits=CV_SPLITS, n_repeats=CV_REPEATS,
                                           random_state=SEED_CV)

            X_train=df_train[[i_feature]]
            y_train=df_train[target]
            
            cv=cross_val_score(pipeline,X_train,y_train,n_jobs=-1,cv=rskf)
            plt.hist(cv,color="blue")
            
            # Permute (remove any feature signal)
            X_train[i_feature]=np.random.permutation(X_train[i_feature].values)
            if X_train[i_feature].dtype=="object":
                X_train[i_feature]=X_train[i_feature].astype("category")  
                
            rskf = RepeatedStratifiedKFold(n_splits=CV_SPLITS, 
                                           n_repeats=CV_REPEATS,random_state=SEED_CV)

            cv_random=cross_val_score(pipeline,X_train,y_train,n_jobs=-1,cv=rskf)
            plt.hist(cv_random,color="gray")
            
            plt.title(f"{i_feature}:{np.round(np.mean(cv)*100,3)} vs {np.round(np.mean(cv_random)*100,3)}")
            plt.show()
            
            list_results.append(pd.DataFrame({"feature":i_feature,
                                              "cv_score":cv,
                                              "cv_score_random":cv_random,
                                              "improved":cv-cv_random}).reset_index())


# # Save results for later reuse

# In[ ]:


df_results=pd.concat(list_results)
df_results


# In[ ]:


df_results.groupby("feature")[["improved"]].mean().sort_values("improved",ascending=False)


# In[ ]:


df_results.to_csv("classification_auc_per_feature.csv",index=False)

