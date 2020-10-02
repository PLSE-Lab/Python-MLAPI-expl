#!/usr/bin/env python
# coding: utf-8

# Importing All useful Libraries

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from catboost import CatBoostClassifier


# In[ ]:


train_meta = pd.read_csv("../input/siim-isic-melanoma-classification/train.csv")
test_meta = pd.read_csv("../input/siim-isic-melanoma-classification/test.csv")

useful_cols = ['sex',"age_approx","anatom_site_general_challenge"]
TARGET = "target"
ID = "image_name"
train_meta = train_meta[useful_cols+[TARGET,ID]]


# In[ ]:


train_meta.isna().sum()


# We see that we have some Nan values in repective columns , so as test data also contains anatom_site_general_challenge wit some nan values ,hence we fill a new class into these nan positions and drop records where age or sex are provided as nan.

# In[ ]:


train_meta['anatom_site_general_challenge'] = train_meta['anatom_site_general_challenge'].fillna("unknown_site")
test_meta['anatom_site_general_challenge'] = test_meta['anatom_site_general_challenge'].fillna("unknown_site")


# In[ ]:


train_meta.dropna(inplace=True)


# We use LabelEncoder to encode the string columns present in our dataset. and store respective encoders for each column into a dictionary so that it can be used in test data also.

# In[ ]:


from sklearn.preprocessing import LabelEncoder
from collections import defaultdict

encoders = defaultdict(LabelEncoder)
for column in train_meta.select_dtypes("object").columns:
    if column in [ID,TARGET]:
        continue
    encoder = LabelEncoder()
    train_meta[column] = encoder.fit_transform(train_meta[[column]])
    encoders[column] = encoder


# In[ ]:


train_meta.shape


# Now comes the main model part.
# I have implemented 5 kfold ,to examine my model performance.

# In[ ]:


X = train_meta.drop([ID,TARGET],axis=1)
Y = train_meta[[TARGET]]

from sklearn.model_selection import KFold
folds = KFold(n_splits=5,shuffle=True)

params ={
    "od_type":"Iter",
    'od_wait':100,
    "eval_metric":"AUC",
    'loss_function':'Logloss',
    "iterations":1000,
    "verbose":100
}

scores = []

max_score = -np.inf
for (train_idx,test_idx),i in zip(folds.split(X,Y),range(0,5)):
    print("Working On fold ",i)
    model = CatBoostClassifier(**params)
    model.fit(X.iloc[train_idx],Y.iloc[train_idx],
              eval_set=(X.iloc[test_idx],Y.iloc[test_idx]),
              cat_features = ["sex","anatom_site_general_challenge"])
    
    score = model.score(model.predict(X.iloc[test_idx]),Y.iloc[test_idx])
    print("Achieved AUC Score :" ,score)
    scores.append(score)
    print(scores)
    if score > max_score:
        best_idx = (train_idx,test_idx)
        max_score = score
        
    print("-"*100)
    

print("Final Results from 5 KFOLD")
print("Min Score",min(scores))
print("Mean Score",sum(scores)/len(scores))
print("Max Score",max(scores))


# I have used the best indices provided from above experiment to prepare my model for final submission

# In[ ]:


model = CatBoostClassifier(**params)
model.fit(X.iloc[best_idx[0]],Y.iloc[best_idx[0]],
              eval_set=(X.iloc[best_idx[1]],Y.iloc[best_idx[1]]),
              cat_features = ["sex","anatom_site_general_challenge"])

score = model.score(model.predict(X.iloc[test_idx]),Y.iloc[test_idx])
print("Achieved AUC Score :" ,score)
    


# ## Submission Time

# converting classes back to there respective encoded values using LabelEncoders we trained before.

# In[ ]:


for key in encoders.keys():
    test_meta[key] = encoders[key].transform(test_meta[key])
test_meta.head()


# Final Touch to submit  csv file.

# In[ ]:


testing = test_meta.drop([ID,'patient_id'],axis=1)
predictions = model.predict_proba(testing)
predictions = [i[1] for i in predictions]
test_meta[TARGET] = predictions
test_meta[[ID,TARGET]].to_csv("catboost_submission.csv",index=None)


# With this code , i was able to achieve 0.7 score on public leaderboard.
# I know this is not much to be used , but atleast it can be useful for some one.
# 
# Will be updating this notebook after hyperparameter tuning .
# 
# Kindly comment if you like or dislike something in this notebook, and if you lked please upvote it too.
# and also do share some suggestions which can be used to improve it.
# 
# Thank You.

# In[ ]:




