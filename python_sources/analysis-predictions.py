#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# In[ ]:


sample_submission = pd.read_csv('../input/sample_submission.csv')


# In[ ]:


train.head()


# In[ ]:


train = train.drop('unique_id', axis=1)
test = test.drop('unique_id', axis=1)


# In[ ]:


train1 = train.replace([np.inf, -np.inf], np.nan)
test1 = test.replace([np.inf, -np.inf], np.nan)


# In[ ]:


X = train1.drop('targets', axis=1)
y = train1['targets']


# In[ ]:


X.shape


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)

X_res, y_res = smote.fit_resample(X, y)


# In[ ]:


X_res.shape


# In[ ]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_res_scaled = scaler.fit_transform(X_res, y_res)


# In[ ]:


from sklearn.decomposition import PCA

pca = PCA(n_components=40)

principal_comp = pca.fit_transform(X_res_scaled)


# In[ ]:


principal_comp.shape


# In[ ]:





# In[ ]:


from sklearn.model_selection import train_test_split

xtrain, xtest, ytrain, ytest = train_test_split(principal_comp, y_res, random_state=0, test_size = 0.2)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=300, n_jobs=-1, oob_score=True, random_state = 0)

rf.fit(xtrain, ytrain)

rf.score(xtrain, ytrain)


# In[ ]:


rf.score(xtest, ytest)


# In[ ]:





# In[ ]:





# In[ ]:


test2 = scaler.fit_transform(test1)


# In[ ]:


test_comp = pca.fit_transform(test2)


# In[ ]:


test_comp.shape


# In[ ]:


predictions = rf.predict_proba(test_comp)


# In[ ]:


len(predictions)


# In[ ]:


predictions[:5]


# In[ ]:


prob_cols = sample_submission.columns[1:]
prob_cols


# In[ ]:


from tqdm import tqdm


# In[ ]:


result = {}

for i in tqdm(range(9)):
    arr = []
    for j in tqdm(range(len(predictions))):
        arr.append(predictions[j][i])
    result[prob_cols[i]] = arr


# In[ ]:


for key in result.keys():
    sample_submission[key] = result[key]


# In[ ]:


sample_submission.head()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


sample_submission.to_csv('sub6.csv', sep=',', index=False)


# In[ ]:




