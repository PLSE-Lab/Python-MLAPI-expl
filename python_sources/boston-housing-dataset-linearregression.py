#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error as MAE
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold, cross_val_score
from itertools import combinations
import scipy


# In[ ]:


df = pd.read_csv('../input/boston_data.csv')
print("objects: %d\nfeatures: %d\n"%(df.shape[0], df.shape[1]-1))


# In[ ]:


print(df.describe())


# looking for the best feature combination

# In[ ]:


#y -> target
y = df.medv
features = df.drop(['medv'], axis=1)

regr = LinearRegression()
MAEs = []

for i in range(1, len(features.columns)+1):
    print("step: ", i)
    min_MAE = 100
    best_feature_sequence = []
    for combination in combinations(features.columns, i):
        slice_features = features[list(combination)]
        cvs = -cross_val_score(regr, slice_features, y, cv=10, scoring='neg_mean_absolute_error').mean()
        if cvs < min_MAE:
            min_MAE = cvs
            best_feature_sequence = list(slice_features.columns)
    print("best_feature_sequence: ", best_feature_sequence)
    print("MAE: ", min_MAE, "\n")
    MAEs.append(min_MAE)


# after adding 6th feature in a row error reduction becomes almost imperceptible

# In[ ]:


plt.plot(range(1, 14), MAEs)

for i in range(len(MAEs)):
    plt.hlines(MAEs[i], 0, 14, colors='r', linestyle='--', alpha=0.6)


# In[ ]:


features = features[['nox', 'rm', 'dis', 'ptratio', 'black', 'lstat']]
regr.fit(features, y)


# In[ ]:


pd.DataFrame({"feature": features.columns, "importance": regr.coef_})


# dependence between "rm" and target is nonlinear

# In[ ]:


plt.figure(figsize=(20, 10))
plt.plot(features['rm'], y, 'o', c='red')
plt.show()


# lets transform the dependency

# In[ ]:


coefs = scipy.polyfit(features.rm, y, deg = 2)
poly = scipy.poly1d(coefs)
plt.figure(figsize=(20, 10))
plt.scatter(features.rm, y, c='red')
plt.plot(sorted(features.rm), poly(sorted(features.rm)))
plt.show()


# In[ ]:


features['rm'] = poly(np.array(features.rm))

