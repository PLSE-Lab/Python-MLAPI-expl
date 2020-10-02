#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


train = pd.read_csv(r"../input/train.csv")    .drop(["id"], axis=1)    .rename(lambda x: "x"+str(x), axis='columns')    .rename(index=str, columns={"xtarget":"target"})
    
test = pd.read_csv(r"../input/test.csv")    .drop(["id"], axis=1)    .rename(lambda x: "x"+str(x), axis='columns')


# In[ ]:


print("Train shape =", train.shape)
print("Test shape =", test.shape)


# In[ ]:


# Feature Engineering add the mean as variable
train["mean"]=train.iloc[:,1:].mean(axis=1)
test["mean"]=test.iloc[:,:].mean(axis=1)


# In[ ]:


from sklearn.preprocessing import scale
train.iloc[:,1:] = scale(train.iloc[:,1:])
test.iloc[:,:] = scale(test)


# In[ ]:


import pymc3 as pm


# In[ ]:


with pm.Model() as mdl_1:
    features = ['x'+str(i) for i in range(300)]+["mean"]
    formula = 'target ~ ' + " + ".join(features)
    #priors_model = dict({"Intercept": pm.Cauchy.dist(0, 10)},**{str(i):pm.Laplace.dist(mu=0,b=0.1) for i in features})
    priors_model = dict({"Intercept": pm.Cauchy.dist(0, 10)},**{str(i):pm.StudentT.dist(nu=1, mu=0, sd=0.03) for i in features})
    pm.glm.GLM.from_formula(formula, train, family=pm.glm.families.Binomial(),priors=priors_model)
    trace_1 = pm.sample(2000, chains=1, tune=1000, init='adapt_diag')


# In[ ]:


from math import exp
df_trace = pm.trace_to_dataframe(trace_1)
pred = test.apply(lambda row: 1/(1+exp(-pd.Series(row*df_trace[features].mean()+df_trace["Intercept"].mean()).sum(axis=0))),axis=1)


# In[ ]:


df = pd.DataFrame({"id":pd.read_csv("../input/test.csv").pop('id'),'target':pred})
df[['id', 'target']].to_csv('submission.csv', index=False) 


# In[ ]:




