#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
os.listdir('../input')


# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


import h2o


# In[ ]:


h2o.init()


# In[ ]:


dFrame = h2o.import_file('../input/heart-disease-uci/heart.csv')


# In[ ]:


dFrame.head()


# In[ ]:


dFrame.columns


# In[ ]:


dFrame.summary()


# In[ ]:


dFrame.describe()


# In[ ]:


dFrame.cor()


# In[ ]:


Corr = dFrame.cor().as_data_frame()
Corr.index = dFrame.columns
mask = np.triu(np.ones_like(Corr, dtype=np.bool))
f, ax = plt.subplots(figsize=(11, 9))
sns.heatmap(Corr, mask=mask, cmap='RdYlGn', vmax=.3, center=0,square=True, linewidths=.5, cbar_kws={"shrink": .5},annot=True)
 


# In[ ]:


train,valid, test = dFrame.split_frame(ratios=[0.6,0.1],seed=1234)

train["target"] = train["target"].asfactor()
valid["target"] = valid["target"].asfactor()
test["target"]  = test["target"].asfactor()


# In[ ]:


print(train.shape)
print("*"*20)
print(valid.shape)
print("*"*20)
print(test.shape)
print("*"*20)


# In[ ]:


predct = dFrame.columns[:-1]


# In[ ]:


from h2o.estimators.glm import H2OGeneralizedLinearEstimator
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.estimators.random_forest import H2ORandomForestEstimator


# In[ ]:


gbm = H2OGradientBoostingEstimator()
gbm.train(x=predct,y="target",training_frame=train)


# In[ ]:


print(gbm)


# In[ ]:


prfm = gbm.model_performance(valid)
print(prfm)


# In[ ]:


tuning = H2OGradientBoostingEstimator(
    ntrees = 1000,
    learn_rate = 0.01,
    stopping_rounds = 22,
    stopping_metric = "AUC",
    col_sample_rate = 0.7,
    sample_rate = 0.8,
    seed = 1234
) 


# In[ ]:


tuning.train(x=predct, y="target", training_frame=train, validation_frame=valid)


# In[ ]:


print (tuning.model_performance(valid).auc()*100)


# In[ ]:


from h2o.estimators import H2OXGBoostEstimator


# In[ ]:


xgb = H2OXGBoostEstimator(ntrees=1000,learn_rate=0.05,stopping_rounds=20,stopping_metric="AUC",nfolds=10,seed=1234)


# In[ ]:


xgb.train(x=predct,y="target",training_frame=train, validation_frame=valid)


# In[ ]:


print(xgb.model_performance(valid).auc()*100)


# In[ ]:


xgb.varimp


# In[ ]:


xgb.varimp_plot(num_of_features =5)


# In[ ]:


#Use with autoML
from h2o.automl import H2OAutoML


# In[ ]:


autoML = H2OAutoML(max_models=15,max_runtime_secs=150,seed=3)
autoML.train(x=predct,y="target",training_frame = train, validation_frame=valid)


# In[ ]:


print("*"*102)
print(autoML.leaderboard)
print("*"*102)


# In[ ]:


#We can get the information as GBM is perfoming well with aprox ~90% area under curve and 0.39 log loss.


# In[ ]:


#Please upvote the kernal if this feels informative. :)

