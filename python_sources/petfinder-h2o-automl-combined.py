#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import h2o
import numpy as np
import random as r
import pandas as pd
import matplotlib.pyplot as plt
h2o.init()
import os
print(os.listdir("../input"))


# In[ ]:


train = h2o.upload_file(r'../input/train/train.csv')
test = h2o.upload_file(r'../input/test/test.csv')
test2 = test["PetID"]


# In[ ]:


train["AdoptionSpeed"] = train["AdoptionSpeed"].asfactor()
train["desc_length"] = train["Description"].nchar()
train["name_length"] = train["Name"].nchar()
train["PureBreed"] = (train["Breed2"] == 0).ifelse(1,0)
train["SingleColor"] = (train["Color2"] == 0).ifelse(1,0)
train["Fee"] = (train["Fee"] == 0).ifelse(1,0)
train["VideoAmt"] = (train["VideoAmt"] == 0).ifelse(1,0)
test["desc_length"] = test["Description"].nchar()
test["name_length"] = test["Name"].nchar()
test["PureBreed"] = (test["Breed2"] == 0).ifelse(1,0)
test["SingleColor"] = (test["Color2"] == 0).ifelse(1,0)
test["Fee"] = (test["Fee"] == 0).ifelse(1,0)
test["VideoAmt"] = (test["VideoAmt"] == 0).ifelse(1,0)


# In[ ]:


train.describe()


# In[ ]:


train, validation = train.split_frame([0.9], seed = 345)


# In[ ]:


print("%d/%d/%d" % (train.nrows,validation.nrows,test.nrows))


# In[ ]:


y = "AdoptionSpeed"
ignoreFields = [y,"Description","RescuerID","Name", "PetID"] 
x = [i for i in train.names if i not in ignoreFields]


# In[ ]:


from h2o.automl import H2OAutoML


# In[ ]:


mA = H2OAutoML(
               #exclude_algos = ["DeepLearning"], 
               max_runtime_secs = 360000,
               seed = 123
                )
mA.train(x,y,train)


# In[ ]:


mA.leaderboard


# In[ ]:


test.drop(["Description", "RescuerID", "Name", "PetID"])


# In[ ]:


# p = mA.leader.predict(validation)


# In[ ]:


mA.leader.model_performance(validation)


# In[ ]:


p2 = mA.leader.predict(test)


# In[ ]:


p2["predict"]


# In[ ]:


test["PetID"] = test2["PetID"]
test["AdoptionSpeed"] = p2["predict"]


# In[ ]:


submission = test[["PetID","AdoptionSpeed"]]
h2o.export_file(submission,'submission.csv')

