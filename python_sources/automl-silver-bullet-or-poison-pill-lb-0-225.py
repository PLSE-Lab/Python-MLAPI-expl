#!/usr/bin/env python
# coding: utf-8

# # Automated Machine Learning with H2O AutoML

# H2O has been dedicated to make ML accessible by non data science expert. One of the major achievements is the AutoML package they developed (and the enterprise version: Driverless AI) It aims to eliminate some of the most time-consuming (yet repetitive) work out of the data science pipeline. (ex. feature engineering, crosss validation, proper targt encoding etc). . How it works is that you specify the feature, set how long you want it to run, and the max memory it can use, and then.........just let it run for as long as you want. At the end, it will evaluate all the models it created, stack ensemble them, and give you the final result.
# 
# I am not sure whether this package is able to handle the complexity of industry-level data science project (where most of the time is spent on cleaning/find the right data...), but it seems like a "Perfec" tool for Kaggle. Knowing this actually makes me a little bit anxious. Indeed, we are here to learn about machine learning and practice our skills in data science, but at the end of the day, this is a competition site. Do you remember the frustration you had we you see an blend of blend of blend public kernel solution earns a silver? This is a similar feeling to me. As these packages become more and more sophisticated, a kaggle competition could turn into a pure competition of computation resources. Just randomly initiate some AutoML instance, let them run for as long you can, create ensemble, and then repeat the process. Indeed...someone with 0 machine learning experience will be able to get good scores this way, especially for competitions with anonymized data.
# 
# Think of it another way, these packages might be able to raise the bar to a new level. I heard from a GM that, 5 years ago, if you know how to do stacking properly, you are already in the top 100 range. Nowadays, almost every kaggler knows how to do stacking (thanks to all generous contributors!) With packages like these, it is no longer enough to play around with sklearn and keras to get a descent score, you have to out-smart these auto-x packages (created by some of the most respectable Grand Masters) to be able to maintain a good stadnding. Challenging yet exiting! 
# 
# Anyways, here you go. All features come from public kernels. I will add references later (cannot remember which kernels I got them from...but I will figure it out)
# 
# #### To replicate the 0.225 score, asumme you have 4 cores, set the run time to 28800s (on your own machine ofc), and give as much ram as you can.  Each run would have different iniialization so scores might be different from time to time. 

# In[ ]:


import time
notebookstart= time.time()

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import gc
print("Data:\n",os.listdir("../input"))

# Models Packages
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn import feature_selection
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import metrics
from sklearn.metrics import mean_squared_error

# Viz
import seaborn as sns
import matplotlib.pyplot as plt


# ### Initializaing
# Initializing H2O AutoML instance, you can pass in your custom setting
# - See here for more details: http://docs.h2o.ai/h2o/latest-stable/h2o-docs/automl.html

# In[ ]:


import h2o
from h2o.automl import H2OAutoML
# Set it according to kernel limits
h2o.init(max_mem_size = "10G")


# In[ ]:


print("\nData Load Stage")
training = pd.read_csv('../input/train.csv', index_col = "item_id", parse_dates = ["activation_date"])#.sample(1000)
traindex = training.index
testing = pd.read_csv('../input/test.csv', index_col = "item_id", parse_dates = ["activation_date"])#.sample(1000)
testdex = testing.index
y = training.deal_probability.copy()
print('Train shape: {} Rows, {} Columns'.format(*training.shape))
print('Test shape: {} Rows, {} Columns'.format(*testing.shape))


# In[ ]:


# Combine Train and Test
df = pd.concat([training,testing],axis=0)
del training, testing
gc.collect()
print('\nAll Data shape: {} Rows, {} Columns'.format(*df.shape))

print("Feature Engineering")
df["price"] = np.log(df["price"]+0.001)
df["price"].fillna(-999,inplace=True)
df["image_top_1"].fillna(-999,inplace=True)

print("\nCreate Time Variables")
df["Weekday"] = df['activation_date'].dt.weekday
df["Weekd of Year"] = df['activation_date'].dt.week
df["Day of Month"] = df['activation_date'].dt.day

# Remove Dead Variables
df.drop(["activation_date","image"],axis=1,inplace=True)

print("\nEncode Variables")
categorical = ["user_id","region","city","parent_category_name","category_name","item_seq_number","user_type","image_top_1"]
messy_categorical = ["param_1","param_2","param_3","title","description"] # Need to find better technique for these
print("Encoding :",categorical + messy_categorical)

# Encoder:
lbl = preprocessing.LabelEncoder()
for col in categorical + messy_categorical:
    df[col] = lbl.fit_transform(df[col].astype(str))
    
X = df.loc[traindex,:].copy()
print("Training Set shape",X.shape)
test = df.loc[testdex,:].copy()
print("Submission Set Shape: {} Rows, {} Columns".format(*test.shape))
del df
gc.collect()


# In[ ]:


test.drop("deal_probability",axis=1, inplace=True)


# In[ ]:


# Training and Validation Set
#X_train, X_valid, y_train, y_valid = train_test_split(
#    X, y, test_size=0.10, random_state=23)


# In[ ]:


#del X, y
#gc.collect()


# In[ ]:


htrain = h2o.H2OFrame(X)
#hval = h2o.H2OFrame(X_valid)
htest = h2o.H2OFrame(test)


# In[ ]:


#del X_train,X_valid,test
#gc.collect()


# In[ ]:


x =htrain.columns
y ='deal_probability'
x.remove(y)

# Set maximum runtime according to Kaggle limits
aml = H2OAutoML(max_runtime_secs = 9989000)
aml.train(x=x, y =y, training_frame=htrain)

print('Generate predictions...')
htrain.drop(['deal_probability'])
#preds = aml.leader.predict(hval)
#preds = preds.as_data_frame()


# In[ ]:


#print('RMSLE H2O automl leader: ', np.sqrt(metrics.mean_squared_error(y_valid, preds)))


# In[ ]:


aml.leader


# In[ ]:


preds = aml.leader.predict(htest)
preds = preds.as_data_frame()


# In[ ]:


#preds


# In[ ]:


lgsub = pd.DataFrame(preds.predict.values,columns=["deal_probability"],index=testdex)
lgsub['deal_probability'].clip(0.0, 1.0, inplace=True) # Between 0 and 1
lgsub.to_csv("lgsub.csv",index=True,header=True)
#print("Model Runtime: %0.2f Minutes"%((time.time() - modelstart)/60))
print("Notebook Runtime: %0.2f Minutes"%((time.time() - notebookstart)/60))


# In[ ]:





# In[ ]:




