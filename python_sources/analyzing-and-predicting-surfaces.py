#!/usr/bin/env python
# coding: utf-8

# Let's get started. This is like a starter kernel.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


df_train_x = pd.read_csv("../input/X_train.csv")
df_test_x = pd.read_csv("../input/X_test.csv")
df_train_y = pd.read_csv("../input/y_train.csv")


# In[ ]:


df_train_x.info()


# NO null values

# In[ ]:


df_train_y.info()


# NO null values...but size of two DFs (train and test) is different.

# In[ ]:


df_train_x.sample()


# In[ ]:


df_test_x.series_id.value_counts()


# In[ ]:


df_train_y.sample(5)


# In[ ]:


df_train_x.sample(5)


# So, I would like to append the surface column values to respective Series,measurement_number.

# In[ ]:


df_train_y[df_train_y.series_id == 1690]


# In[ ]:


df_train_x[df_train_x.series_id == 1690].shape


# So, for each series ID there are 127 records with different meaurment number

# In[ ]:


"""series_id_y = df_train_y.series_id.tolist()
group_id_y = df_train_y.group_id.tolist()
surface = df_train_y.surface.tolist()
series_id_x = df_train_x.series_id.tolist()
measurment_id = df_train_x.measurement_number.tolist()
just_check_y = list(zip(series_id_y,group_id_y))
just_check_x = list(zip(series_id_x,measurment_id))
store = [-1]*len(just_check_x)
for i in just_check_y:
    if i in just_check_x:
        store[just_check_x.index(i)] = surface[just_check_y.index(i)]
df_train_x["target"] = store"""


# Attached the target column to df_train_x
# * **I WOULD LIKE TO KNOW IF THERE'S ANOTHER SIMPLER WAY TO JOIN **

# In[ ]:


df = pd.merge(df_train_x,df_train_y,how='left',on='series_id')


# In[ ]:


df.sample()


# In[ ]:


df.surface.value_counts()


# In[ ]:


df.sample(3)


# In[ ]:


df.info()


# In[ ]:


df.sample()


# Dropping the columns that are not necessary for building the model

# In[ ]:


df.drop(columns=["row_id","measurement_number","group_id"], inplace=True)
df.sample(2)


# In[ ]:


just_check =  df.groupby("series_id").mean().reset_index()
df = pd.merge(just_check,df_train_y,how='left',on='series_id')
df.sample(3)


# In[ ]:


df.drop(columns=["series_id","group_id"],inplace=True)


# Lets' analyze if there are any outliers that completely deviate and affect the model

# In[ ]:


df.boxplot()
plt.xticks(rotation = 90)


# let's see last 3 columns

# In[ ]:


df[df.columns[-4:]].boxplot()


# These outliers are informative and may help model. I would like to keep them. I will check the model by removing the outlieres in my next commit.

# In[ ]:


df.corr()


# In[ ]:


df.sample(3)


# Just checking whether any two columns correlate each other..
# 
# We can see that Orientation_W and orientation_Z are correlated. I would like to edit it in my next commit.

# * In the above groupby using target variable, I took the average values of all attributes. Based on them we can understand that there are some variations among the target variables and attributes.
# * **But values of llast three columns(linear accelerations) doesn't change much.**

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
train = scaler.fit_transform(df[df.columns[:-1]])
train_x, test_x, train_y, test_y = train_test_split(train,df[df.columns[-1]],test_size = 0.1)


# I'm using Gradient Boosting for building my model

# In[ ]:


train_x.shape


# In[ ]:


from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(train_x, train_y)
target = model.predict(test_x)
mat = confusion_matrix(test_y, target)
print("******confusion******\n",mat)


# In[ ]:


"""from sklearn.metrics import confusion_matrix,accuracy_score
from xgboost import XGBClassifier
for i in range(4,10):
    model = XGBClassifier(model_depth = i)
    model.fit(train_x, train_y)
    target = model.predict(test_x)
    print("accuracy : ", accuracy_score(target, test_y))
mat = confusion_matrix(test_y, target)
print("******confusion******\n",mat)"""


# In[ ]:


df_test_x.drop(columns=["row_id","measurement_number"],inplace=True)
testing = df_test_x.groupby("series_id").mean().reset_index()
testing.sample(3)
testing.shape


# In[ ]:


testing.shape


# In[ ]:


from sklearn.preprocessing import StandardScaler
scale = StandardScaler()
fin_train = scale.fit_transform(df[df.columns[:-1]])
test = scale.transform(testing[testing.columns[1:]])


# In[ ]:


from xgboost import XGBClassifier

model = XGBClassifier()
model.fit(fin_train,df[df.columns[-1]])
target = model.predict(test)


# In[ ]:


a = testing.series_id.tolist()
b = target
submission = pd.DataFrame({"series_id":a,"surface":b})


# In[ ]:


submission.surface.value_counts()


# In[ ]:


#encoding = submission.surface.map({"concrete":1,"soft_pvc":2,"wood":3,"tiled":4,"fine_concrete":5,"soft_tiles":6,"hard_tiles_large_space":7,"carpet":8,"hard_tiles":9})


# In[ ]:


#submission["encoding"] = encoding


# In[ ]:


submission.to_csv("submission.csv",index=False)


# Please guve suggestions to improve my way of approach towards model. Also I would like to know how to improve the model.

# In[ ]:





# In[ ]:




