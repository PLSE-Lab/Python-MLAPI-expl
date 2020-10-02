#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
from sklearn.model_selection import cross_val_score,train_test_split
from lightgbm import LGBMClassifier
import xgboost as xgb
import os
import csv
print(os.listdir("../input"))
import re
# Any results you write to the current directory are saved as output.


# In[2]:


TestFile="../input/test.csv"
TrainFile="../input/train.csv"
Building_structure_File="../input/Building_Structure.csv"
Building_ownership_File="../input/Building_Ownership_Use.csv"


# In[3]:


train=pd.read_csv(TrainFile)
ownership=pd.read_csv(Building_ownership_File)
structure=pd.read_csv(Building_structure_File)
test=pd.read_csv(TestFile)
train


# In[4]:


test


# In[5]:


train.columns


# In[6]:


ownership.columns


# In[7]:


structure.columns


# In[8]:


train.info()


# In[9]:


trainy=pd.DataFrame(train[['building_id','damage_grade']])
train.drop('damage_grade',inplace=True,axis=1)


# In[10]:


intersect=list(set(train.columns).intersection(set(structure.columns),set(ownership.columns)))
intersect


# In[11]:


sizetrain=train.shape[0]
sizetrain


# In[12]:


mix=pd.concat([train,test])
mix.info()


# In[13]:


mix.has_repair_started[np.isnan(mix.has_repair_started)].shape[0] #number of rows having nan values


# In[14]:


mix.has_repair_started.fillna(value=0,inplace=True) # fill nan values with 0
mix.has_repair_started[np.isnan(mix.has_repair_started)].shape[0] 


# In[15]:


mix.info()


# In[16]:


mix.area_assesed.value_counts()


# In[17]:


merged=pd.merge(mix,ownership,on=intersect)
merged=pd.merge(merged,structure,on=intersect)


# In[18]:


merged #let's check if all columns have values


# In[19]:


merged.info()


# In[20]:


for value in merged.columns.values:
    if((merged[value][pd.isnull(merged[value])].shape[0])!=0):
        print(value,merged[value][pd.isnull(merged[value])].shape[0])
        merged[value].fillna(value=merged[value].mode()[0],inplace=True)


# In[21]:


for value in merged.columns:
    print(value,merged[value].unique(),merged[value].unique().shape[0],merged.dtypes[value])


# Finding Features that can be one hot encoded

# In[22]:


encodable_features=[]
for value in merged.columns:
    if(merged[value].unique().shape[0]<=15 and merged[value].unique().shape[0]>2):
        print(value,merged[value].unique().shape[0])
        encodable_features.append(value)


# In[23]:


encodable_features


# In[24]:


new_merged=pd.get_dummies(merged,columns=encodable_features)
new_merged.head()


# In[25]:


train_mod=new_merged[0:sizetrain]
test_mod=new_merged[sizetrain:]
train_mod.head()
## back to original dataset


# In[26]:


trainy.damage_grade=trainy.damage_grade.apply(lambda x:re.sub('Grade ','',x)).astype(int) 
# converted damage_grade from form "Grade 1" to 1  i.e to type int also


# In[27]:


trainy.info()


# In[28]:


train_modx,test_modx,train_mody,test_mody=train_test_split(train_mod,trainy['damage_grade'],test_size=0.33,random_state=0) 
#33% data as test data


# ### Accuracy on Dtree with no fixed length

# In[29]:


from sklearn.tree import DecisionTreeClassifier
Dtree=DecisionTreeClassifier()
Dtree.fit(X=train_modx.select_dtypes(include=['int','float']),y=train_mody)
print(Dtree.score(X=test_modx.select_dtypes(include=['int','float']),y=test_mody))
print(Dtree.tree_.max_depth) 
#tree depth also printed


# ### Trying different lengths

# In[30]:


a=[]
for i in range(1,40):
    Dtree=DecisionTreeClassifier(max_depth=i)
    Dtree.fit(X=train_modx.select_dtypes(include=['int','float']),y=train_mody)
    a.append([i,Dtree.score(X=test_modx.select_dtypes(include=['int','float']),y=test_mody)])


# In[31]:


plt.plot([a[i][1] for i in range(len(a))])


# In[32]:


a[17] #tree length is 18


# In[33]:


np.asarray(a)[:,1].max() ## maximum obtained accuracy 


# In[35]:


Dtree.max_depth=18
Dtree.fit(X=train_mod.select_dtypes(include=['int','float']),y=trainy.damage_grade)
predicted=Dtree.predict(X=test_mod.select_dtypes(['int','float']))


# In[36]:


with open("predict1_DT.csv","w") as outfile:
    writer=csv.writer(outfile,delimiter=",")
    writer.writerow(("building_id","damage_grade"))
    for i in range(test_mod.shape[0]):
        writer.writerow([test_mod.building_id.values[i],"Grade {}".format(predicted[i])])


# lgbm=LGBMClassifier(feature_fraction=.5,bagging_fraction=.01,boosting='dart',learning_rate=.001)
# lgbm.fit(X=train_modx.select_dtypes(['int','float']),y=train_mody)
# lgbm.score(X=test_modx.select_dtypes(['int','float']),y=test_mody)

# from sklearn.ensemble import RandomForestClassifier
# rfc=RandomForestClassifier(n_estimators=7)
# rfc.fit(X=train_modx.select_dtypes(['int','float']),y=train_mody)
# rfc.score(X=test_modx.select_dtypes(['int','float']),y=test_mody)

# rfc.fit(X=train_mod.select_dtypes(['int','float']),y=trainy.damage_grade)
# predicted=rfc.predict(X=test_mod.select_dtypes(['int','float']))

# with open("predict1_RF.csv","w") as outfile:
#     writer=csv.writer(outfile,delimiter=",")
#     writer.writerow(("building_id","damage_grade"))
#     for i in range(test_mod.shape[0]):
#         writer.writerow([test_mod.building_id.values[i],"Grade {}".format(predicted[i])])

# In[37]:


os.listdir("../working/")


# In[ ]:





# In[ ]:




