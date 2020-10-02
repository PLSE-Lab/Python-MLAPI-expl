#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[2]:


train=pd.read_csv('../input/dataset2147b1d/train_file.csv')


# In[3]:


train.head()


# In[4]:


train.drop(columns=['Description','GeoLocation','QuestionCode','Patient_ID','YEAR'],inplace=True)


# In[5]:


train.head()


# In[6]:


test=pd.read_csv('../input/dataset2147b1d/test_file.csv')


# In[7]:


test_pid=test.Patient_ID


# In[8]:


train.head()


# In[9]:


import category_encoders as ce


# In[10]:


ce1=ce.TargetEncoder(cols = ['LocationDesc','Greater_Risk_Question','Race','Grade','StratID1','StratID2','StratID3'], min_samples_leaf = 20)


# In[11]:


train.loc[:,['LocationDesc','Greater_Risk_Question','Race','Grade','StratID1','StratID2','StratID3']]=ce1.fit_transform(train.loc[:,['LocationDesc','Greater_Risk_Question','Race','Grade','StratID1','StratID2','StratID3']],train.loc[:,['Greater_Risk_Probability']])


# In[12]:


train.head()


# In[13]:


train=pd.get_dummies(data=train,columns=['Sex','StratificationType'])


# In[14]:


train.head()


# In[15]:


X=train.drop(columns=['Greater_Risk_Probability'])
Y=train['Greater_Risk_Probability']


# In[16]:


from sklearn.model_selection import train_test_split


# In[17]:


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,random_state=10,test_size=0.2)


# In[18]:


from xgboost import XGBRegressor
#xgbr=XGBRegressor()


# In[68]:


xgbr = XGBRegressor(colsample_bytree=0.4,
                    objective="reg:linear",
                 gamma=0.5,                 
                 learning_rate=0.01,
                 max_depth=5,
                 min_child_weight=1.5,
                 n_estimators=5000,                                                                    
                 reg_alpha=0.75,
                 reg_lambda=0.45,
                 subsample=0.6,
                 seed=42) 


# In[28]:


from keras.models import Sequential
from keras.layers import Dense,Dropout,BatchNormalization


# In[64]:


from catboost import CatBoostRegressor
sgd=CatBoostRegressor(iterations=10000,learning_rate=0.001,depth=5)


# In[29]:


X_train.shape


# In[69]:


xgbr.fit(X_train,Y_train)
#sgd.fit(X_train,Y_train)


# In[56]:


xgbr.feature_importances_


# In[57]:


from sklearn.metrics import mean_absolute_error


# In[ ]:


y_pred=xgbr.predict(X_test)


# In[ ]:


mean_absolute_error(Y_test,y_pred)


# In[ ]:


test.head()


# In[ ]:


test.drop(columns=['Description','GeoLocation','QuestionCode','Patient_ID','YEAR'],inplace=True)


# In[ ]:


test.head()


# In[ ]:


test.loc[:,['LocationDesc','Greater_Risk_Question','Race','Grade','StratID1','StratID2','StratID3']]=ce1.transform(test.loc[:,['LocationDesc','Greater_Risk_Question','Race','Grade','StratID1','StratID2','StratID3']])


# In[ ]:


test.head()


# In[ ]:


test=pd.get_dummies(data=test,columns=['Sex','StratificationType'])


# In[ ]:


test.head()


# In[ ]:


test_pred=xgbr.predict(test)


# In[ ]:


output=pd.DataFrame([np.array(test_pid).astype(np.int64),test_pred])


# In[ ]:


output=output.T


# In[ ]:


output.columns=['Patient_ID','Greater_Risk_Probability']


# In[ ]:


output.Greater_Risk_Probability=output.Greater_Risk_Probability.astype(np.float64)


# In[ ]:


output.head()


# In[ ]:


output.to_csv('submissionapr28v1.csv',index=None)


# In[ ]:


#tg=train.groupby('Greater_Risk_Question')


# In[ ]:


#train.Greater_Risk_Question.value_counts().plot(kind='bar')


# In[ ]:


#from matplotlib import pyplot as plt


# In[ ]:


#list(tg.groups.keys())


# In[ ]:


'''plt.suptitle('Histogram of Numerical Column', fontsize=20)
for i in range(1,21):
    plt.figure(num=20,figsize=(10,150))
    plt.subplot(20,2,i)
    f=plt.gca()
    f.set_title(list(tg.groups.keys())[i-1])
   # vals=np.size(train.iloc[tg.groups[list(tg.groups.keys())[i-1]],:]['Greater_Risk_Probability'].unique())
    plt.hist(train.iloc[tg.groups[list(tg.groups.keys())[i-1]],:]['Greater_Risk_Probability'],color='#3f5d7d')'''


# In[ ]:


#train.Sex.v


# In[ ]:


#test.Sex.value_counts()


# In[ ]:


#pk=train.groupby(['Greater_Risk_Question','Sex','Race']).Greater_Risk_Probability.median()


# In[ ]:


#len(pk)


# In[ ]:


#train.groupby(['LocationDesc',]).Greater_Risk_Probability.median().sort_values(ascending=True).plot(kind= 'bar',figsize=(50,10),fontsize=20,)


# In[ ]:


#train.iloc[tg.groups['Currently used marijuana'],:]['Greater_Risk_Probability'].hist(),
#plt.hist(train.iloc[tg.groups['Ever drank alcohol'],:]['Greater_Risk_Probability'])
#plt.show()


# In[ ]:


#train.iloc[tg.groups['Currently used marijuana'],:]['Grade'].value_counts().plot(kind='bar')


# In[ ]:


#train.iloc[tg.groups['Currently used marijuana'],:]['QuestionCode'].value_counts().plot(kind='bar')


# In[ ]:


#train.Greater_Risk_Question.value_counts().plot(kind='bar')


# In[ ]:


#train.StratificationType.value_counts()


# In[ ]:


#train.GeoLocation.value_counts().plot(kind='bar', figsize=(120,30),fontsize=40)


# In[ ]:


#train.groupby('LocationDesc').get_group('Houston, TX').GeoLocation.value_counts()


# In[ ]:


#train.GeoLocation = train.groupby(['LocationDesc'])['GeoLocation']\.transform(lambda x: x.fillna(x))


# In[ ]:


#train.isnull().sum()


# In[ ]:


#train[train.GeoLocation.isnull()].LocationDesc.value_counts()


# In[ ]:


#train[~train.GeoLocation.isnull()].LocationDesc.value_counts()


# In[ ]:


'''from geopy.geocoders import Nominatim

geolocator = Nominatim() 

for location in ('California USA', 'United States','Shelby County, TN'):
    geoloc = geolocator.geocode(location)
    print(location, ':', geoloc,(geoloc.latitude, geoloc.longitude))'''


# In[ ]:


mylist=[1,2,3]


# In[ ]:


import numpy as np
myarray = np.asarray(mylist)


# In[ ]:


myarray

