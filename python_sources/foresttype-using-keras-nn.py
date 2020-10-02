#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
import keras
from keras.layers import *
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df = pd.read_csv("../input/train.csv")
testset = pd.read_csv("../input/test.csv")


# In[ ]:


df.head()


# In[ ]:


testset.head()


# Save the ID column of the testset, which is used at submission.

# In[ ]:


ids = testset['Id']


# Remove Id column as it has no significance.

# In[ ]:


df = df.drop('Id',axis = 1)
testset = testset.drop('Id',axis = 1)


# In[ ]:


df.info()


# Check for null values. Looks like we have no null values.

# In[ ]:


df.isnull().sum()


# In[ ]:


testset.isnull().sum()


# Lets check if any column has no values/not significant

# In[ ]:


df.sum()


# In[ ]:


testset.sum()


# So, soiltype 7 and 15 has no values in it, that means none of the 7 cover types has a soiltype of 7 and 15, so lets these two from the list, as they have no real use.

# In[ ]:


df = df.drop(['Soil_Type7', 'Soil_Type15'],axis =1)
testset = testset.drop(['Soil_Type7', 'Soil_Type15'],axis =1)


# Soiltype 8 and 25 has only one observation each, so lets check what those are.

# In[ ]:


df[df['Soil_Type8'] == 1]


# In[ ]:


df[df['Soil_Type25'] == 1]


# So, both the soiltypes 8 and 25 are for only Covertype 2. So, I think we can remove these columns also. But first lets run our model with these two columns

# I want to try neural networks on this data, since we have a lot of numerical data.
# We need to scale the data to (0,1) for the better results in neural networks

# In[ ]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range = (0,1))


# Lets create our labels

# In[ ]:


labels = df.Cover_Type


# Lets convert the labels to one hot format which is preferred for labels. 
# There are many ways to do, i chose the one below

# In[ ]:


#labels = pd.get_dummies(labels)


# Convert to numpy array, since NN's accept numpy arrays only

# In[ ]:


labels = labels.values


# Lets create our features matrix

# In[ ]:


features = df.drop('Cover_Type',axis =1)


# In[ ]:


features.shape


# In[ ]:


testset.shape # since we dont have the covertype already.


# In[ ]:


features = features.values


# In[ ]:


testset = testset.values


# Heres the snapshot of features 

# In[ ]:


features[0]


# In[ ]:


testset[0]


# As we discussed, NN's work better on the data which is scaled. So lets scale our data.

# In[ ]:


features = scaler.fit_transform(features)


# In[ ]:


testset = scaler.transform(testset)


# Here's how it looks after normalization/scaling.

# In[ ]:


features[0]


# In[ ]:


testset[0]


# In[ ]:


print(type(labels))
print(type(features))
print(type(testset))


# Lets split our dataset into testset and trainset

# In[ ]:


labels = labels - 1


# In[ ]:


train_x,test_x,train_y,test_y = train_test_split(features,labels)


# In[ ]:


print(train_x.shape,train_y.shape,test_x.shape,test_y.shape)


# In[ ]:


train_y


# # KERAS

# model = keras.models.Sequential()
# model.add(Dense(300,input_dim = 52,activation = 'relu'))
# model.add(Dense(700,activation = 'relu'))
# model.add(Dense(200,activation = 'relu'))
# model.add(Dense(7,activation='softmax'))
# model.compile(loss = 'categorical_crossentropy',optimizer = 'adam',metrics = ['categorical_accuracy'])
# model.fit(train_x,train_y,epochs=60,shuffle=True, verbose =1)

# print("The Accuracy on the sampled test set is", model.evaluate(test_x,test_y)[1])

# So, lets now run the same keras model on the whole train set (on whole features) and predict the testset.

# modelmain = keras.models.Sequential()
# modelmain.add(Dense(300,input_dim = 52,activation = 'relu'))
# modelmain.add(Dense(700,activation = 'relu'))
# modelmain.add(Dense(200,activation = 'relu'))
# modelmain.add(Dense(7,activation='softmax'))
# modelmain.compile(loss = 'categorical_crossentropy',optimizer = 'adam',metrics = ['categorical_accuracy'])
# modelmain.fit(features,labels,epochs=120,shuffle=True, verbose =0) #verbose 0 to display no logs

# pred = modelmain.predict(testset)

# covertype = [np.argmax(i)+1 for i in pred]

# In[ ]:


import lightgbm as lgb
from sklearn.metrics import accuracy_score


# In[ ]:


gbm = lgb.LGBMClassifier(objective="mutliclass",n_estimators=10000)
gbm.fit(train_x,train_y,early_stopping_rounds = 100, eval_set = [(test_x,test_y)],verbose = 300)


# In[ ]:


ypred1 = gbm.predict(test_x)


# In[ ]:


ypred1


# In[ ]:


accuracy_score(test_y,ypred1)


# In[ ]:


labels


# In[ ]:


gbm1 = lgb.LGBMClassifier(objective="mutliclass",n_estimators=4000)
gbm1.fit(features,labels,verbose = 1000)


# In[ ]:


finalval = gbm1.predict(testset)


# In[ ]:


covertype = finalval + 1


# In[ ]:


sub = pd.DataFrame({'Id':ids,'Cover_Type':covertype})


# In[ ]:


output = sub[['Id','Cover_Type']]


# In[ ]:


output.to_csv("output1.csv",index = False)


# # PLEASE UPVOTE, IF YOU LIKE IT.
