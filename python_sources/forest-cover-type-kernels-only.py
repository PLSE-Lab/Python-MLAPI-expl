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

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


pd.options.display.max_rows = None
pd.options.display.max_columns = None


# In[ ]:


df_train = pd.read_csv('../input/train.csv')


# In[ ]:


df_train.head()


# In[ ]:


df_train_feature = df_train.iloc[:,1:-1]


# In[ ]:


df_train_feature.head()


# In[ ]:


df_train_label = df_train.loc[:,['Cover_Type']]


# In[ ]:


df_train_label.head()


# In[ ]:


from sklearn.preprocessing import StandardScaler


# In[ ]:


StdScl = StandardScaler()


# In[ ]:


df_train_feature.iloc[:,0:11].head()


# In[ ]:


df_train_feature.iloc[:,0:10] = StdScl.fit_transform(df_train_feature.iloc[:,0:10])


# In[ ]:


df_train_feature.head()


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,f1_score,classification_report


# In[ ]:


RFC = RandomForestClassifier()


# In[ ]:


RFC.fit(df_train_feature,df_train_label)


# In[ ]:


RFC


# In[ ]:


df_test = pd.read_csv('../input/test.csv')


# In[ ]:


df_test.head()


# In[ ]:


df_test_label = df_test.loc[:,['Id']]


# In[ ]:


df_test_feature = df_test.iloc[:,1:]


# In[ ]:


df_test_feature.iloc[:,0:10] = StdScl.transform(df_test_feature.iloc[:,0:10])


# In[ ]:


df_test_feature.head()


# In[ ]:


prediction = RFC.predict(df_test_feature)


# In[ ]:


prediction = pd.DataFrame(prediction,columns={'Cover_Type'})


# In[ ]:


prediction.head()


# In[ ]:


submission = df_test_label.join(prediction)


# In[ ]:


submission.head()


# In[ ]:


submission.to_csv('sample_upload_forest.csv',index=False)


# In[ ]:


import keras
from keras import layers


# In[ ]:


corr = df_train_feature.corr() 


# In[ ]:


#Feature importance
imp = pd.DataFrame(RFC.feature_importances_)


# In[ ]:


df_train_feature.shape


# In[ ]:


classifier = keras.Sequential()


# In[ ]:


#input layer 1
classifier.add(layers.Dense(128,activation='relu',input_shape=(54,)))


# In[ ]:


#dropout layer for input layer
classifier.add(layers.Dropout(0.3))


# In[ ]:


#hidden layer 1
classifier.add(layers.Dense(64,activation='relu'))
classifier.add(layers.Dropout(0.3))


# In[ ]:


#hidden layer 2
classifier.add(layers.Dense(32,activation='relu'))
# classifier.add(layers.Dropout(0.3))


# In[ ]:


#output layer
classifier.add(layers.Dense(8,activation='sigmoid'))


# In[ ]:


classifier.summary()


# In[ ]:


classifier.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])


# In[ ]:





# In[ ]:


df_train.head()


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train , X_test ,y_train , y_test = train_test_split(df_train_feature,df_train_label,test_size=0.3,random_state=1)


# In[ ]:


keras_result = classifier.fit(X_train,y_train,epochs=50,batch_size=5,validation_data=(X_test,y_test)) 


# In[ ]:


np.mean(keras_result.history['acc'])


# In[ ]:


classifier.fit(df_train_feature,df_train_label,epochs=50,batch_size=5)


# In[ ]:


class_predict = classifier.predict_classes(df_test_feature)


# In[ ]:


class_predict = pd.DataFrame(class_predict,columns={'Cover_Type'})


# In[ ]:


class_predict.head()


# In[ ]:


keras_submission = df_test_label.join(class_predict)

# drop keras_submission


# In[ ]:


keras_submission.head()


# In[ ]:


keras_submission.to_csv('keras_submission',index=False)


# In[ ]:




