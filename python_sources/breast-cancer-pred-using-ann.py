#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


#import tensorflow 
import tensorflow as tf


# In[ ]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df = pd.read_csv("../input/breast-cancer-prediction-dataset/Breast_cancer_data.csv", delimiter=",")


# In[ ]:


df


# In[ ]:


#we have to predict from the given parameters if the patient is in danger or not
# in the diagnosis column,
#0 - not in danger and 1 - in danger  


# In[ ]:


#we'll check the correlations between all the columns in the heatmap to get a better understanding of it
import seaborn as sns

#putting annotation = true let's us see the values of correlations
sns.heatmap(df.corr(),annot= True)


# In[ ]:


#every parameter has plays a role in determining our diagnosis
# now we will split our data into X (the parameters) and y(diagnosis)

X = df.drop('diagnosis',axis=1).values
y = df['diagnosis'].values


# In[ ]:


#we'll split our model into training and test set

from sklearn.model_selection import train_test_split

#we'll give 80% to ur training set and remaining 20% to test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[ ]:


#it is necessary to scale our data 

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()


# In[ ]:


X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 5 indicates the number of columns also intput parameters for our model
X_train.shape


# In[ ]:


from keras.models import Sequential
from keras.layers import Dense,Dropout


# In[ ]:


#we created a sequential model for our network
model = Sequential()


# In[ ]:


model.add(Dense(units = 5,activation = 'relu'))
#using sigmoid function because our diagnosis can be either 0 or 1
model.add(Dense(units=1,activation='sigmoid'))

#we use binary_cross_entropy because we have to predict if the patient is in danger (1) or not (0)
model.compile(optimizer='adam',loss='binary_crossentropy')


# In[ ]:


#from tensorflow.keras.callbacks import EarlyStopping

#early_stop = EarlyStopping(monitor='val_loss',mode='min',verbose=1,patience=20)

model.fit(X_train,y_train,epochs=40,validation_data=(X_test,y_test))


# In[ ]:


loss = pd.DataFrame(model.history.history)

loss.plot()


# In[ ]:


from sklearn.metrics import classification_report,confusion_matrix

#predict the diagnosis of our training set on test set
pred = model.predict_classes(X_test)

print(classification_report(y_test,pred))

print(confusion_matrix(y_test,pred))


# In[ ]:





# In[ ]:




