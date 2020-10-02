#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


df=pd.read_csv("../input/churn-modelling/Churn_Modelling.csv")


# In[ ]:


df.head()


# In[ ]:


df.info()


# In[ ]:


df.describe()


# **Here I separate the dependent and independent features.Exited is the dependent feature**

# In[ ]:


x=df.iloc[:,3:13]
y=df.iloc[:,13]


# In[ ]:


x.head()


# **Here the geography and gender are categorical variables. So I have to change that into numerical value**

# In[ ]:


geo=pd.get_dummies(x['Geography'],drop_first=True)
gen=pd.get_dummies(x['Gender'],drop_first=True)


# In[ ]:


x=x.drop(['Gender','Geography'],axis=1)


# In[ ]:


x=pd.concat([x,geo,gen],axis=1)


# In[ ]:


x.head()


# # Train Test split

# In[ ]:


from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=0)


# # Feature Scaling

# In[ ]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
xtrain=sc.fit_transform(xtrain)
xtest=sc.transform(xtest)


# # Model Development

# In[ ]:


import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout


# **Initialize the ANN**

# In[ ]:


classifier = Sequential()


# **First hidden layer**

# In[ ]:


classifier.add(Dense(units=6,activation='relu',kernel_initializer='he_uniform',input_dim=11))


# **Units is the no of hidden unit in the first layer ,he_uniform is the weight initialization technique for relu activation func and and input_dim is the no of features in the dataset**

# **Second hidden layer**

# In[ ]:


classifier.add(Dense(units=6,activation='relu',kernel_initializer='he_uniform'))


# **Output layer**

# In[ ]:


classifier.add(Dense(units=1,activation='sigmoid',kernel_initializer='glorot_uniform'))


# In[ ]:


classifier.summary()


# **Compile the ANN**

# In[ ]:


classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])


# **Fitting the ANN model to train data**

# In[ ]:


model_history=classifier.fit(xtrain,ytrain,batch_size=10,epochs=100,validation_split=0.33)


# **The accuracy of train is pretty much similar to the validation data without overfitting to the train data**

# **we can also change he_normal,dropouts and more hidden layers**

# In[ ]:


print(model_history.history.keys())


# In[ ]:


plt.plot(model_history.history['accuracy'])
plt.plot(model_history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[ ]:


plt.plot(model_history.history['loss'])
plt.plot(model_history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()


# In[ ]:


ypred=classifier.predict(xtest)
ypred=(ypred>0.5)


# In[ ]:


from sklearn.metrics import confusion_matrix
conf=confusion_matrix(ytest,ypred)


# In[ ]:


print(conf)


# **The true positive and false negative are good in this model**

# In[ ]:


from sklearn.metrics import accuracy_score
acc=accuracy_score(ypred,ytest)
print(acc)


# # Perform Hyper parameter tuning for the model

# In[ ]:


from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

from keras.layers import Dropout,BatchNormalization,Flatten,Dense,Activation
from keras.activations import relu,sigmoid


# In[ ]:


def createmodel(layers,activation):
    model=Sequential()
    
    for i,nodes in enumerate(layers):
        if i==0:
            model.add(Dense(nodes,input_dim=xtrain.shape[1]))
            model.add(Activation(activation))
            model.add(Dropout(0.3))
        else:
            model.add(Dense(nodes))
            model.add(Activation(activation))
            model.add(Dropout(0.3))
            
    model.add(Dense(units=1,activation='sigmoid',kernel_initializer='glorot_uniform'))
    
    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    return model
            
        


# In[ ]:


model=KerasClassifier(build_fn=createmodel,verbose=1)


# In[ ]:


layers=[(20),(40,20),(45,30,15)]
activations=['sigmoid','relu']
param=dict(layers=layers,activation=activations,batch_size=[128,256],epochs=[30])
grid=GridSearchCV(estimator=model,param_grid=param,cv=5)


# In[ ]:


gridresult=grid.fit(xtrain,ytrain)


# In[ ]:


print(gridresult.best_params_)


# In[ ]:


print(gridresult.best_score_)


# **From the above result , we can say that relu activation, batch size of 128, epoch of 30 and layers =(45,30,15) will give higher accuracy score of 85%**

# In[ ]:




