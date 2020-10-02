#!/usr/bin/env python
# coding: utf-8

# ## Introduction
# **This is a continuation of my previous kernel. Link - [A Deep Dive into Atoms and Molecules](https://www.kaggle.com/basu369victor/a-deep-dive-into-atoms-and-molecules). In this kernel I have implemented all types of possible data visualizations with the available datasets, and have also implemented a K-Nearest Nighbour  regressor model for prediction. Please go check it out if you haven't and do not forget to upvote it if you like it.<br><br>**
# **Now I am trying to find out if regression with Deep Neural networks could help us achieving a better and more accurate prediction.. Lets see what happens next..**

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from keras import backend as K
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from keras.layers import Dense,Input,Dropout
from keras.models import Model
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from keras.models import load_model
import matplotlib.pyplot as plt
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# In[ ]:


train.head()


# In[ ]:


train.drop("molecule_name", axis=1, inplace=True)
test.drop("molecule_name", axis=1, inplace=True)


# In[ ]:


test_id = test['id']
train.drop("id", axis=1, inplace=True)
test.drop("id", axis=1, inplace=True)


# In[ ]:


train_type = pd.get_dummies(train['type'])
test_type = pd.get_dummies(test['type'])


# In[ ]:


train['type'] = train['type'].astype("category").cat.codes
test['type'] = test['type'].astype("category").cat.codes


# In[ ]:


#train_new = pd.concat([train, train_type], axis=1)
#train_new.drop("type", axis=1, inplace=True)
#test_new = pd.concat([test, test_type], axis=1)
#test_new.drop("type", axis=1, inplace=True)


# In[ ]:


train.head()


# In[ ]:


y = train["scalar_coupling_constant"]
train.drop("scalar_coupling_constant", axis=1, inplace=True)
X = train


# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=42)


# In[ ]:


K.clear_session()
def RegressionModel(in_):
    model = Dense(256,kernel_initializer='normal',activation="relu")(in_)
    model = Dense(128,kernel_initializer='normal',activation="relu")(model)
    model = Dense(64,kernel_initializer='normal',activation="relu")(model)
    model = Dense(32,kernel_initializer='normal',activation="relu")(model)
    
    model = Dense(1,kernel_initializer='normal',activation="linear")(model)
    
    return model


# In[ ]:


Input_Sample = Input(shape=(x_train.shape[1],))
Output_ = RegressionModel(Input_Sample)
EnhanceRegression = Model(inputs=Input_Sample, outputs=Output_)


# In[ ]:


EnhanceRegression.compile(loss='mean_squared_error', optimizer='adam',metrics=['mse','mae'])
EnhanceRegression.summary()


# In[ ]:


ES = EarlyStopping(monitor='val_loss', min_delta=0.5, patience=200, verbose=1, mode='auto', baseline=None,
                              restore_best_weights=False)
MC = ModelCheckpoint('best_model.h5', monitor='val_acc', mode='auto', verbose=1, save_best_only=True)


# In[ ]:


num_epochs =300
num_batch_size = 1080
ModelHistory = EnhanceRegression.fit(x_train, y_train, batch_size=num_batch_size, epochs=num_epochs, 
                                    validation_data=(x_test, y_test),
                                     callbacks = [ES,MC],
                                    verbose=1)


# In[ ]:


#Loss Curves
plt.figure(figsize=[20,9])
plt.plot(ModelHistory.history['loss'], 'r')
plt.plot(ModelHistory.history['val_loss'], 'b')
plt.legend(['Training Loss','Validation Loss'])
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Curves')


# In[ ]:


#Accuracy Curves
plt.figure(figsize=[20,9])
plt.plot(ModelHistory.history['mean_absolute_error'], 'r')
plt.plot(ModelHistory.history['val_mean_absolute_error'], 'b')
plt.legend(['Training Accuracy','Validation Accuracy'])
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy Curves')


# In[ ]:


#saved_model = load_model('best_model.h5')


# In[ ]:


y_pred_test = EnhanceRegression.predict(test)


# In[ ]:


prediction = y_pred_test.flatten()
prediction 


# In[ ]:


my_submission = pd.DataFrame({'id':test_id ,'scalar_coupling_constant': prediction })
my_submission.to_csv('SubmissionVictorX.csv', index=False)

