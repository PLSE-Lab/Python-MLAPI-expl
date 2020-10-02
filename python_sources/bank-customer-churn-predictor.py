#!/usr/bin/env python
# coding: utf-8

# **Data Preprocessing

# In[ ]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 


# Importing the data

# In[ ]:


dataset = pd.read_csv('../input/churn-predictions-personal/Churn_Predictions.csv')


# In[ ]:


dataset.head()


# Assigning Dependent and Independent Variables

# In[ ]:


x= dataset.iloc[:, 3:13]
y= dataset.iloc[: ,13]


# Creating Dummy Variables for 

# In[ ]:


geography = pd.get_dummies(x["Geography"], drop_first= True)
gender = pd.get_dummies(x["Gender"], drop_first=True)


# COoncatinating DUmmy Variables

# In[ ]:


x=pd.concat([x,geography,gender], axis=1)


# In[ ]:


x.head()


# Deleting the orginal geography and gender column

# In[ ]:


x=x.drop(['Geography','Gender'],axis=1)


# In[ ]:


x.head()


# Splitting the training and testing data

# In[ ]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2 , random_state =0)


# Feature Scaling 

# In[ ]:


from sklearn.preprocessing import StandardScaler
sc= StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)


# Building the Artificial Neural Network

# Importing Keras and other relevant libraries

# In[ ]:


from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding, Flatten, LeakyReLU, BatchNormalization, Dropout
from keras.activations import relu, sigmoid


# Hyperparameter tuning

# In[ ]:


def create_model(layers, activation):
    model = Sequential()
    for i, nodes in enumerate(layers):
        if i==0:
            model.add(Dense(nodes,input_dim=x_train.shape[1]))
            model.add(Activation(activation))
            model.add(Dropout(0.3))
        else:
            model.add(Dense(nodes))
            model.add(Activation(activation))
            model.add(Dropout(0.3))
            
    model.add(Dense(units = 1, kernel_initializer= 'glorot_uniform', activation = 'sigmoid')) # Note: no activation beyond this point
    
    model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])
    return model
    
model = KerasClassifier(build_fn=create_model, verbose=0)


layers = [(20,), (40, 20), (45, 30, 15)]
activations = ['sigmoid', 'relu']
param_grid = dict(layers=layers, activation=activations, batch_size = [128, 256], epochs=[30])
grid = GridSearchCV(estimator=model, param_grid=param_grid,cv=5)


# In[ ]:


grid_result = grid.fit(x_train, y_train)


# To identify the best parameter

# In[ ]:


print(grid_result.best_score_,grid_result.best_params_)


# Predicting the test set results

# In[ ]:


y_pred= grid.predict(x_test)
y_pred = (y_pred>0.5)


# Making the confusion Matrix
# 

# In[ ]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
cm


# Calcuting the accuracy

# In[ ]:


from sklearn.metrics import accuracy_score
score = accuracy_score(y_test, y_pred)
score

