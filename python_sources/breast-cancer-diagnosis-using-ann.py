#!/usr/bin/env python
# coding: utf-8

# Attribute Information:
# 
# 1) ID number
# 2) Diagnosis (M = malignant, B = benign)
# 3-32)
# 
# Ten real-valued features are computed for each cell nucleus:
# 
# a) radius (mean of distances from center to points on the perimeter)
# 
# b) texture (standard deviation of gray-scale values)
# 
# c) perimeter
# 
# d) area
# 
# e) smoothness (local variation in radius lengths)
# 
# f) compactness (perimeter^2 / area - 1.0)
# 
# g) concavity (severity of concave portions of the contour)
# 
# h) concave points (number of concave portions of the contour)
# 
# i) symmetry
# 
# j) fractal dimension ("coastline approximation" - 1)
# 

# In[ ]:


#importing the libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


# In[ ]:


#reading the data
data=pd.read_csv("../input/breast-cancer-wisconsin-data/data.csv")


# In[ ]:


#understanding the data
data.head()


# In[ ]:


#deleting unwanted columns
drop_cols=["id","Unnamed: 32"]
data.drop(drop_cols,axis=1,inplace=True)


# In[ ]:


data.columns


# In[ ]:


#checking the null values 
data.isna().sum()


# In[ ]:


#and converting M to 1 and B to 0 
data['diagnosis']=[1 if x=='M' else 0 for x in data['diagnosis']]


# In[ ]:


#checking the distribution of the target variable
data.diagnosis.value_counts()


# In[ ]:


#separating the dependent and independent features
y=data.diagnosis
X=data
X.drop("diagnosis",axis=1,inplace=True)


# In[ ]:


X.head()


# In[ ]:


#spliting the data into train and test
train_X,test_X,train_y,test_y=train_test_split(X,y,test_size=0.3,random_state=123)
print(train_X.shape)
print(test_X.shape)
print(train_y.shape)
print(test_y.shape)


# In[ ]:


#standardizing the data
std=StandardScaler()
std.fit(train_X)
train_X=pd.DataFrame(std.transform(train_X),index=train_X.index)
test_X=pd.DataFrame(std.transform(test_X),index=test_X.index)


# In[ ]:


train_X.head()


# In[ ]:


#importing the keras and other libraries 
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras import regularizers,optimizers
from keras.callbacks import EarlyStopping,ReduceLROnPlateau


# In[ ]:


#building the neural network...
#Defining the Optimizer
adam=keras.optimizers.adam(lr=0.001,decay=0.0005)

## Just a way to define neural nets. There are two ways sequential and functional
## Sequential model lets you add neural net layers one after another by calling function
model=Sequential()

## Adding layers sequentially one by one...
## Notice our data has 30 input columns which goes into as the "input_shape" parameter
model.add(Dense(16,input_shape=(30,)))

model.add(Dense(8,init="uniform",activation="relu"))

model.add(Dense(4,init="uniform",activation="relu"))

## Notice the use of l2 regularizer
model.add(Dense(1,activation="sigmoid",kernel_regularizer=regularizers.l2()))

## Callbacks
earlystopper = EarlyStopping(monitor='val_loss', patience=10)
reduce_lr = ReduceLROnPlateau(patience=5, verbose=1)

#compiling our model and defining the loss function
model.compile(optimizer=adam,loss="binary_crossentropy",metrics=["accuracy"])

#training the neural nets
history=model.fit(train_X,train_y,validation_data=(test_X,test_y),epochs=200,batch_size=50,callbacks=[earlystopper,reduce_lr])


# In[ ]:


train_acc = history.history['accuracy']
train_loss = history.history['loss']

val_acc = history.history['val_accuracy']
val_loss = history.history['val_loss']

from matplotlib import pyplot as plt #plt is a visualization module in matplotlib.  
get_ipython().run_line_magic('matplotlib', 'inline')
plt.figure(figsize=(20,5))
plt.subplot(1,2,1)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.plot(train_loss)
plt.plot(val_loss)

plt.subplot(1,2,2)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.plot(train_acc)
plt.plot(val_acc)


# In[ ]:


# USING SKLEARN MLP CLASSIFIER WITH GRID SEARCH
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier

classifier=MLPClassifier(activation="logistic",random_state=123)

param= {"batch_size" : [16, 32, 64, 128],
           "hidden_layer_sizes" : [(11,), (15,), (19,),(21,)],
           "max_iter" : [50, 100, 150, 200]}

grid=GridSearchCV(estimator=classifier,param_grid=param,cv=5,n_jobs=1)
grid.fit(train_X,train_y)
grid.best_estimator_


# In[ ]:


train_preds=grid.best_estimator_.predict(train_X)
test_preds=grid.best_estimator_.predict(test_X)


# In[ ]:


from sklearn.metrics import accuracy_score
print("Train Accuracy  :  ",accuracy_score(train_y,train_preds))
print("Test Accuracy   :  ",accuracy_score(test_y,test_preds))

