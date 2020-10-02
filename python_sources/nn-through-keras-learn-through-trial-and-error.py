#!/usr/bin/env python
# coding: utf-8

# ## Using Neural Networks through Keras

# lets  make it better !**strong text**

# In[1]:


## Importing standard libraries
get_ipython().run_line_magic('pylab', 'inline')
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


## Importing sklearn libraries

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# In[3]:


## Keras Libraries for Neural Networks

from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation
from keras.utils.np_utils import to_categorical


# In[4]:


## Set figure size to 20x10

from pylab import rcParams
rcParams['figure.figsize'] = 10,10


# In[5]:


## Read data from the CSV file

data = pd.read_csv('../input/train.csv')
parent_data = data.copy()    ## Always a good idea to keep a copy of original data
ID = data.pop('id')


# In[6]:


data.shape


# In[7]:


## Since the labels are textual, so we encode them categorically

y = data.pop('species')
y = LabelEncoder().fit(y).transform(y)
print(y.shape)


# In[8]:


## Most of the learning algorithms are prone to feature scaling
## Standardising the data to give zero mean =)

X = StandardScaler().fit(data).transform(data)
print(X.shape)


# In[9]:


## We will be working with categorical crossentropy function
## It is required to further convert the labels into "one-hot" representation

y_cat = to_categorical(y)
print(y_cat.shape)


# In[10]:


model = Sequential()
model.add(Dense(256, input_dim=108, init='uniform', activation='relu'))
model.add(Dense(128, init='normal', activation='sigmoid'))
model.add(Dense(99, activation='softmax'))


# In[11]:


## Developing a layered model for Neural Networks
## Input dimensions should be equal to the number of features
## We used softmax layer to predict a uniform probabilistic distribution of outcomes

model = Sequential()
model.add(Dense(512,input_dim=192,  init='uniform', activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(256, activation='sigmoid'))
model.add(Dropout(0.3))
model.add(Dense(99, activation='softmax'))


# In[12]:


## Error is measured as categorical crossentropy or multiclass logloss
model.compile(loss='categorical_crossentropy',optimizer='rmsprop', metrics = ["accuracy"])


# In[13]:


## Fitting the model on the whole training data
history = model.fit(X,y_cat,batch_size=192,
                    nb_epoch=130,verbose=0, validation_split=0.1)


# In[14]:


min(history.history['val_acc'])


# In[15]:


## Plotting the error with the number of iterations
## With each iteration the error reduces smoothly

plt.plot(history.history['val_acc'],'o-')
#plt.plot(history.history['loss'],'o-')
plt.xlabel('Number of Iterations')
plt.ylabel('Categorical Crossentropy')
plt.title('Train Error vs Number of Iterations')


# In[16]:


test = pd.read_csv('../input/test.csv')


# In[17]:


index = test.pop('id')


# In[18]:


test = StandardScaler().fit(test).transform(test)


# In[19]:


yPred = model.predict_proba(test)


# In[20]:


## Converting the test predictions in a dataframe as depicted by sample submission

yPred = pd.DataFrame(yPred,index=index,columns=sort(parent_data.species.unique()))


# In[21]:


fp = open('submission_nn_kernel.csv','w')
fp.write(yPred.to_csv())


# ---------
# 
# Earlier` we used a 4 layer network but the result came out to be overfitting the test set. We dropped the count of neurones in the network and also restricted the number of layers to 3 so as to keep it simple.
# Instead of submitting each test sample as a one hot vector we submitted each samples as a probabilistic distribution over all the possible outcomes. This "may" help reduce the penalty being exercised by the multiclass logloss thus producing low error on the leaderboard! ;)
# Any suggestions are welcome!

# In[ ]:




