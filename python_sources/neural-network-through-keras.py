#!/usr/bin/env python
# coding: utf-8

# # Leaf Classification - Forked
# ## Using Neural Networks through Keras

# __author__ : Najeeb Khan, Yasir Mir, Zafarullah Mahmood
# 
# __team__ : artificial_stuPiDity
# 
# __institution__ : Jamia Millia Islamia
# 
# __email__ : najeeb.khan96@gmail.com

# In[ ]:


## Importing standard libraries

get_ipython().run_line_magic('pylab', 'inline')
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


## Importing sklearn libraries

from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelEncoder


# In[ ]:


## Keras Libraries for Neural Networks

from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation
from keras.utils.np_utils import to_categorical


# In[ ]:


## Set figure size to 20x10

from pylab import rcParams
rcParams['figure.figsize'] = 10,10


# In[ ]:


## Read data from the CSV file

data = pd.read_csv('../input/train.csv')
parent_data = data.copy()    ## Always a good idea to keep a copy of original data
ID = data.pop('id')


# In[ ]:


data.shape


# In[ ]:


## Since the labels are textual, so we encode them categorically

y = data.pop('species')
y = LabelEncoder().fit(y).transform(y)
print(y.shape)


# In[ ]:


## Most of the learning algorithms are prone to feature scaling
## Standardising the data to give zero mean =)

X = StandardScaler().fit(data).transform(data)
print(X.shape)


# In[ ]:


## We will be working with categorical crossentropy function
## It is required to further convert the labels into "one-hot" representation

y_cat = to_categorical(y)
print(y_cat.shape)


# In[ ]:


## Developing a layered model for Neural Networks
## Input dimensions should be equal to the number of features
## We used softmax layer to predict a uniform probabilistic distribution of outcomes

model = Sequential()
model.add(Dense(1024,input_dim=192,  init='uniform', activation='relu'))
#model.add(Dropout(0.3))
model.add(Dense(99, activation='softmax'))


# In[ ]:


## Error is measured as categorical crossentropy or multiclass logloss
model.compile(loss='categorical_crossentropy',optimizer='adam', metrics = ["accuracy"])


# In[ ]:


## Fitting the model on the whole training data
history = model.fit(X,y_cat,batch_size=32,
                    nb_epoch=400,verbose=0)


# In[ ]:


#validation:

#scores = pd.DataFrame(history.history)
#min(scores['val_loss'])
#scores.loc[20:,["loss", "val_loss"]].plot()


# In[ ]:


test = pd.read_csv('../input/test.csv')


# In[ ]:


index = test.pop('id')


# In[ ]:


test = StandardScaler().fit(test).transform(test)


# In[ ]:


yPred = model.predict_proba(test)


# In[ ]:


## Converting the test predictions in a dataframe as depicted by sample submission

yPred = pd.DataFrame(yPred,index=index,columns=sort(parent_data.species.unique()))


# In[ ]:


fp = open('submission_nn_kernel.csv','w')
fp.write(yPred.to_csv())


# ---------
# 
# Earlier` we used a 4 layer network but the result came out to be overfitting the test set. We dropped the count of neurones in the network and also restricted the number of layers to 3 so as to keep it simple.
# Instead of submitting each test sample as a one hot vector we submitted each samples as a probabilistic distribution over all the possible outcomes. This "may" help reduce the penalty being exercised by the multiclass logloss thus producing low error on the leaderboard! ;)
# Any suggestions are welcome!

# In[ ]:




