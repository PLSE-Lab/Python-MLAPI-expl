#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import tensorflow as tf
import pandas as pd
import sklearn
import matplotlib.pyplot as plt


# In[ ]:


## import the data

df = pd.read_csv('../input/heart-diseases-data/heart_deases_data.csv',delim_whitespace=True)


# In[ ]:


df.head()


# In[ ]:


correlation = df.corr()


# In[ ]:


correlation


# In[ ]:


## plot correlation


# In[ ]:


import seaborn as sns
sns.heatmap(correlation)


# In[ ]:


## extract the target from the data
## this time target is three 
## 1) chest_pain_type
## 2) depression
## 3) class


# In[ ]:


X = df.drop('class',axis=1)


# In[ ]:


X.head()


# In[ ]:


Y = df[['class']]


# In[ ]:


Y.head()


# In[ ]:


## normalize the feature matrix


# In[ ]:


#  def normalize(df):
#      return (df-df.mean())/df.std()


# In[ ]:


# X = normalize(X)


# In[ ]:


X ## after normalization


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size = .2)


# In[ ]:


x_train.head()


# In[ ]:


x_test.head()


# In[ ]:


y_test.head()


# In[ ]:


y_train.head()


# In[ ]:


print (x_train.shape)
print (x_test.shape)
print (y_train.shape)
print (y_test.shape)


# In[ ]:


n_col = x_train.shape[1]  ## find th column number


# In[ ]:


n_col


# In[ ]:


from keras.layers import Input
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Dropout
from keras.models import Sequential
from keras.layers import Dense


# In[ ]:


def neural_net():
    model = Sequential()
    model.add(Dense(512, input_dim=n_col, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(.2))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(.2))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(.2))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(.2))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(.2))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(.2))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(.2))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    # Compile model
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# In[ ]:


model = neural_net()


# In[ ]:


model.summary()


# In[ ]:


history = model.fit(x_train, y_train, epochs=300, batch_size=10)


# In[ ]:


print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()


# In[ ]:


loss,accuracy = model.evaluate(x_test,y_test)


# In[ ]:


print ("Accuracy "+str(accuracy*100)+"%")


# In[ ]:


predict = model.predict(x_test)


# In[ ]:


predict


# # from probability to direct predicted  ans

# In[ ]:


result=[]
for item in predict:
    result.append(np.argmax(item))


# ## converting into integer

# In[ ]:


y_test = y_test.astype('int')


# In[ ]:


y_test


# In[ ]:


from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import confusion_matrix


# In[ ]:


class_confusion_matrix = confusion_matrix(result,y_test)


# In[ ]:


class_confusion_matrix


# In[ ]:


sns.heatmap(class_confusion_matrix)


# In[ ]:




