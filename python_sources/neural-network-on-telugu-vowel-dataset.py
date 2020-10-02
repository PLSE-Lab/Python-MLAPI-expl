#!/usr/bin/env python
# coding: utf-8

# # <font color='tomato'>Neural Network</font> on Telugu Vowel Dataset

# ## Importing Libraries

# In[ ]:


import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Loading Data

# In[ ]:


df = pd.read_csv("../input/CSV_datasetsix_vowel_dataset_with_class.csv")
df.head()


# ## Splitting data into 'train' and 'test'

# In[ ]:


pix=[]
for i in range(784):
    pix.append('pixel'+str(i))
features=pix
X = df.loc[:, features].values
y = df.loc[:,'class'].values

X_train, X_test, y_train, y_test = train_test_split(
 X, y, test_size = 0.30, random_state = 100)
y_train=y_train.ravel()
y_test=y_test.ravel()


# ## Seeing the Images in the dataset

# In[ ]:


def row2img(data):
    return np.asfarray(data).reshape((28,28))


# In[ ]:


data=X_train[11]
f, ax1 = plt.subplots(1, 1, sharey=True)
f.suptitle('Respective image of X_train[11]', size='20')
ax1.imshow(255-row2img(data), cmap=plt.cm.binary);


# ## Normalize the dataset

# In[ ]:


X_train = tf.keras.utils.normalize(X_train,axis=1)
X_test = tf.keras.utils.normalize(X_test,axis=1)


# ## Building Neural Network

# In[ ]:


model=tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(7,activation=tf.nn.softmax))


# ## Compiling Model

# In[ ]:


model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# In[ ]:


model.fit(X_train,y_train,epochs=50)


# ## Accuracy and Loss

# In[ ]:


_,acc=model.evaluate(X_test,y_test)
print('Accuracy: {}'.format(acc))


# ## <font color='tomato'>Accuracy of the model is: </font><font color='MediumSpringGreen'>84.17</font>

# ## Predict

# In[ ]:


pred=model.predict([X_test])
print('Predicted Label: ',np.argmax(pred[11]))


# In[ ]:


f, (ax1) = plt.subplots(1, 1, sharey=True)
ax1.set_title('Actual Label: '+str(y_test[11]))
ax1.imshow(255-row2img(X_test[11]),cmap=plt.cm.binary);


# ## Conclusion:
# * <pre><font color='navy' face='lucida console'><strong>Since the Neural Networks needs lots of data for the training purpose.
#   So the data that we have is not sufficient to train the model to get better reults when compared to the models like SVM and K-NNC.</strong></font></pre>

# In[ ]:




