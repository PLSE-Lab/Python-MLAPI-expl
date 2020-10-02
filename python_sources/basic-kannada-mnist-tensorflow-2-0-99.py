#!/usr/bin/env python
# coding: utf-8

# # Kannada MNIST using CNN Tensorflow 2.0
# 
# ### A simple CNN implementation for Kannada MNIST dataset classification
# 
# author : anand_s_m

# In[ ]:


#Imports
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.utils import to_categorical


# In[ ]:


#Lets load the data
df_train = pd.read_csv('/kaggle/input/Kannada-MNIST/train.csv')
df_test = pd.read_csv('/kaggle/input/Kannada-MNIST/test.csv')
#Info on dataframe
df_train.info()


# In[ ]:


#Lets visualize the class and checking for class imbalance
num = df_train.label.value_counts()
sns.barplot(num.index,num)


# In[ ]:


#Seperating all the features and target for training data
train_data = df_train.iloc[:,1:]
train_label = df_train.iloc[:,0]
print(f"train_data shape :{train_data.shape}")
print(f"train_label shape :{train_label.shape}")


# In[ ]:


#Normalize
X = train_data.values / 255.0
X_test = df_test.iloc[:,1:].values / 255.0
y = train_label


# In[ ]:


#Lets take a look at few samples
plt.imshow(X[2].reshape(28,28))
plt.show()   
print(f"label : {y[2]}")


# In[ ]:


#data splitting with 90% on the train set and 10% on validation set
X_train,X_val,y_train,y_val = train_test_split(X,y,test_size = 0.1)


# In[ ]:


#input reshape
input_shape = (-1,28,28,1)
X_train = X_train.reshape(input_shape)
X_val = X_val.reshape(input_shape)


# In[ ]:


#Now let us encode our labels
y_train = to_categorical(y_train)
y_val = to_categorical(y_val)


# In[ ]:


#Now we have categoricaly encoded our labels
print(y_train.shape)


# In[ ]:


#Building the CNN model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(64,(3,3),activation='relu',
                                input_shape = (28,28,1)))
model.add(tf.keras.layers.Conv2D(64,(3,3),activation='relu'))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2)))
model.add(tf.keras.layers.Conv2D(32,(3,3),activation='relu'))
model.add(tf.keras.layers.Conv2D(32,(3,3),activation='relu'))
model.add(tf.keras.layers.Dropout(0.25))
model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128,activation='relu'))
model.add(tf.keras.layers.Dropout(0.25))
model.add(tf.keras.layers.Dense(256,activation='relu'))
model.add(tf.keras.layers.Dropout(0.25))
model.add(tf.keras.layers.Dense(10,activation='softmax'))


# In[ ]:


optimzer = tf.keras.optimizers.Adam(learning_rate=0.01)
model.compile(optimizer='Adam',
             loss='categorical_crossentropy',
             metrics=['acc'])


# In[ ]:


epochs = 10
batch_size = 128
hist = model.fit(X_train,y_train,
                validation_data=(X_val,y_val),
                epochs=epochs,
                batch_size=batch_size).history


# In[ ]:


#lets just evaluate the model
model.evaluate(X_val,y_val)


# In[ ]:


X_test = X_test.reshape(-1,28,28,1)
result = model.predict_classes(X_test)


# In[ ]:


_id = [i for i in range(0,X_test.shape[0])]
sub_df = pd.DataFrame({'id':_id,'label':result})
sub_df.to_csv('submission.csv',index=False)


# In[ ]:


sub_df.head()


# In[ ]:





# In[ ]:




