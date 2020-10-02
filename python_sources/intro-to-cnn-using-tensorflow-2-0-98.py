#!/usr/bin/env python
# coding: utf-8

# # Introduction to CNN using Tensorflow 2.0 - MNIST (98%)
# Author: Anand
# 
# 
# Well this is my first kaggle notebook.So which better dataset to start my keggle journey.
# Wish me luck kagglers :)

# In[ ]:


#Lets import the needed Libraries
#Kaggle comes with the latest tensorflow 2.0 version so i don't have to install it :p
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.model_selection import train_test_split
import itertools
tf.__version__


# In[ ]:


#Now lets load the dataset
train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')


# In[ ]:


#Now we have loaded the data now lets see its content
train.head()


# In[ ]:


#Now we can see that the first column labels is the target and rest are the features
X_train = train.iloc[:,1:]
y_train = train.iloc[:,0]
#Same for the test set
X_test = test.iloc[:,:]
train.shape


# In[ ]:


#Now lets just visualize the MNIST class count tp check imbalance
g = sns.countplot(y_train)


# In[ ]:


#We can see that all the classes are around similar range.That's what we needed 
#Now lets the normalize the data to make the pixels within (0 - 1) range
X_train = X_train / 255.0
X_test = X_test / 255.0


# In[ ]:


#In the above section[24] we could see that dataset shape (42000, 785),
#but since we are using Convolution our model expects an input in the shape (height x width x depth)
#Since we are using a black and white image the depth/colour channel should be 1.
#So lets reshape the inputs
X_train = X_train.values.reshape(-1,28,28,1)
X_test = X_test.values.reshape(-1,28,28,1)
X_train.shape


# In[ ]:


#Now lets encode the labels since that what our loss function expects
y_train = tf.keras.utils.to_categorical(y_train)
y_train.shape


# In[ ]:


#Now lets split our dataset for train and validation.90% for training and 10% for validation
random_seed = 42
X_train,X_val,y_train,y_val = train_test_split(X_train,y_train,test_size = 0.1)
input_shape = (28,28,1)


# In[ ]:


#Now let us take a look at sample MNIST data
plt.imshow(X_train[0].reshape(28,28),cmap=plt.set_cmap('gray'))


# ## CNN
# #### Now is the interesting part of building a model.I am just building a very simple Model with two convolution layers.Instead of going with keras API here i am going with latest tensorflow 2.0 framework (which is almost similar to keras :p ). So lets build the model

# In[ ]:


#Building the model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(32,kernel_size = (3,3),activation='relu',input_shape=input_shape))
model.add(tf.keras.layers.Conv2D(64,kernel_size = (3,3),activation='relu'))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128,activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(10,activation='softmax'))


# In[ ]:


#Lets checkout the model summary
model.summary()


# In[ ]:


#Looks okay :) 
#Now lets compile the model
model.compile(optimizer='adam',metrics=['acc'],loss='categorical_crossentropy')


# In[ ]:


epochs = 10
batch_size = 32
history = model.fit(X_train,y_train,
                   epochs=epochs,
                   validation_data=(X_val,y_val),
                   batch_size=batch_size,
                   verbose=1).history


# #### That's a 98.8%+ (99%) accuracy on the validation data.Well i guess that's great for my first try without doing much. :)

# In[ ]:


#Now we can plot how our accuracy and loss went
loss = history['loss']
val_loss = history['val_loss']
epochs = range(1, len(loss)+ 1)
print(loss)
print(epochs)
line1 = plt.plot(epochs,loss,label="Validation/Test loss")
line2 = plt.plot(epochs,val_loss,label="Training loss")
plt.setp(line1,linewidth=2.0,marker='+',markersize="10.0")
plt.setp(line2,linewidth=2.0,marker='4',markersize="10.0")
plt.title("Model Loss")
plt.ylabel("Epocs")
plt.xlabel("Loss")
plt.grid(True)
plt.legend()
plt.show()


# In[ ]:



acc = history['acc']
val_acc = history['val_acc']
line1 = plt.plot(epochs,acc,label="Validation/Test Accuracy")
line2 = plt.plot(epochs,val_acc,label="Training Accuracy")
plt.setp(line1,linewidth=2.0,marker='+',markersize="10.0")
plt.setp(line2,linewidth=2.0,marker='4',markersize="10.0")
plt.title("Model Accuracy")
plt.ylabel("Epocs")
plt.xlabel("Accuracy")
plt.grid(True)
plt.legend()
plt.show()


# In[ ]:


#Check model score
score = model.evaluate(X_val,y_val)
print(f"loss :{score[0]}")
print(f"Accuracy :{score[1]}")


# ### Well 98% + looks great :] 

# ### Lets check the prediction accuracy on test data
# 
# Well this might seem a bit unconventional for kaggle where kagglers mostly stops at the point where they get a great accuracy score on the validation & test data.Well here i am trying to pick a random number from the MNIST test set and predict its result.

# In[ ]:


#lets plot a random image from the test set
rand = np.random.randint(0,len(X_test))
plt.imshow(X_test[rand].reshape(28,28))


# In[ ]:


# 7 thats my favourite number :)
#Now lets see how the model predict
result = model.predict_classes(X_test[rand].reshape(-1,28,28,1))[0]
print(f"Predicted Result : {result}")


# # Wow :)
# 
# #### Well that's it then, i hope i have done good for my first kaggle notebook.If i have done any mistake please comment.
# #### if you find this notebook useful please like :)
# 
