#!/usr/bin/env python
# coding: utf-8

# # Convolutional MNIST classification

# If you are starting with Deeplearning, solving MNIST classification problem is the "Hello World" to deep-learning! This is an attempt to put in practice follwoing neural networks using MNIST dataset:
# * Convolution neural networks (CNN)
# * VGG-16
# * AlexNet
# 
# First, let us implement CNN on MNIST classification problem,
# 
# 

# ### Required libraries
# Here, _numpy_ and _pandas_ will help us handle simple mathematical operations and dataframe manipulations. 
# In addition, we will use _tensofrlow.keras_ to define, train, evaluate, and moinotor CNN model. In addition, we will also use it for making predicitons for submission to the KAGGLE'S board.  

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import Flatten, Conv2D, MaxPooling2D
from sklearn.model_selection import train_test_split
import os

import matplotlib.pyplot as plt


# Following line of code check all the files available in "../input/Kannada-MNIST/". i.e.
# We have following files:
# * train.csv (comprises of 60,000 hand-written digits)
# * test.csv (comprises of 10,000 hand-written digits), and 
# * sample_submission.csv ( a submission format comprising of dummy list of ids and labels. We need to replace, the entries with our predictions after designing and training our MNIST-CNN model.)
# 

# In[ ]:


print(os.listdir("../input/Kannada-MNIST/"))


# Below line of code reads the _train.csv_ and _test.csv_ data files and stores them in dataframes i.e. _df_train_ and _df_test_. 

# In[ ]:


df_train = pd.read_csv("../input/Kannada-MNIST/train.csv")
df_test = pd.read_csv("../input/Kannada-MNIST/test.csv")


# Let us check size of train and test data. Train and test data have 785 columns with 60K and 5K entires.

# In[ ]:


print(df_train.shape)
print(df_test.shape)


# Here, checking first 5 rows of train and test dataset. Train dataset has the _label_ in the 1st column which we will use as target label while training. 

# In[ ]:


df_train.head()


# In[ ]:


df_test.head()


# Let us store the pixels columns in variable "X" . and store the label in variable "y". Additionally, let us store the test pixels in "X_test_actual"; this we will use it to predict. 

# In[ ]:


X = df_train.iloc[:,1:]
y = df_train.iloc[:,0]
X_test_actual = df_test.iloc[:,1:]


# ## Sample digits
# We saw both train and test data has 784 pixels. Each row entry here, represents a digit. i.e. A single digit has 28 x 28 pixel distributed in 2D plane. However, for analysis purpose the pixels were flattened into 1D array i.e. 28 x 28 = 784 pixels ( columns). 
# 
# Let us visualize some of the hand-written digitized pixels. 
# 
# 

# In[ ]:


data = np.matrix(X) # Convert the dataframe to matrix for visualization


# We have use numpy.reshape() function to reshape the 784 1-D array to 2-D matrix.

# In[ ]:


print(X)# 1-D matrix


# In[ ]:


print(data) # 2-D matrix


# Let us see the kannada words stored in the following rows out of 60000:
# * 1st row kannada digit
# * 25th row Kannada digit
# * 50th row kannada digit

# In[ ]:


img = data[0].reshape(28,28)
plt.imshow(img, cmap="gray")


# In[ ]:


img = data[24].reshape(28,28)
plt.imshow(img, cmap="gray")


# In[ ]:


img = data[49].reshape(28,28)
plt.imshow(img, cmap="gray")


# * **Given, no literal awareness about kannada digits; it is not possible to recognize the digits 1st, 25th, and 50th represents.** However, we do have the lables from the dataframe df_train. Let us print the 1st, 25th, and 50th row labels showing the actual numeric digits these image represent:
# 

# In[ ]:


print(df_train.iloc[[0,24,49],0])


# NOTE: From the above, it can be concluded that the images shows earlier from the pixels of 1st, 25th, and 50th rows are 0, 4, and 9 respectively. 

# ## Data prepration

# In[ ]:


#X_train.head()
#_train
#X_test
#len(X_train)


# In[ ]:


X = X.to_numpy().reshape(len(X), 28, 28,1).astype('float32')
X_test_actual = X_test_actual.to_numpy().reshape(len(X_test_actual), 28, 28, 1).astype('float32')


# In[ ]:


X = X/255
X_test_actual = X_test_actual/255


# In[ ]:


n_classes=10
y = to_categorical(y, n_classes)


# In[ ]:


X_train, X_test, y_train,  y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# ## Architecture of neural netrowk 

# In[ ]:


model = Sequential()

model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(28,28,1)))

model.add(Conv2D(32, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(128, activation='relu'))
#model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(n_classes, activation='softmax'))


# In[ ]:


model.summary()


# In[ ]:


model.compile(loss='categorical_crossentropy',
             optimizer='nadam',
             metrics=['accuracy'])


# In[ ]:


len(X_test)


# In[ ]:


history = model.fit(X_train, 
                    y_train, 
                    batch_size=128, 
                    epochs=100,
                    verbose=1,
                    validation_data=(X_test, y_test)
                   )


# In[ ]:


import matplotlib.pyplot as plt

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()


# In[ ]:


plt.clf()                                              

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()


# In[ ]:


predictions = model.predict(X_test_actual)


# In[ ]:


submission_df = df_test.iloc[:,0]


# In[ ]:


submission_df.head()


# In[ ]:


data_submission = pd.read_csv("../input/Kannada-MNIST/sample_submission.csv")


# In[ ]:


y_pre=model.predict(X_test_actual)     ##making prediction
y_pre=np.argmax(y_pre,axis=1) ##changing the prediction intro labels


# In[ ]:


data_submission['label']=y_pre
data_submission.to_csv('submission.csv',index=False)


# In[ ]:


data_submission.head()

