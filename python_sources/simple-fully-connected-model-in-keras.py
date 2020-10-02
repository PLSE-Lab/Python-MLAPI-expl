#!/usr/bin/env python
# coding: utf-8

# As it is my first Kernel, I focus on a simple approach with a fully-connected model in keras. 
# Of course, there are many other ways to achieve better results (e.g. CNN), but for beginners (like I am) this could be an easy example.
# Your feedback is highly appreciated!
# 
# 
# **1.) General Imports**
# 
# **1.1) Import Modules**

# In[ ]:


import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from sklearn import preprocessing
import matplotlib.pyplot as plt


# **1.2) Import Data**

# In[ ]:


#import train data in Pandas DataFrame
df_train = pd.read_csv('../input/digit-recognizer/train.csv').astype(float)

#import test data in Pandas DataFrame
df_test = pd.read_csv('../input/digit-recognizer/test.csv').astype(float)


# In[ ]:


#DataFrame df_train: Show first 5 entries
df_train.head()


# **2.) Simple Fully-Connected Neural Network (96.33%)**
# 
# ![](https://storage.googleapis.com/kaggle-datasets/98112/230269/FC.jpg?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1546422464&Signature=aXTRq0bJC4aRnPqG614J9gMRw8D9PA8tmpD7WNISUEM%2FGIZ8Hw%2FLdO5oJKcbA2V7AYFEareLLIy%2BpipKC0pP3XnLKfZP19fWA7HBwjMhGmRjwNF%2F5P7%2BhizHLGdvDc%2FaXjw04BzEjIa%2FZEAeqNwX%2FZ3xPxa38geoDt%2BibDGJGD8lFCEcDkIMtKkhwWkapM7X02MLNwRnImMIiIRf4OuBjx6Ku9C6B7pPNtuy%2FP9QoyPeiSrdMboyNpYC72a0hLZZ7wY%2FiT6Pezju7UCaXNFOiYsa4sw8KDhm8ftCM0R82gGaSofgdXhVKAVsPD%2F%2FUZiW00anv2mCGlFdXkcC%2F%2FIICQ%3D%3D)
# 
# 
# In this example we start already with flattend data.
# 

# **2.1) Data Preprocessing**

# In[ ]:


#Split DataFrame df_train in 2 parts: X (inputs) & y (labels)
#Split & Scale (X_train_flat)
X_train_flat = preprocessing.scale(df_train[df_train.columns[1:]])

#Scale (X_test_flat)
X_test_flat = preprocessing.scale(df_test)

#One-Hot Encoding labels
y_train = to_categorical(df_train[df_train.columns[0]])


# In[ ]:


#Show first 5 rows of training labels (before One-Hot Enconding)
pd.DataFrame(df_train[df_train.columns[0]]).head()


# In[ ]:


#Show first 5 rows of training labels (after One-Hot Enconding)
pd.DataFrame(y_train).head()


# In[ ]:


#Plot one example picture
#Convert flattend data in to 2D array for each image with size of 28 x 28
X_train_2d = X_train_flat.reshape(42000, 28,28)
#Plot image #100
plt.imshow(X_train_2d[100])


# **2.2) Create & Train Model**

# In[ ]:


#Very simple fully-connected (dense) layer model
model = Sequential()
#First layer with 64 units expects input of 784 (28 x 28)
model.add(Dense(units=64, activation='relu', input_dim=784))
#Second layer with 32 units
model.add(Dense(units=32, activation='relu'))
#Output layer with 10 units (for '0' to '9') 
model.add(Dense(units=10, activation='softmax'))

#Compile model
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
#Fit Model (15 epochs, batch-size 64 and validation split of 30%)
history = model.fit(x=X_train_flat, y=y_train, batch_size=64, epochs=15, validation_split=0.3)


# In[ ]:


#Max validation accuracy during training
np.max(history.history['val_acc'])


# **Figure below shows an overfitting during training**
# 
# (there is still bigger potential for improvements)

# In[ ]:


acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label = 'Training')
plt.plot(epochs, val_acc, 'b', label = 'Validierung')
plt.title('Correct Classification Rate training/validation')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Loss training')
plt.plot(epochs, val_loss, 'b', label='Loss Validation')
plt.title('Value of Loss Function training/validation')
plt.legend()


# **2.2) Create & Train Model with Dropout**

# In[ ]:


from keras.layers import Dropout

#Very simple fully-connected (dense) layer model
model = Sequential()
#add dropout
model.add(Dropout(0.2, input_shape=(784,)))
#First layer with 64 units expects input of 784 (28 x 28)
model.add(Dense(units=64, activation='relu'))
#add dropout
model.add(Dropout(0.2))
#Second layer with 32 units
model.add(Dense(units=32, activation='relu'))
#Output layer with 10 units (for '0' to '9') 
model.add(Dense(units=10, activation='softmax'))

#Compile model
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
#Fit Model (15 epochs, batch-size 64 and validation split of 30%)
history = model.fit(x=X_train_flat, y=y_train, batch_size=64, epochs=25, validation_split=0.3)


# In[ ]:


#Max validation accuracy during training
np.max(history.history['val_acc'])


# In[ ]:


acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label = 'Training')
plt.plot(epochs, val_acc, 'b', label = 'Validierung')
plt.title('Correct Classification Rate training/validation')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Loss training')
plt.plot(epochs, val_loss, 'b', label='Loss Validation')
plt.title('Value of Loss Function training/validation')
plt.legend()


# With dropout I could already improve the result (overfitting is reduced & max validation accuracy increased). But a significant higher accuracy (>99%) I could only achieve by different model architecutre (in my case CNN).

# **2.4) Predict Labels for test data**

# In[ ]:


#use trained model predict label (one hot encoded) for test data
y_hat_one_hot = model.predict(X_test_flat)

#convert one-hot encoded values into label values
y_hat = np.argmax(y_hat_one_hot, axis=1)
#write prediction into Pandas DataFrame
y_hat = pd.DataFrame(y_hat, columns=['Label'])
y_hat.index += 1 
y_hat.index.name = 'ImageId'


# In[ ]:


#Show first 5 rows of prediction table
y_hat.head()


# In[ ]:




