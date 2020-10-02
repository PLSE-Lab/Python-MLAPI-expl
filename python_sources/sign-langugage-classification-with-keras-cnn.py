#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#libraries
import numpy as np # linear algebra
import keras
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

import os
print(os.listdir("../input/Sign-language-digits-dataset"))


# In[ ]:


# load data
X = np.load("../input/Sign-language-digits-dataset/X.npy")
Y = np.load("../input/Sign-language-digits-dataset/Y.npy")

print("Samples :", X.shape[0])


# In[ ]:


#sample image
plt.imshow(X[345], cmap = "gray")
plt.show()


# In[ ]:


#split train and test 
x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size = .33, shuffle = True)

x_train =  x_train.reshape(-1,64,64,1)
x_test =  x_test.reshape(-1,64,64,1)

print("x_train shape:", x_train.shape)
print("x_test shape:", x_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)


# **Create Model**

# In[ ]:


batch_size = 128
epoch = 20
input_shape = (64,64,1)
num_classes = 10

model = Sequential()


# In[ ]:


model.add(Conv2D(64,(4,4), input_shape = input_shape))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size = (4,4)))

model.add(Conv2D(64,(5,5)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size = (4,4)))

model.add(Flatten())

model.add(Dense(128,activation = "relu"))
model.add(Dropout(0.25))

model.add(Dense(num_classes, activation = "softmax"))


model.summary()


# In[ ]:


#compile
model.compile(loss = keras.losses.categorical_crossentropy,
             optimizer = keras.optimizers.Adadelta(),
             metrics = ["accuracy"])
#fit
model.fit(x_train,y_train,
         batch_size = batch_size,
         epochs = epoch,
         verbose = 1,
         validation_data = (x_test,y_test))


# In[ ]:


score = model.evaluate(x_test, y_test, verbose = 0)

print("Test Loss: ", score[0])
print("Test Accucary: ", score[1])


# In[ ]:


#save model
model.save("..your_path/your_model.h5")

#load model
model_test = load_model("..your_path/your_model.h5)


# **Trying  Model**

# In[ ]:


classes = ["9", "0", "7", "6", "1", "0", "4","3", "2", "5"]

#index for test data 0 ~ 680
index = 254
plt.imshow(x_test[index].reshape(64,64), cmap = "gray")
y_test[index]


# In[ ]:


#predict
test = x_test[index].reshape(1,64,64,1)
pre = model.predict(test, batch_size = 1)
#pre = model_test(test, batch_size = 1) with loaded model

print("Prediction: ", np.round(pre, 0))
print("Real value: ", y_test[index])
print("Number: ",classes[np.argmax(pre)])

