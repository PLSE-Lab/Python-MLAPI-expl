#!/usr/bin/env python
# coding: utf-8

# # CNN - Beginner

# ## Importing Packages & CSV

# In[ ]:


#Importing essential packages
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, Flatten, MaxPool2D
from keras.optimizers import adam, RMSprop
from keras.preprocessing.image import img_to_array, ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau


# In[ ]:


#importing train & test Data
train = pd.read_csv("../input/digit-recognizer/train.csv")
test = pd.read_csv("../input/digit-recognizer/test.csv")
sam_sub = pd.read_csv("../input/digit-recognizer/sample_submission.csv")
train[:1]


# ### Checking for null values

# In[ ]:


print("No. of Null value ",train.isnull().any().sum())
print("No. of Null value ",test.isnull().any().sum())
print(train.shape)
print(test.shape)
#there is no null values


# In[ ]:


#seperating features & target values
y_train = train["label"]
x_train = train.drop("label",axis=1)
y_train[:1]


# In[ ]:


#target variable distribution
sns.countplot(y_train)


# In[ ]:


#scaling values to increase its speed
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
test = scaler.transform(test)


# In[ ]:


#reshaping
x_train = x_train.reshape(-1,28,28,1)
test = test.reshape(-1,28,28,1)


# In[ ]:


#viewing sample value
plt.imshow(x_train[1][:,:,0])
plt.title(y_train[1])


# In[ ]:


#converting target value into categorical array
print(y_train[:1])
y_train = to_categorical(y_train,10)
y_train[:1]


# In[ ]:


X_train, X_test, Y_train, Y_test = train_test_split(x_train, y_train, test_size = 0.1, random_state=7)


# In[ ]:


plt.imshow(X_train[0][:,:,0])
plt.title(Y_train[0])


# Model Used in NN
# * Activation function: 
# 
#   ReLu
#   
#   * ReLU is important because it does not saturate; the gradient is always high (equal to 1) if the neuron activates. As long as it is not a dead neuron, successive updates are fairly effective. ReLU is also very quick to evaluate. 
# 
#   Softmax
# 
#   * The softmax activation function is used in neural networks when we want to build a multi-class classifier which solves the problem of assigning an instance to one class when the number of possible classes is larger than two.
# 
# * Max Pooling
# 
#   * Max pooling is a sample-based discretization process. The objective is to down-sample an input representation (image, hidden-layer output matrix, etc.), reducing its dimensionality and allowing for assumptions to be made about features contained in the sub-regions binned.
# 
# * Padding
# 
#   * Padding is a term relevant to convolutional neural networks as it refers to the amount of pixels added to an image when it is being processed by the kernel of a CNN. For example, if the padding in a CNN is set to zero, then every pixel value that is added will be of value zero.

# In[ ]:


model = Sequential()
model.add(Conv2D(filters=44,kernel_size=(3,3),padding="same",activation="relu",input_shape=(28,28,1)))
model.add(Conv2D(filters=33,kernel_size=(3,3),padding="same",activation="relu"))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(.21))
model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same",activation="relu"))
model.add(Conv2D(filters=55,kernel_size=(3,3),padding="same",activation="relu"))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(.21))

model.add(Flatten())
model.add(Dense(256,activation="relu"))
model.add(Dropout(.20))
model.add(Dense(10,activation="softmax"))


# In[ ]:


opt = RMSprop()
opt1= adam()
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=4, min_lr=0.0001)


# In[ ]:


model.compile(optimizer=opt1,loss="categorical_crossentropy",metrics=["accuracy"])


# In[ ]:


model.fit(X_train,Y_train,batch_size=25,epochs=30,callbacks=[reduce_lr])


# In[ ]:


y_pred = model.predict(X_test)


# In[ ]:


pred = np.argmax(y_pred,axis = 1) 
y_test = np.argmax(Y_test,axis = 1)


# In[ ]:


confusion_matrix(pred,y_test)


# In[ ]:


tst_pred = model.predict(test)
preds = np.argmax(tst_pred,axis = 1)


# In[ ]:


rst = pd.DataFrame(preds,columns=["Label"])
rst["ImageId"] = pd.Series(range(1,(len(preds)+1)))


# In[ ]:


sub = rst[["ImageId","Label"]]
sub.shape


# In[ ]:


sub.to_csv("submission.csv",index=False)


# In[ ]:


sam_sub[:4]


# In[ ]:


sub[:4]


# In[ ]:




