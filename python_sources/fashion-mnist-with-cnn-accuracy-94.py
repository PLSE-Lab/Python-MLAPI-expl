#!/usr/bin/env python
# coding: utf-8

# #### I imported necessary libraries

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPool2D
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# #### I loaded the train dataset and test dataset

# In[ ]:


train_dataset=pd.read_csv("../input/fashionmnist/fashion-mnist_train.csv")
test_dataset=pd.read_csv("../input/fashionmnist/fashion-mnist_test.csv")


# #### I observed the labels' numbers

# In[ ]:


sns.countplot(train_dataset.label)
plt.title("Count Plot")
plt.xlabel("Labels")
plt.ylabel("Count")
plt.show()


# #### I converted csv file to arrray

# In[ ]:


x_train=train_dataset.iloc[:,1:].values
y_train=train_dataset.iloc[:,0].values
x_test=test_dataset.iloc[:,1:].values
y_test=test_dataset.iloc[:,0].values


# In[ ]:


print("x train :",x_train.shape)
print("y train :",y_train.shape)
print("x test :",x_test.shape)
print("y test :",y_test.shape)


# #### I reshaped the array's shape four dimension because CNN algorithm can work that

# In[ ]:


x_train=x_train.reshape(x_train.shape[0],28,28,1)
x_test=x_test.reshape(x_test.shape[0],28,28,1)
print("new x train shape :",x_train.shape)
print("new x test shape :",x_test.shape)


# #### I wanted to see an image

# In[ ]:


img=x_train[48]
plt.imshow(img.reshape(28,28))
plt.title(y_train[48])
plt.axis("off")
plt.show()


# #### I categorized the labels

# In[ ]:


y_train=to_categorical(y_train,num_classes=10)
y_test=to_categorical(y_test,num_classes=10)
print("new y train shape :",y_train.shape)
print("new y test shape :",y_test.shape)


# #### I normalized pixels of images

# In[ ]:


x_train=x_train / 255.0
x_test =x_test / 255.0


# # CONVOLUTIONAL NEURAL NETWORK

# #### I created my CNN model

# In[ ]:


model=Sequential()

model.add(Conv2D(filters=100,kernel_size=(3,3),padding="same",activation="relu",input_shape=(28,28,1)))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.4))

model.add(Conv2D(filters=100,kernel_size=(3,3),padding="same",activation="relu"))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.4))

model.add(Flatten())
model.add(Dense(206,activation="relu"))
model.add(Dropout(0.4))
model.add(Dense(103,activation="relu"))
model.add(Dropout(0.4))
model.add(Dense(10,activation="softmax"))


# In[ ]:


model.summary()


# In[ ]:


model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])


# #### I started to work my model

# In[ ]:


hist=model.fit(x_train,y_train,batch_size=250,epochs=100,validation_data=(x_test,y_test))


# In[ ]:


print(hist.history.keys())


# In[ ]:


plt.plot(hist.history["loss"],color="green",label="Train Loss")
plt.plot(hist.history["val_loss"],color="red",label="Validation Loss")
plt.legend()
plt.title("Loss Plot")
plt.xlabel("Number of Epochs")
plt.ylabel("Loss Values")
plt.grid()
plt.show()


# In[ ]:


plt.plot(hist.history["accuracy"],color="green",label="Train Accuracy")
plt.plot(hist.history["val_accuracy"],color="red",label="Validation Accuracy")
plt.legend()
plt.title("Accuracy Plot")
plt.xlabel("Number of Epochs")
plt.ylabel("Accuracy Values")
plt.grid()
plt.show()


# In[ ]:


prediction=model.predict(x_test)


# In[ ]:


predicted_classes=np.argmax(prediction,axis=1)
y_true=np.argmax(y_test,axis=1)


# In[ ]:


print("predicted classes shape :",predicted_classes.shape)
print("y true shape :",y_true.shape)


# In[ ]:


cfm=confusion_matrix(y_true,predicted_classes)
f,ax=plt.subplots(figsize=(10,10))
sns.heatmap(cfm,annot=True,cmap="coolwarm",linecolor="black",linewidths=1,fmt=".0f",ax=ax)
plt.xlabel("Real Values")
plt.ylabel("Predicted Values")
plt.show()

