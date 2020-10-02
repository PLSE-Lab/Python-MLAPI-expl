#!/usr/bin/env python
# coding: utf-8

# **Import all Necessary Libraries**

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import Conv2D, Lambda
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras.utils.np_utils import to_categorical
from keras.datasets import mnist
from keras.layers.advanced_activations import PReLU


# **Load Train and Test Data**

# In[ ]:


(X_train,y_train),(X_test,y_test)=mnist.load_data()
X_train=X_train.reshape(X_train.shape[0],28,28,1).astype('float32')
X_test=X_test.reshape(X_test.shape[0],28,28,1).astype('float32')
print(X_train.shape)
print(X_test.shape)


# **Data Visualization**

# In[ ]:


X_train_ = X_train.reshape(X_train.shape[0], 28, 28)

for i in range(0, 3):
    plt.subplot(330 + (i+1))
    plt.imshow(X_train_[i], cmap=plt.get_cmap('gray'))
    plt.title(y_train[i]);


# **One-Hot Encoding**

# In[ ]:


y_train=to_categorical(y_train)
y_test=to_categorical(y_test)
num_classes=y_test.shape[1]


# **Data Augmentation**

# In[ ]:


gen=image.ImageDataGenerator()
batches=gen.flow(X_train,y_train,batch_size=64)


# **Normalization**

# In[ ]:


mean=np.mean(X_train)
std=np.std(X_train)

def standardize(x):
    return (x-mean)/std


# **Model Definition**

# In[ ]:


def model():
    model=Sequential()
    model.add(Lambda(standardize,input_shape=(28,28,1)))
    model.add(Conv2D(64,(3,3),activation="linear"))
    model.add(PReLU())
    model.add(Conv2D(64,(3,3),activation="linear"))
    model.add(PReLU())
    
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(BatchNormalization())
    model.add(Conv2D(128,(3,3),activation="linear"))
    model.add(PReLU())
    model.add(Conv2D(128,(3,3),activation="linear"))
    model.add(PReLU())
    
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(BatchNormalization())
    model.add(Conv2D(256,(3,3),activation="linear"))
    model.add(PReLU())
    
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    model.add(Flatten())
    model.add(BatchNormalization())
    model.add(Dense(512,activation="linear"))
    model.add(PReLU())
    model.add(Dense(10,activation="softmax"))
    
    model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=["accuracy"])
    model.fit_generator(generator=batches,steps_per_epoch=batches.n,epochs=3)
    return model


# **Model Training**

# In[ ]:


model=model()


# **Model Evaluation**

# In[ ]:


score=model.evaluate(X_test,y_test,verbose=0)
print("CNN Error:%.2f%%" %(100-score[1]*100))


# **Prediciting the Outputs**

# In[ ]:


X_test=pd.read_csv('../input/test.csv')
X_test=X_test.values.reshape(X_test.shape[0],28,28,1)
preds=model.predict_classes(X_test,verbose=1)
model.save('digit_recognizer.h5')


# **Function to Submit Results to Kaggle**

# In[ ]:


def write_preds(preds,fname):
    pd.DataFrame({"ImageId":list(range(1,len(preds)+1)),"Label":preds}).to_csv(fname,index=False,header=True)


# **Submit Results**

# In[ ]:


write_preds(preds,"cnn-test.csv")

