#!/usr/bin/env python
# coding: utf-8

# # **Digit determine with Keras and TensorFlow**

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt

from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD

get_ipython().run_line_magic('matplotlib', 'inline')


# # ** Data preparation**

# In[ ]:


data_train = pd.read_csv('../input/train.csv')
X_train = data_train.iloc[:,1:]
y_train = data_train.iloc[:,:1]

data_test = pd.read_csv('../input/test.csv')
X_test = data_test #data_test useful for the section "Check prediction" below


# In[ ]:


#Rationing
X_train /= 255
X_train = X_train.as_matrix()
y_train = np_utils.to_categorical(y_train, 10)

X_test /= 255
X_test = X_test.as_matrix()


# # **Building model**

# In[ ]:


model = Sequential()
model.add(Dense(128, input_shape=(784,)))
model.add(Activation('relu'))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Activation('softmax'))
model.summary()

model.compile(loss='categorical_crossentropy',
optimizer=SGD(),
metrics=['accuracy']) 


# ***Training:***

# In[ ]:


model.fit(X_train, y_train, batch_size=128, epochs=40, verbose=1, validation_split=0.2)


# ***Prediction:***

# In[ ]:


predictions = model.predict_classes(X_test, verbose=1)


# In[ ]:


data_predictions = pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),"Label": predictions})
data_predictions.to_csv('predictions01.csv', index=False, header=True)
print(data_predictions.head(10))


# # **Check prediction**

# In[ ]:


def check_prediction(number):
    if number >=0 and number < 28000:
        img=data_test.iloc[number].as_matrix()
        img=img.reshape(28,28)
        plt.imshow(img,cmap='gray')
        pred=data_predictions['Label'][number]
        print(f"The model recognized the digit ---> {pred} <---")
    else:
        print('Enter correct number from [0; 27999]')


# In[ ]:


check_prediction(27999)


# In[ ]:


check_prediction(3)


# # Bonus
# Check your own images on this neural network
# https://github.com/Sasha654/RecognizeYourDigit

# In[ ]:




