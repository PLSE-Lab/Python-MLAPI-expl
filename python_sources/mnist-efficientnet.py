#!/usr/bin/env python
# coding: utf-8

# Simple example of transfer learning from pretrained model using Keras and Efficientnet (https://pypi.org/project/efficientnet/).

# In[ ]:


import numpy as np
import pandas as pd
import os
print(os.listdir("../input"))


# In[ ]:


get_ipython().system('pip install git+https://github.com/qubvel/efficientnet')


# In[ ]:


from efficientnet import EfficientNetB3


# In[ ]:


from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten 
from keras.models import Model
from keras import optimizers
from keras.utils import np_utils
import cv2
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# Load the data
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")


# In[ ]:


Y_train = train["label"]

# Drop 'label' column
X_train = train.drop(labels = ["label"],axis = 1) 


# In[ ]:


X_train.shape, test.shape


# In[ ]:


# Normilize data
X_train = X_train.astype('float32')
test = test.astype('float32')
X_train /= 255
test /= 255


# In[ ]:


X_train = X_train.values.reshape(-1,28,28,1)
test = test.values.reshape(-1,28,28,1)


# In[ ]:


X_train.shape, test.shape


# In[ ]:


X_train3 = np.full((42000, 28, 28, 3), 0.0)

for i, s in enumerate(X_train):
    X_train3[i] = cv2.cvtColor(s, cv2.COLOR_GRAY2RGB) 


# In[ ]:


g = plt.imshow(X_train3[1])


# In[ ]:


test3 = np.full((28000, 28, 28, 3), 0.0)

for i, s in enumerate(test):
    test3[i] = cv2.cvtColor(s, cv2.COLOR_GRAY2RGB) 


# In[ ]:


g = plt.imshow(test3[1])


# In[ ]:


X_train3.shape, test3.shape


# In[ ]:


Y_train = np_utils.to_categorical(Y_train, 10)
Y_train


# In[ ]:


model = EfficientNetB3(weights='imagenet', input_shape = (28,28,3), include_top=False)


# In[ ]:


model.trainable = False


# In[ ]:


x = model.output
x = Flatten()(x)
x = Dense(1024, activation="relu")(x)
x = Dropout(0.5)(x)
predictions = Dense(units = 10, activation="softmax")(x)
model_f = Model(input = model.input, output = predictions)
model_f.compile(optimizers.rmsprop(lr=0.0001, decay=1e-6),loss='categorical_crossentropy',metrics=['accuracy'])


# In[ ]:


get_ipython().run_cell_magic('time', '', '# Train model\nhistory = model_f.fit(X_train3, Y_train,\n              epochs=10,\n              batch_size = 128,\n              validation_split=0.1,\n              shuffle=True,\n              verbose=2)')


# In[ ]:


import json

with open('history.json', 'w') as f:
    json.dump(history.history, f)

history_df = pd.DataFrame(history.history)
history_df[['loss', 'val_loss']].plot()
history_df[['acc', 'val_acc']].plot()


# In[ ]:


get_ipython().run_cell_magic('time', '', '# Prediction\ntest_predictions = model_f.predict(test3)')


# In[ ]:


test_predictions.shape


# In[ ]:


test_predictions[0]


# In[ ]:


# select the index with the maximum probability
results = np.argmax(test_predictions,axis = 1)
results = pd.Series(results,name="Label")


# In[ ]:


submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

submission.to_csv("submission.csv",index=False)


# In[ ]:


submission.head()

