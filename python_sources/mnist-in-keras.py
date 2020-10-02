#!/usr/bin/env python
# coding: utf-8

# # Imports

# In[ ]:


import os
from pathlib import Path
import keras 
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
PATH=Path("../input/")
print(os.listdir("../input/"))


# # Load Data

# In[ ]:


train=pd.read_csv(PATH/'train.csv')
test=pd.read_csv(PATH/'test.csv')
train.shape,test.shape


# # Extract Input and Target Variable

# In[ ]:


x=train.drop("label",axis=1)
y=np.array(train['label'])
x.shape,y.shape


# # Train Test Split

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


x_train, x_valid, y_train, y_valid = train_test_split(x,y,test_size=0.2,random_state=123)
print(x_train.shape,x_valid.shape)
x_train = x_train.values.reshape(33600, 784)
x_valid = x_valid.values.reshape(8400, 784)
x_train = x_train.astype('float32')
x_valid = x_valid.astype('float32')
x_train /= 255
x_valid /= 255
print(x_train.shape[0], 'train samples')
print(x_valid.shape[0], 'valid samples')


# # Convert class vectors to binary class matrices

# In[ ]:


y_train.shape,y_train[:2]


# In[ ]:


num_classes=10
y_train = keras.utils.to_categorical(y_train, num_classes)
y_valid = keras.utils.to_categorical(y_valid, num_classes)
print(y_train.shape,y_valid.shape)


# # Model Architecture

# In[ ]:


model = Sequential()
model.add(Dense(256, activation='relu', input_shape=(784,)))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))
model.summary()


# In[ ]:


model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=0.01),
              metrics=['accuracy'])


# In[ ]:


epochs=5
batch_size=64
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_valid, y_valid))


# In[ ]:


score = model.evaluate(x_valid, y_valid, verbose=0)
print('Valid loss:', score[0])
print('Valid accuracy:', score[1])


# In[ ]:


#from pathlib import Path
#import simplejson
#serialize model to JSON
#filepath_json=Path('../input/')
#model_json = model.to_json()
#with open(filepath_json/"mnist_keras.json", "w") as json_file:
 #   json_file.write(simplejson.dumps(simplejson.loads(model_json), indent=4))


# In[ ]:


model.save_weights("mnist_keras.h5")


# # Load Test Data

# In[ ]:


test = pd.read_csv("../input/test.csv")
print(test.shape)


# # Preprocessing

# In[ ]:


x_test=test.loc[:,test.columns != "label"]
x_test = x_test.astype('float32')
x_test /= 255
print(x_test.shape[0], 'test samples')


# In[ ]:


score = model.predict(x_test, verbose=0)
print(score.shape)


# # Take a look

# In[ ]:


np.argmax(score,axis=1)[:4],np.argmax(score,axis=1).shape


# # Generate Predictions

# In[ ]:


predictions=np.argmax(score,axis=1)
print("Prediction shape",predictions.shape)


# # Create Submission File

# In[ ]:


submissions=pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),
                         "Label": predictions})
submissions.to_csv("my_submissions_keras.csv", index=False, header=True)

