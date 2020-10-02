#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, CuDNNLSTM
from keras.layers import Dropout
from keras.optimizers import Adam, RMSprop
from keras.callbacks import ReduceLROnPlateau
from sklearn.model_selection import train_test_split

import pandas as pd


# In[ ]:


train = pd.read_csv("../input/train.csv")
print(train.shape)
train.head()


# In[ ]:


class_count = train['label'].value_counts()
class_count.plot(kind='bar', title='Check imbalanced data');


# In[ ]:


test = pd.read_csv("../input/test.csv")
print(test.shape)
test.head()


# In[ ]:


x_train = train.iloc[:,1:].values.astype('float32')
y_train = train.iloc[:,0].values.astype('int32')
x_test = test.values.astype('float32')


# In[ ]:


x_train = x_train/255.0
x_test = x_test/255.0


# In[ ]:


x_train = x_train.reshape(x_train.shape[0], 28, 28)
x_test = x_test.reshape(x_test.shape[0], 28, 28)
print('Input shape: ', x_train.shape[1:])


# In[ ]:


x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=42)


# In[ ]:


model = Sequential()

model.add(CuDNNLSTM(128, input_shape=(x_train.shape[1:]), return_sequences=True))
model.add(Dropout(0.2))

model.add(CuDNNLSTM(128))
model.add(Dropout(0.2))

model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(10, activation='softmax'))


# Using ADAM optimizer

# In[ ]:


opt = Adam(lr=1e-3, decay=1e-5)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-7, verbose=1)


# In[ ]:


model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])


# In[ ]:


history = model.fit(x_train, y_train, epochs=30, validation_data=(x_val, y_val), callbacks=[reduce_lr])


# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# Get training and test loss histories
training_loss = history.history['loss']
test_loss = history.history['val_loss']

# Create count of the number of epochs
epoch_count = range(1, len(training_loss) + 1)

# Visualize loss history
plt.ylim(0, 0.3)
plt.plot(epoch_count, training_loss, 'r--')
plt.plot(epoch_count, test_loss, 'b-')
plt.legend(['Training Loss', 'Test Loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()


# In[ ]:


import seaborn as sn
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

pred = model.predict_classes(x_val)
plt.figure(figsize = (11,9))
sn.heatmap(confusion_matrix(y_val, pred), annot=True)

print(y_val[:25])
print(pred[:25])
print("******************************************")
print("Precision: ", precision_score(y_val, pred), average='macro')
print("Recall: ", recall_score(y_val, pred), average='micro')
print("f1 score: ", f1_score(y_val, pred), average='micro')
print("******************************************")


# Using RMSProp optimizer

# In[ ]:


opt = RMSprop(lr=1e-3, rho=0.9, epsilon=1e-08, decay=0.0)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-7, verbose=1)


# In[ ]:


model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])


# In[ ]:


history = model.fit(x_train, y_train, epochs=30, validation_data=(x_val, y_val), callbacks=[reduce_lr])


# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# Get training and test loss histories
training_loss = history.history['loss']
test_loss = history.history['val_loss']

# Create count of the number of epochs
epoch_count = range(1, len(training_loss) + 1)

# Visualize loss history
plt.ylim(0, 0.3)
plt.plot(epoch_count, training_loss, 'r--')
plt.plot(epoch_count, test_loss, 'b-')
plt.legend(['Training Loss', 'Test Loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()


# In[ ]:


prediction = model.predict_classes(x_test)


# In[ ]:


submissions = pd.DataFrame({"ImageId": list(range(1,len(prediction)+1)),
                         "Label": prediction})
submissions.to_csv("submission.csv", index=False, header=True)


# In[ ]:




