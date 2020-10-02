#!/usr/bin/env python
# coding: utf-8

# Benchmark 0.932.

# In[ ]:


# import essentials
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from keras.models import Model, load_model
from keras.layers import Dense, Dropout, Flatten, Reshape, Input, BatchNormalization, Conv2D, MaxPooling2D
from keras import regularizers
from keras.callbacks import ModelCheckpoint


# In[ ]:


# load data locally
# you may want to check out loading data directly from keras
# https://keras.io/datasets/#fashion-mnist-database-of-fashion-articles
train_data = pd.read_csv('../input/fashion-mnist_train.csv')
test_data = pd.read_csv('../input/fashion-mnist_test.csv')


# In[ ]:


X_train = np.array(train_data.iloc[:, 1:]).reshape((-1, 28, 28))
y_train = train_data.iloc[:, 0]

X_test = np.array(test_data.iloc[:, 1:]).reshape((-1, 28, 28))
y_test = test_data.iloc[:, 0]


# In[ ]:


# if you want to load image directly from keras, uncomment code below
# from keras.datasets import fashion_mnist
# (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()


# In[ ]:


# check data shapes
print('X_train shape: ', X_train.shape)
print('y_train shape: ', y_train.shape)
print('X_test shape: ', X_test.shape)
print('y_test shape: ', y_test.shape)


# In[ ]:


# data augmentation
X_train_flip = np.fliplr(X_train)
y_train_flip = np.copy(y_train)

X_train = np.vstack([X_train, X_train_flip])
y_train = np.vstack([y_train.reshape((-1,1)), y_train_flip.reshape((-1,1))])
print('New data shapes after augmentation:')
print('X_train shape: ', X_train.shape)
print('y_train shape: ', y_train.shape)
print('X_test shape: ', X_test.shape)
print('y_test shape: ', y_test.shape)


# In[ ]:


# build a dictionary for easy access to object classes
objects = {0: 'T-shirt/top',
           1: 'Trouser',
           2: 'Pullover',
           3: 'Dress',
           4: 'Coat',
           5: 'Sandal',
           6: 'Shirt',
           7: 'Sneaker',
           8: 'Bag',
           9: 'Ankle boot'}


# In[ ]:


# let's have a quick look of those images
f, axes = plt.subplots(4, 4, figsize=(9,9))
for row in axes:
    for axe in row:
        index = np.random.randint(10000)
        img = X_train[index]
        obj = y_train[index][0]
        axe.imshow(img, cmap='gray')
        axe.set_title(objects[obj])
        axe.set_axis_off()


# In[ ]:


# structure data for training
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=np.random.randint(300))
X_train = X_train.reshape((-1, 28, 28, 1))
X_val = X_val.reshape((-1, 28, 28, 1))
# one_hot encoding
y_train = np_utils.to_categorical(y_train)
y_val = np_utils.to_categorical(y_val)


# In[ ]:


print('Structured data shapes for training:')
print('X_train shape: ', X_train.shape)
print('X_val shape: ', X_val.shape)
print('y_train shape: ', y_train.shape)
print('y_val shape: ', y_val.shape)


# In[ ]:


# to not exceed kaggle running time limit, the model size is reduced
def get_model(input_shape):
    
    drop = 0.3
    # l2 regularization as well as dropout can help prevent overfitting
    l2_reg = regularizers.l2(0.01)
    
    X_input = Input(input_shape)
    X = BatchNormalization()(X_input)
    X = Conv2D(8, (3,3), strides=(1,1), activation='relu',
               kernel_regularizer=l2_reg,
               kernel_initializer='glorot_normal')(X)
    X = MaxPooling2D((2,2))(X)
    
    X = Conv2D(16, (3,3), strides=(1,1), activation='relu',
               kernel_regularizer=l2_reg,
               kernel_initializer='glorot_normal')(X)
    X = MaxPooling2D((2,2))(X)    
    
    X = Conv2D(32, (2,2), strides=(1,1), activation='relu',
               kernel_regularizer=l2_reg,
               kernel_initializer='glorot_normal')(X)
    
    X = MaxPooling2D((2,2))(X)
    
    X = Flatten()(X)
    
    X = BatchNormalization()(X)
    X = Dense(128, activation='relu')(X)
    X = Dropout(drop)(X)
    
    X = Dense(32, activation='relu')(X)
    X = Dropout(0.1)(X)
    
    X = Dense(16, activation='relu')(X)
#     X = Dropout(drop)(X)    
    
    X = Dense(10, activation='softmax')(X)
    
    model = Model(inputs=[X_input], outputs=[X])
    
    return model


# In[ ]:


model = get_model((28,28,1))


# In[ ]:


# compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# In[ ]:


print(model.summary())


# In[ ]:


# training
f_path = 'model.h5'
msave = ModelCheckpoint(f_path, save_best_only=True)
epochs = 10 
batch_size = 128


# In[ ]:


training = model.fit(X_train, y_train,
                     validation_data=(X_val, y_val),
                     epochs=epochs,
                     callbacks = [msave],
                     batch_size=batch_size, 
                     verbose=1)


# In[ ]:



# show the loss and accuracy
loss = training.history['loss']
val_loss = training.history['val_loss']
acc = training.history['acc']
val_acc = training.history['val_acc']

# loss plot
tra = plt.plot(loss)
val = plt.plot(val_loss, 'r')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss')
plt.legend(["Training", "Validation"])

plt.show()

# accuracy plot
plt.plot(acc)
plt.plot(val_acc, 'r')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Accuracy')
plt.legend(['Training', 'Validation'], loc=4)
plt.show()


# In[ ]:





# In[ ]:


y_test = np.array(y_test)
X_test = np.array(X_test).reshape((-1, 28, 28, 1))


# In[ ]:


from sklearn.metrics import accuracy_score
model = load_model(f_path)
pred = model.predict(X_test)
# convert predicions from categorical back to 0...9 digits
pred_digits = np.argmax(pred, axis=1)


# In[ ]:


accuracy_score(y_test, pred_digits)


# Make the network bigger and train it longer will push the performance further.
# 
# Upvote if you find useful. Thanks :)

# In[ ]:





# In[ ]:




