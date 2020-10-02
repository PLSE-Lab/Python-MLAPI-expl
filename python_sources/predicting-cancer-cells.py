#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import os
import pandas as pd
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split


# In[ ]:


test = pd.read_csv('../input/hmnist_28_28_RGB.csv')

test.head(10)


# In[ ]:


cancer_dict = {
    0: 'Melanocytic nevi',
    1: 'Melanoma',
    2: 'Benign keratosis-like lesions ',
    3: 'Basal cell carcinoma',
    4: 'Actinic keratoses',
    5: 'Vascular lesions',
    6: 'Dermatofibroma'
}


# In[ ]:


#input_shape = (28,28,3)
X = test.iloc[:,0:-1]
Y = test.iloc[:,-1]


# In[ ]:


X.shape, Y.shape


# In[ ]:


X = np.array(X)
Y = np.array(Y)


# In[ ]:


X = X.reshape(X.shape[0],28,28,3)


# In[ ]:


trainX,testX,trainY,testY = train_test_split(X,Y,test_size=0.1,random_state=13)


# In[ ]:


trainX = trainX.astype('float64') / 255.0
testX =  testX.astype('float64') / 255.0


# In[ ]:


trainY = to_categorical(trainY)
testY = to_categorical(testY)


# In[ ]:


from keras.layers import Flatten, Dense, Activation, Conv2D, MaxPooling2D, Dropout, BatchNormalization
from keras.models import load_model, Sequential
from keras.callbacks import EarlyStopping,ReduceLROnPlateau, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator


# In[ ]:


datagen = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
    height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
    horizontal_flip=True, fill_mode="nearest")

model = Sequential()
model.add(Conv2D(32, (3,3), padding='same', input_shape=(28,28,3)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(3,3)))

model.add(Conv2D(64, (3,3), padding='same'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Conv2D(64, (3,3), padding='same'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3,3), padding='same'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Conv2D(128, (3,3), padding='same'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(7))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=['accuracy'])

datagen.fit(trainX)

filepath = 'cancer.model'

checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=3, verbose=1, mode='max', min_lr=0.0001)

callbacks_list = [checkpoint, reduce_lr]


# In[ ]:



model.fit(trainX, trainY,
          batch_size=32,
          epochs=25,
          verbose=1,
          validation_data=(testX, testY),shuffle=True)


# In[ ]:


model.save('cancer.model')


# In[ ]:


pred = model.predict(testX)
from sklearn.metrics import accuracy_score


# In[ ]:


import matplotlib.pyplot as plt
plt.imshow(testX[11].reshape(28,28,3), cmap='viridis')
plt.title('Prediction: {} \nTrue Value: {}'.format(cancer_dict[np.argmax(pred[11])], cancer_dict[np.argmax(testY[11])]))
plt.show()


# In[ ]:





# In[ ]:




