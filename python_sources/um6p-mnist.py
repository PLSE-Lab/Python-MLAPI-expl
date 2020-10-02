#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from keras.layers import Input, Dense, Conv2D, Flatten, Activation, MaxPooling2D, BatchNormalization
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt


# In[ ]:


df = pd.read_csv('../input/train.csv')


# In[ ]:


df.head(1)


# In[ ]:


labels = pd.get_dummies(df['label'], columns='label').values  #df['label'].values
images = df.drop('label', axis=1).values

images = StandardScaler().fit_transform( images )

images.shape


# In[ ]:


images = images.reshape(len(images),28,28, 1)




images.shape


# In[ ]:


def create_cnn_model():
    np.random.seed(0)
    inpt_image = Input( shape=(28,28, 1) )
    
    conv1 = Conv2D( filters=20, kernel_size=(3,3), strides=(1,1), padding='same' ) (inpt_image)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    conv1 = MaxPooling2D()(conv1)
    
    conv1 = Conv2D( filters=20, kernel_size=(3,3), strides=(1,1), padding='same' ) (conv1)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    conv1 = MaxPooling2D()(conv1)

    
    f = Flatten()(conv1)
    
    output = Dense(10)(f)
    output = Activation('softmax')(output)
    
    model = Model(inpt_image, output)
    
    return model
    


# In[ ]:


create_cnn_model().summary()


# In[ ]:


from sklearn.model_selection import StratifiedKFold
from keras.optimizers import SGD


# In[ ]:


cv = StratifiedKFold( n_splits=2, random_state=0, shuffle=True )

for train_index, test_index in cv.split(images, df['label'].values):
    
    X_train = images[train_index]
    y_train = labels[train_index]

    X_test = images[test_index]
    y_test = labels[test_index]

    model = create_cnn_model()
    
    sgd = SGD(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['acc'])
    
    stop = EarlyStopping(patience=5, mode='max',  monitor='val_acc')
    save = ModelCheckpoint('./save_best.h5',monitor='val_acc', save_best_only=True, mode='max')
    
    hist = model.fit( X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test,y_test ) , 
                     callbacks=[stop, save])    
    
    break


# In[ ]:


Epoch 29/100
20997/20997 [==============================] - 3s 164us/step - loss: 0.0146 - acc: 0.9972 - val_loss: 0.0673 - val_acc: 0.9807


# In[ ]:


get_ipython().run_line_magic('pinfo', 'Conv2D')


# In[ ]:




