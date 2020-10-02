#!/usr/bin/env python
# coding: utf-8

# ## Table of contents
# * [Import Dependencies and run Tensorboard](#Import-Dependencies-and-run-Tensorboard)
# * [Load and Visualize data](#Load-and-Visualize-data)
# * [Build Model](#Build-Model)
# * [Train Model](#Train-Model)
# * [Evaluate Model](#Evaluate-Model)

# ## Import Dependencies and run Tensorboard

# In[ ]:


get_ipython().system('wget https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-amd64.zip')
get_ipython().system('unzip ngrok-stable-linux-amd64.zip')
LOG_DIR = './logs' # Here you have to put your log directory
get_ipython().system_raw(
    'tensorboard --logdir {} --host 0.0.0.0 --port 6006 &'
    .format(LOG_DIR)
)
get_ipython().system_raw('./ngrok http 6006 &')
get_ipython().system(' curl -s http://localhost:4040/api/tunnels | python3 -c     "import sys, json; print(json.load(sys.stdin)[\'tunnels\'][0][\'public_url\'])"')


# In[ ]:


import numpy as np
import pandas as pd 
import cv2
from IPython.display import Image
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
print(os.listdir("../input/utkface_aligned_cropped/"))


# ## Load and Visualize data

# In[ ]:


def imshow(img):
    plt.imshow(img, interpolation='bicubic')
    plt.xticks([]), plt.yticks([]) 
    plt.show()


# In[ ]:


onlyfiles = os.listdir("../input/utkface_aligned_cropped/UTKFace")
y = np.array([[[i.split('_')[0]],[i.split('_')[1]]] for i in onlyfiles])
# y = np.array([[i.split('_')[1] for i in onlyfiles]]).T
print(y.shape)
print(y[0])


# In[ ]:


X_data =[]
for file in onlyfiles:
    face = cv2.imread("../input/utkface_aligned_cropped/UTKFace/"+file,cv2.IMREAD_COLOR)
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face =cv2.resize(face, (32,32) )
    X_data.append(face)
X_data=np.array(X_data)
X_data.shape


# In[ ]:


X = np.squeeze(X_data)
imshow(X[1])
print(y[1])


# In[ ]:


X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.33)
y_train=[y_train[:,1],y_train[:,0]]
y_valid=[y_valid[:,1],y_valid[:,0]]


# ## Build Model

# In[ ]:



def gen_model():
    inputs = tf.keras.layers.Input(shape=(32, 32, 3))
    x = inputs
    x = layers.Conv2D(32,3,activation='relu')(x)
    x = layers.Conv2D(32,3,activation='relu')(x)
    x = layers.MaxPool2D(2)(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Conv2D(64,3,activation='relu')(x)
    x = layers.Conv2D(64,3,activation='relu')(x)
    x = layers.MaxPool2D(2)(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Conv2D(84,3,activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Flatten()(x)
    x1 = layers.Dense(64,activation='relu')(x)
    x2 = layers.Dense(64,activation='relu')(x)
    x1 = layers.Dense(1,activation='sigmoid',name='sex_out')(x1)
    x2 = layers.Dense(1,activation='relu',name='age_out')(x2)
    model = tf.keras.models.Model(inputs=inputs, outputs=[x1, x2])
    model.compile(optimizer='Adam', loss=['binary_crossentropy','mae'])
    tf.keras.utils.plot_model(model, 'model.png',show_shapes=True)  
    return model
model=gen_model()

Image('model.png')


# ## Train Model

# In[ ]:


import random
random_id=random.random()
model.summary()
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=75, monitor='val_loss',restore_best_weights=True),
    tf.keras.callbacks.TensorBoard(log_dir='./logs/'+str(random_id))
]
model.fit(X_train, y_train, epochs=200,batch_size=240,validation_data=(X_valid,y_valid),callbacks=callbacks, shuffle=True)


# ## Evaluate Model

# In[ ]:


model.evaluate(X_valid,y_valid)


# In[ ]:


p_id=2
imshow(X_valid[p_id])
print(y_valid[0][p_id],y_valid[1][p_id])
print(model.predict([[X_valid[p_id]]]))

