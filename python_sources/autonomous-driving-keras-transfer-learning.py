#!/usr/bin/env python
# coding: utf-8

# # Import Dependencies

# In[ ]:


from __future__ import division
import random,pickle,csv,cv2,os,scipy,pickle,warnings,matplotlib
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import gc

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from scipy.stats import norm,skew
from itertools import islice

import keras.backend as K
from keras.callbacks import History
from keras.layers import Dense,Flatten,Dropout,Concatenate,GlobalAveragePooling2D,Lambda,ZeroPadding2D,GlobalMaxPooling2D
from keras.layers import SeparableConv2D,BatchNormalization,MaxPooling2D,Conv2D
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.utils import print_summary
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint,EarlyStopping,TensorBoard,CSVLogger,ReduceLROnPlateau,LearningRateScheduler
from keras.optimizers import Adam,SGD
from keras import applications
from keras.utils.vis_utils import plot_model

print(os.listdir('../input/self driving car training data/data'))
warnings.filterwarnings('ignore')


# # Helper Functions

# In[ ]:


def show_final_history(history):
    fig, ax = plt.subplots(1, 2, figsize=(15,5))
    ax[0].set_title('loss')
    ax[0].plot(history.epoch, history.history["loss"], label="Train loss")
    ax[0].plot(history.epoch, history.history["val_loss"], label="Validation loss")
    ax[1].set_title('acc')
    ax[1].plot(history.epoch, history.history["acc"], label="Train acc")
    ax[1].plot(history.epoch, history.history["val_acc"], label="Validation acc")
    ax[0].legend()
    ax[1].legend()


# In[ ]:


def image_preprocessing(img):
    resized_image = cv2.resize((cv2.cvtColor(img,cv2.COLOR_RGB2HSV))[:,:,1],(40,40))
    return resized_image


# In[ ]:


def load_training(delta):
    logs = []
    features = []
    labels = []
    with open(labels_file,'rt') as f:
        reader = csv.reader(f)
        for line in reader:
            logs.append(line)
        log_labels = logs.pop(0)
        
    for i in range(len(logs)):
        for j in range(3):
            img_path = logs[i][j]
            img_path = features_directory + 'IMG' + (img_path.split('IMG')[1]).strip()
            img = plt.imread(img_path)
            features.append(image_preprocessing(img))
            
            if j == 0:
                labels.append(float(logs[i][3]))
            elif j == 1:
                labels.append(float(logs[i][3]) + delta)
            else:
                labels.append(float(logs[i][3]) - delta)
    return features,labels


# In[ ]:


def loadFromPickle():
    with open('features','rb') as f:
        features = np.array(pickle.load(f))
    with open('labels','rb') as f:
        labels = np.array(pickle.load(f))
    return features,labels

def augmentData(features,labels):
    features = np.append(features,features[:,:,::-1],axis=0)
    labels = np.append(labels,-labels,axis=0)
    return features,labels


# # Load in the data

# In[ ]:


features_directory = '../input/self driving car training data/data/'
labels_file = '../input/self driving car training data/data/driving_log.csv'


# In[ ]:


delta = 0.2
features,labels = load_training(delta)

features = np.array(features).astype('float32')
labels = np.array(labels).astype('float32')

with open('features','wb') as f:
    pickle.dump(features,f,protocol=4)
with open('labels','wb') as f:
    pickle.dump(labels,f,protocol=4)


# In[ ]:


pan = pd.Panel(features)
df = pan.swapaxes(1,2).to_frame()
df.index = df.index.droplevel('major')
df.index = df.index+1


# In[ ]:


features,labels = loadFromPickle()
features,labels = shuffle(features,labels)

x_train,x_val,y_train,y_val = train_test_split(features,labels,random_state=42,test_size=0.2)

x_train = x_train.reshape(x_train.shape[0],40,40,1)
x_val = x_val.reshape(x_val.shape[0],40,40,1)


# # Create The Model

# In[ ]:


base_model = MobileNetV2(include_top=False,weights=None,input_shape=(40,40,1))

for layer in base_model.layers:
    layer.trainable = False
    
model = Sequential()
model.add(base_model)
model.add(GlobalAveragePooling2D())
model.add(Dropout(0.5))
model.add(Dense(1,activation='sigmoid'))
model.summary()
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)


# # Callbacks

# In[ ]:


#-------Callbacks-------------#
best_model_weights = './base.model'
checkpoint = ModelCheckpoint(
    best_model_weights,
    monitor='val_loss',
    verbose=1,
    save_best_only=True,
    mode='min',
    save_weights_only=False,
    period=1
)
earlystop = EarlyStopping(
    monitor='val_loss',
    min_delta=0.001,
    patience=10,
    verbose=1,
    mode='auto'
)
tensorboard = TensorBoard(
    log_dir = './logs',
    histogram_freq=0,
    batch_size=16,
    write_graph=True,
    write_grads=True,
    write_images=False,
)

csvlogger = CSVLogger(
    filename= "training_csv.log",
    separator = ",",
    append = False
)

#lrsched = LearningRateScheduler(step_decay,verbose=1)

reduce = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=10,
    verbose=1, 
    mode='auto',
    cooldown=1 
)

callbacks = [checkpoint,tensorboard,csvlogger,reduce]


# # Now we train the model

# In[ ]:


opt = SGD(lr=1e-4,momentum=0.99)
opt1 = Adam(lr=1e-3)

model.compile(
    loss='binary_crossentropy',
    optimizer='rmsprop',
    metrics=['accuracy']
)
    
history = model.fit(
    x_train,
    y_train,
    validation_data=(x_val,y_val),
    epochs = 50, 
    verbose = 1,
    callbacks=callbacks,
    batch_size = 256
)


# # Visualize the training and save the weights and json file

# In[ ]:


show_final_history(history)
model.load_weights(best_model_weights)

model_json = model.to_json()
with open("model.json","w") as json_file:
    json_file.write(model_json)
    
model.save("model.h5")
print("Weights Saved")
print("JSON Saved")


# # TensorBoard

# In[ ]:


get_ipython().system('wget https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-amd64.zip')
get_ipython().system('unzip ngrok-stable-linux-amd64.zip')
LOG_DIR = './logs' # Here you have to put your log directory
get_ipython().system_raw(
    'tensorboard --logdir {} --host 0.0.0.0 --port 8080 &'
    .format(LOG_DIR)
)
get_ipython().system_raw('./ngrok http 8080 &')
get_ipython().system(' curl -s http://localhost:4040/api/tunnels | python3 -c     "import sys, json; print(json.load(sys.stdin)[\'tunnels\'][0][\'public_url\'])"')


# In[ ]:




