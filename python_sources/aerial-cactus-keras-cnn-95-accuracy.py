#!/usr/bin/env python
# coding: utf-8

# # Project Paper
# ### Here is the link to the research paper for this dataset
# ### https://jivasquez.files.wordpress.com/2019/03/rp_cactus_recognition_elsa-1.pdf

# # Import Dependencies
# ### Load in the libraries that you will need

# In[ ]:


#-------Import Dependencies-------#
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import os,shutil,math,scipy,cv2
import numpy as np
import matplotlib.pyplot as plt
import random as rn


from sklearn.utils import shuffle
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix,roc_curve,auc

from PIL import Image
from PIL import Image as pil_image
from PIL import ImageDraw

from time import time
from glob import glob
from tqdm import tqdm
from skimage.io import imread
from IPython.display import SVG

from scipy import misc,ndimage
from scipy.ndimage.interpolation import zoom
from scipy.ndimage import imread


from keras import backend as K
from keras.utils.np_utils import to_categorical
from keras import layers
from keras.preprocessing.image import save_img
from keras.utils.vis_utils import model_to_dot
from keras.applications.vgg16 import VGG16,preprocess_input
from keras.applications.xception import Xception
from keras.applications.nasnet import NASNetMobile
from keras.models import Sequential,Input,Model
from keras.layers import Dense,Flatten,Dropout,Concatenate,GlobalAveragePooling2D,Lambda,ZeroPadding2D
from keras.layers import SeparableConv2D,BatchNormalization,MaxPooling2D,Conv2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam,SGD
from keras.utils.vis_utils import plot_model
from keras.callbacks import ModelCheckpoint,EarlyStopping,TensorBoard,CSVLogger,ReduceLROnPlateau,LearningRateScheduler


# # Custom Function
# ### function for visualizing training results

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


# # Image Preprocessing
# ### Thank to Keras ImageDataGenerator for image preprocessing and data augmentation

# In[ ]:


augs_gen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True)  

train_gen = augs_gen.flow_from_directory(
    '../input/training_set/training_set',
    target_size = (32,32),
    batch_size=32,
    shuffle=True,
    class_mode = 'binary'
)

test_gen = augs_gen.flow_from_directory(
    '../input/validation_set/validation_set/',
    target_size=(32,32),
    batch_size=32,
    shuffle=False,
    class_mode = 'binary'
)


# # Model
# ### This is the LE-Net5 model from the research paper for this dataset

# In[ ]:


model = Sequential()
model.add(Conv2D(6, kernel_size=(5,5),activation='relu',input_shape=(32,32,3)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=2,strides=2))
model.add(Conv2D(16,kernel_size=5,strides=1,activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=2,strides=2))
model.add(Flatten())
model.add(Dense(120,activation='relu'))
model.add(Dense(84,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1,activation='sigmoid'))
model.summary()


SVG(model_to_dot(model).create(prog='dot', format='svg'))
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)


# # Callbacks
# ### Here we use Keras callbacks to monitor how the training is going and adjust the training accordingly

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
    patience=5,
    verbose=1, 
    mode='auto',
    cooldown=1 
)

callbacks = [checkpoint,tensorboard,csvlogger,reduce]


# # Train The Model
# ### Here we train the model with our generators

# In[ ]:


opt = SGD(lr=2e-4,momentum=0.99)
opt1 = Adam(lr=1e-2)

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)
    
history = model.fit_generator(
    train_gen, 
    validation_data = test_gen,
    validation_steps = 100,
    steps_per_epoch  = 100, 
    epochs = 150,
    verbose = 1,
    callbacks=callbacks
)


# # Evaluation
# ### Here we visualize the training, load the the best weights from the model, evaluate the model, and save the weights

# In[ ]:


show_final_history(history)
model.load_weights(best_model_weights)
model_eval = model.evaluate_generator(test_gen,steps=100)
print("Model Test Loss:",model_eval[0])
print("Model Test Accuracy:",model_eval[1])

model_json = model.to_json()
with open("model.json","w") as json_file:
    json_file.write(model_json)
    
model.save("model.h5")
print("Weights Saved")


# # TensorBoard
# ### Here is a script that will give you a url to view your model's training results.

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




