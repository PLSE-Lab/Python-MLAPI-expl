#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


#-------Import Dependencies-------#
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import os,shutil,math,scipy,cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from sklearn.metrics import confusion_matrix,roc_curve,auc

from PIL import Image
from PIL import Image as pil_image
from time import time
from PIL import ImageDraw
from glob import glob
from tqdm import tqdm
from skimage.io import imread
from IPython.display import SVG

from scipy import misc,ndimage
from scipy.ndimage.interpolation import zoom

from keras import backend as K
from keras import layers
from keras.preprocessing.image import save_img
from keras.utils.vis_utils import model_to_dot
from keras.applications.vgg19 import VGG19,preprocess_input
from keras.applications.xception import Xception
from keras.applications.nasnet import NASNetMobile
from keras.models import Sequential,Input,Model
from keras.layers import Dense,Flatten,Dropout,Concatenate,GlobalAveragePooling2D,Lambda,ZeroPadding2D
from keras.layers import SeparableConv2D,BatchNormalization,MaxPooling2D,Conv2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam,SGD
from keras.utils.vis_utils import plot_model
from keras.callbacks import ModelCheckpoint,EarlyStopping,TensorBoard,CSVLogger,ReduceLROnPlateau,LearningRateScheduler


# In[ ]:


from pathlib import Path
input_path = Path('../input/flowers-recognition/flowers/')
flowers_path = input_path / 'flowers'


# In[ ]:


flower_types = os.listdir(flowers_path)
print("Types of flowers found: ", len(flower_types))
print("Categories of flowers: ", flower_types)


# In[ ]:


# A list that is going to contain tuples: (species of the flower, corresponding image path)
flowers = []

for species in flower_types:
    # Get all the file names
    all_flowers = os.listdir(flowers_path / species)
    # Add them to the list
    for flower in all_flowers:
        flowers.append((species, str(flowers_path /species) + '/' + flower))

# Build a dataframe        
flowers = pd.DataFrame(data=flowers, columns=['category', 'image'], index=None)
flowers.head()


# In[ ]:


flowers.iloc[0,1]


# In[ ]:


# Let's check how many samples for each category are present
print("Total number of flowers in the dataset: ", len(flowers))
fl_count = flowers['category'].value_counts()
print("Flowers in each category: ")
print(fl_count)


# In[ ]:



get_ipython().run_line_magic('mkdir', '-p data/train')
get_ipython().run_line_magic('mkdir', '-p data/valid')
get_ipython().run_line_magic('mkdir', '-p data/validation')


# In[ ]:


# Inside the train and validation sub=directories, make sub-directories for each catgeory
get_ipython().run_line_magic('cd', 'data')
get_ipython().run_line_magic('mkdir', '-p train/daisy')
get_ipython().run_line_magic('mkdir', '-p train/tulip')
get_ipython().run_line_magic('mkdir', '-p train/sunflower')
get_ipython().run_line_magic('mkdir', '-p train/rose')
get_ipython().run_line_magic('mkdir', '-p train/dandelion')

get_ipython().run_line_magic('mkdir', '-p valid/daisy')
get_ipython().run_line_magic('mkdir', '-p valid/tulip')
get_ipython().run_line_magic('mkdir', '-p valid/sunflower')
get_ipython().run_line_magic('mkdir', '-p valid/rose')
get_ipython().run_line_magic('mkdir', '-p valid/dandelion')

get_ipython().run_line_magic('cd', '..')


# In[ ]:


for category in fl_count.index:
    samples = flowers['image'][flowers['category'] == category].values
    perm = np.random.permutation(samples)  #permutation shuffles the values and returns a copy of it
    # Copy first 3 samples to validation, 30 samples to the valid (test) directory and rest to the train directory
    for i in range(3):
        name = perm[i].split('/')[-1]
        shutil.copyfile(perm[i],'./data/validation/' + name)
    for i in range(3,30):
        name = perm[i].split('/')[-1]
        shutil.copyfile(perm[i],'./data/valid/' + str(category) + '/'+ name)
    for i in range(31,len(perm)):
        name = perm[i].split('/')[-1]
        shutil.copyfile(perm[i],'./data/train/' + str(category) + '/' + name)


# In[ ]:


augs = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,  
    zoom_range=0.2,        
    horizontal_flip=True,
    validation_split=0.3)  

train_gen = augs.flow_from_directory(
    'data/train',
    target_size = (150,150),
    batch_size=8,
    class_mode = 'categorical'
)

test_gen = augs.flow_from_directory(
    'data/valid',
    target_size=(150,150),
    batch_size=8,
    class_mode='categorical'
)

validation_gen = augs.flow_from_directory(
    'data/validation',
    target_size=(150,150),
    batch_size=7,
    class_mode='categorical'
)


# In[ ]:


def ConvBlock(model, layers, filters,name):
    for i in range(layers):
        model.add(SeparableConv2D(filters, (3, 3), activation='relu',name=name))
        model.add(BatchNormalization())
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    
def FCN():
    model = Sequential()
    model.add(Lambda(lambda x: x, input_shape=(150, 150, 3)))
    ConvBlock(model, 1, 16,'block_1')
    ConvBlock(model, 1, 32,'block_2')
    ConvBlock(model, 1, 64,'block_3')
    ConvBlock(model, 1, 128,'block_4')
    model.add(Flatten())
    model.add(Dense(1024,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(5,activation='sigmoid'))
    return model

model = FCN()
model.summary()


# In[ ]:


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
#tensorboard = TensorBoard(
#    log_dir = './logs',
#    histogram_freq=0,
#    batch_size=16,
#    write_graph=True,
#    write_grads=True,
#    write_images=False,
#)

csvlogger = CSVLogger(
    filename= "training_csv.log",
    separator = ",",
    append = False
)

#lrsched = LearningRateScheduler(step_decay,verbose=1)

reduce = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=40,
    verbose=1, 
    mode='auto',
    cooldown=1 
)

#callbacks = [checkpoint,tensorboard,csvlogger,reduce]
callbacks = [checkpoint,csvlogger,reduce]


# In[ ]:


opt = SGD(lr=1e-4,momentum=0.99)
opt1 = Adam(lr=2e-4)

model.compile(
    loss='binary_crossentropy',
    optimizer=opt,
    metrics=['accuracy']
)
    
history = model.fit_generator(
    train_gen, 
    steps_per_epoch  = 500, 
    validation_data  = test_gen,
    validation_steps = 500,
    epochs = 20, 
    verbose = 1,
    callbacks=callbacks
)


# In[ ]:


def show_final_history(history):
    fig, ax = plt.subplots(1, 2, figsize=(15,5))
    ax[0].set_title('loss')
    ax[0].plot(history.epoch, history.history["loss"], label="Train loss")
    ax[0].plot(history.epoch, history.history["val_loss"], label="Validation loss")
    ax[1].set_title('acc')
    ax[1].plot(history.epoch, history.history["accuracy"], label="Train acc")
    ax[1].plot(history.epoch, history.history["val_accuracy"], label="Validation acc")
    ax[0].legend()
    ax[1].legend()


# In[ ]:


show_final_history(history)
model.load_weights(best_model_weights)
model_score = model.evaluate_generator(test_gen)
print("Model Test Loss:",model_score[0])
print("Model Test Accuracy:",model_score[1])

model_json = model.to_json()
with open("model.json","w") as json_file:
    json_file.write(model_json)
    
model.save("model.h5")
print("Weights Saved")


# In[ ]:


nb_samples=15 #Total no. of images
batch_size=7
predict = model.predict_generator(validation_gen, steps = np.ceil(nb_samples/desired_batch_size)


# In[ ]:


validation_filenames = os.listdir("../data/validation")
validation_df = pd.DataFrame({
    'filename': validation_filenames
})


# In[ ]:


#validation_df['category'] = np.argmax(predict, axis=-1)

