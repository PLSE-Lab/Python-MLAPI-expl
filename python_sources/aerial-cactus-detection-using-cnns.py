#!/usr/bin/env python
# coding: utf-8

# ## Import libraries

# In[ ]:


import os

import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt

from tqdm import tqdm, tqdm_notebook

from keras import models, layers
from keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dense, LeakyReLU, Dropout
from keras.applications import VGG16, densenet
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator


# ## Basic Setup
# I find it to be a good practice to setup base_dir, train_dir, test_dir in the beginning
# and then use these variable throughout the code whenever something from the file system
# has to be accessed.

# In[ ]:


base_dir = "../input/"
print(os.listdir(base_dir))

train_dir = os.path.join(base_dir, "train/train/")
test_dir = os.path.join(base_dir, "test/test/")

df_train = pd.read_csv(os.path.join(base_dir, "train.csv"))
print(df_train.head())


# ## Sample Image
# Here is a sample image just to get an idea of the kind of images in the dataset.

# In[ ]:


im = cv2.imread(train_dir + df_train["id"][0])
plt.imshow(im)


# ## TensorBoard Visualization
# Magic functions are used to setup the environment for viewing dynamic TensorBoard visualizations when the model is training. TensorBoard allows you to visualize training loss and accuracy as well as validation loss and accuracy. Moreover, it also allows you to visualize the computation graph of your model. For more information checkout: [https://www.tensorflow.org/guide/summaries_and_tensorboard](http://)
# 
# To reload the tensorboard visualization, say when you start training a new model, run: `%reload_ext tensorboard.notebook` instead of `%load_ext tensorboard.notebook`

# In[ ]:


get_ipython().run_line_magic('load_ext', 'tensorboard.notebook')
get_ipython().run_line_magic('tensorboard', '--logdir logs')


# ## Image Generators
# Data Augmentation is a useful technique to increase the size of one's dataset. It is especially useful if the dataset is small as in this case. Augmentation involves rotating, fliping (both horizontal and vertical) and various other operations to obtain similar images for which we have labels. This effectively increases the dataset while also making the model trained on this dataset invariant to such operations.
# 
# Keras provides ImageDataGenerators which lifts the burden of Image Preprocessing, Augmentation as well as Train/Val split from the shoulders' of the programmer. However, there is one drawback in using these generators. For datasets which can be stored in the memory, generators' significantly increase the training time. For datasets like the Cactus Aerial Detection, loading the dataset into the memory and then manually preprocessing it should be preferred. However, for the ease of use (augmentation), I will be using the generators only.

# In[ ]:


df_train['has_cactus'] = df_train['has_cactus'].astype(str)

batch_size = 64
train_size = 15750
validation_size = 1750

datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip=True,
    validation_split=0.1)

data_args = {
    "dataframe": df_train,
    "directory": train_dir,
    "x_col": 'id',
    "y_col": 'has_cactus',
    "shuffle": True,
    "target_size": (32, 32),
    "batch_size": batch_size,
    "class_mode": 'binary'
}

train_generator = datagen.flow_from_dataframe(**data_args, subset='training')
validation_generator = datagen.flow_from_dataframe(**data_args, subset='validation')


# ## Keras Callbacks
# Keras provides this another amazing feature called callbacks. Callbacks essentially allow you to run some computation after every epoch or step to evaluate the training of your model or to store the intermediate results.
# 
# Following is a brief explanation of all the callbacks I have used:
# 1. TensorBoard - Helps in visualizing the loss and acc for both training and validation sets during training
# 2. EarlyStopping - Prevents overfitting. It continuously monitors the model by checking the given metric, say val_acc. If val_acc doesn't improve over the current best val_acc in specified number of epochs (patience) then the model stops training.
# 3. ReduceLROnPlateau - It is similar to EarlyStopping except rather than stopping the training it reduces the learning rate. Lower learning rate allows the model to do more fine-grained learning.
# 4. ModelCheckpoint - This is especially useful for models which take a long time to train. It allows us to store intermediate results i.e. after every epoch or so.

# In[ ]:


ckpt_path = 'aerial_cactus_detection.hdf5'

tensorboard_cb = TensorBoard()
earlystop_cb = EarlyStopping(monitor='val_acc', patience=10, verbose=1, restore_best_weights=True)
reducelr_cb = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=5, verbose=1)
modelckpt_cb = ModelCheckpoint(ckpt_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

callbacks = [tensorboard_cb, earlystop_cb, reducelr_cb, modelckpt_cb]


# ## Model
# My choice for model is mostly random. I have used two blocks of three convolutional layers each. Each convolutional layer is followed by a batch normalization layer. I have used leaky relu for non-linearity. The convolutional layers are followed by two dense layers and then finally the output layer.
# 
# Before this model I tried using a pre-trained model.The results I got VGG16 using weights pretrained on ImageNet were subpar as compared to the current results. I feel this is because of the lack of batch normalization layers in the VGG16 but I could be wrong.

# In[ ]:


model = models.Sequential([
    Conv2D(32, (3,3), input_shape=(32, 32, 3)),
    LeakyReLU(alpha=0.3),
    BatchNormalization(),
    Conv2D(32, (3,3)),
    LeakyReLU(alpha=0.3),
    BatchNormalization(),
    Conv2D(32, (3,3)),
    LeakyReLU(alpha=0.3),
    BatchNormalization(),
    MaxPooling2D(2,2),
   
    Conv2D(64, (3,3)),
    LeakyReLU(alpha=0.3),
    BatchNormalization(),
    Conv2D(64, (3,3)),
    LeakyReLU(alpha=0.3),
    BatchNormalization(),
    Conv2D(64, (3,3)),
    LeakyReLU(alpha=0.3),
    BatchNormalization(),
    MaxPooling2D(2,2),
    
    Flatten(),
    Dense(units=128),
    LeakyReLU(alpha=0.3),
    Dropout(0.4),
    Dense(units=64),
    LeakyReLU(alpha=0.3),
    Dropout(0.4),
    
    Dense(units=1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy',
              optimizer=Adam(lr=0.001),
              metrics=['acc'])


# ## Training

# In[ ]:


history = model.fit_generator(train_generator,
              validation_data=validation_generator,
              steps_per_epoch=train_size//batch_size,
              validation_steps=validation_size//batch_size,
              epochs=100,
              shuffle=True,
              callbacks=callbacks, 
              verbose=2)


# ## Train Vs Validation Visualization

# In[ ]:


# Training plots
epochs = [i for i in range(1, len(history.history['loss'])+1)]

plt.plot(epochs, history.history['loss'], color='blue', label="training_loss")
plt.plot(epochs, history.history['val_loss'], color='red', label="validation_loss")
plt.legend(loc='best')
plt.title('loss')
plt.xlabel('epoch')
plt.show()

plt.plot(epochs, history.history['acc'], color='blue', label="training_accuracy")
plt.plot(epochs, history.history['val_acc'], color='red',label="validation_accuracy")
plt.legend(loc='best')
plt.title('accuracy')
plt.xlabel('epoch')
plt.show()


# In[ ]:


df_test = pd.read_csv(os.path.join(base_dir, "sample_submission.csv"))
print(df_test.head())
test_images = []
images = df_test['id'].values

for image_id in images:
    test_images.append(cv2.imread(os.path.join(test_dir, image_id)))
    
test_images = np.asarray(test_images)
test_images = test_images / 255.0
print(len(test_images))


# In[ ]:


pred = model.predict(test_images)
df_test['has_cactus'] = pred
df_test.to_csv('aerial-cactus-submission.csv', index = False)


# ## Acknowledgement
# To successfully complete this playground challenge I viewed many publicly available kernels. However, some of them were particularly helpful as I learned a lot from them. Here is a special mention to them:
# https://www.kaggle.com/anirudhchak/cnn-using-keras
# https://www.kaggle.com/frlemarchand/simple-cnn-using-keras
# 

# ### Note
# This is my first kernel ever. If you reached till the end then I hope you liked it. I have tried to explain everything well.
