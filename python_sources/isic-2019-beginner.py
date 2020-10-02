#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import keras
from keras.models import Sequential
from keras_preprocessing.image import ImageDataGenerator
from keras.layers import Input,Conv2D,Dense,Dropout,Flatten,MaxPooling2D,AveragePooling2D,Activation,BatchNormalization,GlobalAveragePooling2D
import pandas as pd
import numpy as np

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


train_path = "/kaggle/input/isic-2019/ISIC_2019_Training_Input/ISIC_2019_Training_Input/"
test_path = "/kaggle/input/isic-2019/ISIC_2019_Test_Input/ISIC_2019_Test_Input/"
ground_truth_path = '../input/isic-2019/Training_GroundTruth (1).csv'


# In[ ]:


df = pd.read_csv(ground_truth_path)
df.head()


# In[ ]:


def name(row):
    row.image = row.image + '.jpg'
    return row

df = df.apply(name,axis = 1)


# In[ ]:


columns = list(df.columns)[1:]
print(columns)


# In[ ]:


from sklearn.utils import shuffle
df = shuffle(df,random_state = 10)
df.head()


# In[ ]:


def show_values_on_bars(axs, h_v="v", space=0.4):
    def _show_on_single_plot(ax):
        if h_v == "v":
            for p in ax.patches:
                _x = p.get_x() + p.get_width() / 2
                _y = p.get_y() + p.get_height()+5
                value = int(p.get_height())
                ax.text(_x, _y, value, ha="center") 
        elif h_v == "h":
            for p in ax.patches:
                _x = p.get_x() + p.get_width() + float(space)
                _y = p.get_y() + p.get_height()+5
                value = int(p.get_width())
                ax.text(_x, _y, value, ha="left")

    if isinstance(axs, np.ndarray):
        for idx, ax in np.ndenumerate(axs):
            _show_on_single_plot(ax)
    else:
        _show_on_single_plot(axs)


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
fig = plt.gcf()
fig.set_size_inches(12, 8)
sns.set()
plt.title('Class Count')
data = df[columns].sum()
pal = sns.cubehelix_palette(9,reverse = True)
rank = data.values.argsort().argsort() 
#g=sns.barplot(x='day',y='tip',data=groupedvalues, palette=np.array(pal[::-1])[rank])
br = sns.barplot(data.index,data.values,palette=np.array(pal[::-1])[rank])
show_values_on_bars(br, "v", 1.0)
sns.set()


# In[ ]:


train_datagen = ImageDataGenerator(rescale=1./255.,
                                  horizontal_flip = True,
                                  width_shift_range = 0.2,
                                  height_shift_range = 0.2,
                                  fill_mode = 'nearest',
                                  zoom_range = 0.3,
                                  rotation_range = 30
                                  )
valid_datagen = ImageDataGenerator(rescale=1./255.)

train_generator = train_datagen.flow_from_dataframe(dataframe = df[:22800],
                                              directory = train_path,
                                              x_col = "image",
                                              y_col = columns,
                                              batch_size = 128,
                                              seed = 42,
                                              shuffle = True,
                                              class_mode = "raw",
                                              target_size = (100,100)
                                              #validate_filenames = False
                                              )

#If class_mode="multi_output", y_col must be a list. Received Index.
valid_generator = valid_datagen.flow_from_dataframe(dataframe=df[22800:25231],
                                                   directory = train_path,
                                                   x_col= "image",
                                                   y_col=columns,
                                                   batch_size=128,
                                                   seed=42,
                                                   shuffle=True,
                                                   class_mode="raw",
                                                   target_size=(100,100),
                                                   validate_filenames = True
                                                   )

test_generator = valid_datagen.flow_from_dataframe(dataframe=df[25231:],
                                                   directory = train_path,
                                                   x_col="image",
                                                   #y_col=columns,
                                                   batch_size=1,
                                                   seed=42,
                                                   shuffle=True,
                                                   class_mode=None,
                                                   target_size=(100,100),
                                                   validate_filenames = True
                                                   )


# In[ ]:


keras.__version__


# In[ ]:


model = Sequential()
model.add(Conv2D(32, (3, 3),input_shape=(100,100,3)))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Conv2D(32, (3, 3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3)))
model.add(Conv2D(64, (3, 3)))
model.add(Conv2D(64,(3,3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64,(3,3)))
model.add(Conv2D(64,(3,3)))
model.add(Conv2D(64,(3,3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))

model.add(Conv2D(128,(3,3)))
model.add(Conv2D(128,(3,3)))
model.add(Conv2D(128,(3,3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(128,(3,3)))
model.add(Conv2D(128,(3,3)))
model.add(Conv2D(128,(3,3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Conv2D(256,(3,3)))
model.add(Conv2D(256,(3,3)))
model.add(Conv2D(256,(3,3)))
model.add(Conv2D(256,(3,3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(GlobalAveragePooling2D())
#model.add(Flatten())

model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.25))

model.add(Dense(1024))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(9, activation='sigmoid'))
model.compile(keras.optimizers.rmsprop(lr=0.0001, decay=1e-6),loss="binary_crossentropy",metrics=["accuracy"])


# In[ ]:


from keras.utils import plot_model
from IPython.display import Image 
plot_model(model,to_file = 'multi_output_model_2.png',show_shapes=True)
pil_img = Image('multi_output_model_2.png')
display(pil_img)


# In[ ]:


model.summary()


# In[ ]:


os.mkdir('History')
os.mkdir('Models')


# In[ ]:


from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.callbacks import *
   
#  class myCallback(keras.callbacks.Callback):
#    def on_epoch_end(self,epoch,logs={}):
#      if(logs.get('val_accuracy')>.95):
#         print('\nReached 95% accuracy so cancelling training!')
#         self.model.stop_training = True

#  callback = myCallback()

filepath = 'Models/multi_output_model_7_weights-{epoch:02d}-{val_accuracy:.3f}.hdf5'
checkpoint1 = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
#checkpoint2 = MyLogger()
checkpoint3 = CSVLogger('History/Log_model_7.csv')
callbacks_list = [checkpoint1 , checkpoint3]
#  callbacks_list = [checkpoint,callback]


# In[ ]:


STEP_SIZE_TRAIN = train_generator.n//train_generator.batch_size
STEP_SIZE_VALID = valid_generator.n//valid_generator.batch_size
STEP_SIZE_TEST = test_generator.n//test_generator.batch_size
history = model.fit_generator(generator=train_generator,
                              steps_per_epoch=STEP_SIZE_TRAIN,
                              validation_data=valid_generator,
                              validation_steps=STEP_SIZE_VALID,
                              epochs=10,
                              callbacks = callbacks_list
                              )


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc=0)
plt.figure()

plt.plot(epochs, loss, 'r', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation Loss')
plt.legend(loc=0)
plt.figure()


# In[ ]:




