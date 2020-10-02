#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


get_ipython().system('ls /kaggle/input')


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
from memory_profiler import memory_usage
import os
import pandas as pd
from glob import glob
import numpy as np
import os
import numpy as np
import pandas as pd
import soundfile as sf
import scipy.signal as signal
import matplotlib.pyplot as plt
import gc
import IPython.display as ipd 


# In[ ]:


from keras import layers
from keras import models
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
import keras.backend as K
import librosa
import librosa.display
import pylab
import matplotlib.pyplot as plt
from matplotlib import figure
import gc
from path import Path


# In[ ]:


get_ipython().system('mkdir /kaggle/input/train')
get_ipython().system('mkdir /kaggle/input/test')


# In[ ]:


#!rm /kaggle/working/train/*


# In[ ]:


import os
print(os.listdir('../input'))
import pandas as pd
import numpy as np


# In[ ]:


def create_spectrogram(filename,name):
    plt.interactive(False)
    clip, sample_rate = librosa.load(filename, sr=None)
    fig = plt.figure(figsize=[0.72,0.72])
    ax = fig.add_subplot(111)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_frame_on(False)
    S = librosa.feature.melspectrogram(y=clip, sr=sample_rate)
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
    filename  = '/kaggle/input/train/' + name + '.jpg'
    plt.savefig(filename, dpi=400, bbox_inches='tight',pad_inches=0)
    plt.close()    
    fig.clf()
    plt.close(fig)
    plt.close('all')
    del filename,name,clip,sample_rate,fig,ax,S


# In[ ]:


def create_spectrogram_test(filename,name):
    plt.interactive(False)
    clip, sample_rate = librosa.load(filename, sr=None)
    fig = plt.figure(figsize=[0.72,0.72])
    ax = fig.add_subplot(111)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_frame_on(False)
    S = librosa.feature.melspectrogram(y=clip, sr=sample_rate)
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
    filename  = Path('/kaggle/input/test/' + name + '.jpg')
    fig.savefig(filename, dpi=400, bbox_inches='tight',pad_inches=0)
    plt.close()    
    fig.clf()
    plt.close(fig)
    plt.close('all')
    del filename,name,clip,sample_rate,fig,ax,S


# In[ ]:


Data_dir=np.array(glob("../input/spoken-language-identification/train/train/*"))


# In[ ]:


Data_dir.shape


# In[ ]:


get_ipython().run_line_magic('load_ext', 'memory_profiler')


# In[ ]:


get_ipython().run_cell_magic('time', '', "i=0\nlanguage_label = []\nfor file in Data_dir[i:i+2000]:\n    name = file.split('/')[-1]\n    language_label.append(name)\n    create_spectrogram(file,name)\n    print(name)")


# In[ ]:


gc.collect()


# In[ ]:


#!ls train


# In[ ]:


#from IPython.display import Image
#Image(filename='../working/train/de_f_1996a0f045b3301946a9194dfad545ab.fragment28.noise4.flac.jpg') 


# In[ ]:


get_ipython().run_cell_magic('time', '', "i=2000\nfor file in Data_dir[i:i+2000]:\n    name = file.split('/')[-1]\n    language_label.append(name)\n    create_spectrogram(file,name)\n    print(name)")


# In[ ]:


gc.collect()


# In[ ]:


get_ipython().run_cell_magic('time', '', "i=4000\nfor file in Data_dir[i:i+2000]:\n    name = file.split('/')[-1]\n    language_label.append(name)\n    create_spectrogram(file,name)\n    print(name)")


# In[ ]:


gc.collect()


# In[ ]:


get_ipython().run_cell_magic('time', '', "i=6000\nfor file in Data_dir[i:i+2000]:\n    name = file.split('/')[-1]\n    language_label.append(name)\n    create_spectrogram(file,name)\n    print(name)\n    \ngc.collect()")


# In[ ]:


get_ipython().run_cell_magic('time', '', "i=8000\nfor file in Data_dir[i:i+2000]:\n    name = file.split('/')[-1]\n    language_label.append(name)\n    create_spectrogram(file,name)\n    print(name)\n    \ngc.collect()")


# In[ ]:


get_ipython().run_cell_magic('time', '', "i=10000\nfor file in Data_dir[i:i+2000]:\n    name = file.split('/')[-1]\n    language_label.append(name)\n    create_spectrogram(file,name)\n    print(name)\n    \ngc.collect()")


# In[ ]:


label = []
for i in range(len(language_label)):
    label.append(language_label[i].split('_')[0])


# In[ ]:


traindf=pd.DataFrame(language_label,columns = ['ID'])


# In[ ]:


traindf['Class'] = label


# In[ ]:


def append_ext(fn):
    return fn+".jpg"


# In[ ]:


traindf["ID"]=traindf["ID"].apply(append_ext)


# In[ ]:


get_ipython().run_cell_magic('time', '', "i=12000\ntest_label = []\nfor file in Data_dir[i:i+1000]:\n    name = file.split('/')[-1]\n    test_label.append(name)\n    create_spectrogram_test(file,name)\n    print(name)")


# In[ ]:


gc.collect()


# In[ ]:


testdf=pd.DataFrame(test_label,columns = ['ID'])
testdf["ID"]=testdf["ID"].apply(append_ext)

label_t = []
for i in range(len(test_label)):
    label_t.append(test_label[i].split('_')[0])

testdf['Class'] = label_t


# In[ ]:


from keras_preprocessing.image import ImageDataGenerator


# In[ ]:


datagen=ImageDataGenerator(rescale=1./255.,validation_split=0.2)


train_generator=datagen.flow_from_dataframe(
    dataframe=traindf,
    directory="/kaggle/working/train/",
    x_col="ID",
    y_col="Class",
    subset="training",
    batch_size=32,
    seed=42,
    shuffle=True,
    class_mode="categorical",
    target_size=(64,64))

valid_generator=datagen.flow_from_dataframe(
    dataframe=traindf,
    directory="/kaggle/working/train/",
    x_col="ID",
    y_col="Class",
    subset="validation",
    batch_size=32,
    seed=42,
    shuffle=True,
    class_mode="categorical",
    target_size=(64,64))


# In[ ]:


from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D
from keras import regularizers, optimizers
import pandas as pd
import numpy as np

model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=(64,64,3)))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(Conv2D(128, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(3, activation='softmax'))
model.compile(optimizers.rmsprop(lr=0.0005, decay=1e-6),loss="categorical_crossentropy",metrics=["accuracy"])
model.summary()


# In[ ]:


STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size
model.fit_generator(generator=train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=valid_generator,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=150
)
model.evaluate_generator(generator=valid_generator, steps=STEP_SIZE_VALID
)


# In[ ]:


model.save('../input/Model_CNN.h5')


# In[ ]:


gc.collect()


# In[ ]:


test_datagen=ImageDataGenerator(rescale=1./255.)
test_generator=test_datagen.flow_from_dataframe(
    dataframe=testdf,
    directory="/kaggle/working/test/",
    x_col="ID",
    y_col=None,
    batch_size=32,
    seed=42,
    shuffle=False,
    class_mode=None,
    target_size=(64,64))
STEP_SIZE_TEST=test_generator.n//test_generator.batch_size


# In[ ]:


test_generator.reset()
pred=model.predict_generator(test_generator,
steps=STEP_SIZE_TEST,
verbose=1)
predicted_class_indices=np.argmax(pred,axis=1)


# In[ ]:


predicted_class_indices.shape


# In[ ]:


get_ipython().system('ls test')


# In[ ]:


from PIL import Image
import numpy as np
from skimage import transform
def load(filename):
    np_image = Image.open(filename)
    np_image = np.array(np_image).astype('float32')/255
    np_image = transform.resize(np_image, (64, 64, 3))
    np_image = np.expand_dims(np_image, axis=0)
    return np_image

image = load('../working/test/de_f_5d2e7f30d69f2d1d86fd05f3bbe120c2.fragment30.noise5.flac.jpg')
model.predict(image)


# In[ ]:


train_generator.class_indices


# In[ ]:


get_ipython().system('mkdir /kaggle/input/train')


# In[ ]:


get_ipython().system('mkdir /kaggle/input/test')


# In[ ]:


get_ipython().system('cp /kaggle/working/train/* /kaggle/input/train')


# In[ ]:


get_ipython().system('cp /kaggle/working/test/* /kaggle/input/test')


# In[ ]:


get_ipython().system('gzip -r /kaggle/input/train')


# In[ ]:


get_ipython().system('gzip -r /kaggle/input/test')


# In[ ]:


get_ipython().system('ls /kaggle/input')


# In[ ]:


def create_spectrogram_new(filename,name):
    plt.interactive(False)
    clip, sample_rate = librosa.load(filename, sr=None)
    fig = plt.figure(figsize=[0.72,0.72])
    ax = fig.add_subplot(111)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_frame_on(False)
    S = librosa.feature.melspectrogram(y=clip, sr=sample_rate)
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
    filename  = '/kaggle/working/test/' + name + '.jpg'
    plt.savefig(filename, dpi=400, bbox_inches='tight',pad_inches=0)
    plt.close()    
    fig.clf()
    plt.close(fig)
    plt.close('all')
    del filename,name,clip,sample_rate,fig,ax,S


# In[ ]:


get_ipython().system('ls /kaggle/input/')


# In[ ]:


get_ipython().system('cp /kaggle/input/audio/spanish25.wav /kaggle/working/test')


# In[ ]:


filename = '/kaggle/working/test/spanish25.wav'
name = 'spanish'


# In[ ]:


create_spectrogram_new(filename,name)


# In[ ]:


get_ipython().system('ls /kaggle/working/test')


# In[ ]:


from IPython.display import Image
Image(filename='../working/test/spanish.jpg') 


# In[ ]:


from PIL import Image
import numpy as np
from skimage import transform
def load(filename):
    np_image = Image.open(filename)
    np_image = np.array(np_image).astype('float32')/255
    np_image = transform.resize(np_image, (64, 64, 3))
    np_image = np.expand_dims(np_image, axis=0)
    return np_image

image = load('../working/test/spanish.jpg')
model.predict(image)


# In[ ]:


get_ipython().system('ls')


# In[ ]:


get_ipython().system('ls /kaggle/input')


# In[ ]:


get_ipython().run_cell_magic('capture', '', '!apt-get install zip\n!zip -r /kaggle/input/train.zip /kaggle/input/train/\n!zip -r /kaggle/input/test.zip /kaggle/input/test/')


# In[ ]:


get_ipython().system('mv /kaggle/working/test.zip /kaggle/input')


# In[ ]:


get_ipython().system('rm -r /kaggle/input/train')
get_ipython().system('rm -r /kaggle/input/test')

