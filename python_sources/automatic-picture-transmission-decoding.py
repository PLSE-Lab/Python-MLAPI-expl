#!/usr/bin/env python
# coding: utf-8

# # This example kernel trains APT-picture creation

# Yes I know it is possible to use gnu radio to demodulate the signal to an image.
# 
# This here is just for fun and for learning ...

# In[ ]:


GOOGLE_DRIVE = True
try:
  from google.colab import drive
  drive.mount('/content/gdrive')
  print("using google drive")
except:
  GOOGLE_DRIVE = False


# In[ ]:


# just make shure those are installed (incase we run on gpu)
#!pip install IPython numpy
#!pip show numpy IPython


# In[ ]:


if GOOGLE_DRIVE:
    # for google colab only ...
    get_ipython().system('cd /content && rm input 2>/dev/null && ln -s "/content/gdrive/My Drive/colab" /content/input && mkdir /content/blah 2>/dev/null')
    get_ipython().run_line_magic('cd', '/content/blah')
    get_ipython().system('pwd')
    get_ipython().system('ls -la /content/ && ls /content/input/ && ls /content/input/noa19wavpng && ls "/content/gdrive/My Drive"')


# In[ ]:


# This is optional, we don't need it after commit, but is helpful during testing
# Bad thing is it requires internet ... just ignore this if it is failing.
#!pip install livelossplot


# In[ ]:


CHANNELS = 256 # 256 different levels for a pixel
WIDTH = 2080.0
lr = 0.0021 # the learning rate
SKIP_LINES = 1100.0 # there is known trash at the beginning, so out3.png starts here
KEEP_LINES = 99.0


# In[ ]:


KERNEL_RUN_TYPE = "Interactive"
#KERNEL_RUN_TYPE = "Long Run"


# In[ ]:


import os
if 'KAGGLE_KERNEL_RUN_TYPE' in os.environ:
  KERNEL_RUN_TYPE = os.environ['KAGGLE_KERNEL_RUN_TYPE']
RUN_TYPE_INTERACTIVE = False
if KERNEL_RUN_TYPE == "Interactive":
    RUN_TYPE_INTERACTIVE = True


# # Imports

# In[ ]:


import tensorflow as tf
if RUN_TYPE_INTERACTIVE:
   try:
       from livelossplot import PlotLossesKeras
       LIVE_PLOT_AVAILABLE = True
   except:
       LIVE_PLOT_AVAILABLE = False
else:
    LIVE_PLOT_AVAILABLE = False
import keras
import numpy as np
import pandas as pd
import PIL.Image
import cv2
import IPython.display
import os
import matplotlib.pyplot as plt
import wave
import math
from keras import datasets, layers, models
from keras.layers import Conv1D, Dropout, Flatten, Dense
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from struct import unpack
from math import *
from random import *


# # Read original image part

# In[ ]:


def showarray(img):
    from IPython.display import Image
    _,ret = cv2.imencode('.png', img) 
    return IPython.display.display(Image(data=ret.tobytes()))


# In[ ]:


import imageio
out = imageio.imread('../input/noa19wavpng/out3.png')
out = out[0:int(KEEP_LINES),:,:]
print(f"out.shape: {out.shape}")


# In[ ]:


outX = out.copy()
outf = outX.reshape(-1, outX.shape[-1])
outX = None
print(f"outf.shape: {outf.shape}")
print(f"outf: {outf}")

outf2 = np.array([a for a,_,_ in outf])
outf2 = outf2.reshape((outf2.shape[0],1))
print(f"outf2.shape: {outf2.shape}")
print(f"outf2: {outf2}")
outf = None


# In[ ]:


print(f"out.shape: {out.shape}")
outf3 = outf2.copy()
outf3 = outf3.reshape((out.shape[0],out.shape[1],1))*(1,1,1)
print(f"outf3.shape: {outf3.shape}")
showarray(outf3)


# In[ ]:


wav = wave.open('../input/noa19wavpng/out.wav','rb')
wav.rewind()
frames = wav.getnframes()
samp_rate = wav.getframerate()
print(f"samp_rate: {samp_rate}")
channels = wav.getnchannels()
samplesBin = wav.readframes(frames)
print("samplesBin size: %d" % len(samplesBin))
wav.close()


# In[ ]:


sample_width = samp_rate / (WIDTH * 2.0) # APT -> 2 lines per second


# In[ ]:


#TODO figure out where delay of 26 comes from :-/
#     might be able to calculate it, currently just figured out by comparing prediction image with original
DELAY = 26.0 * sample_width
LEN_SHORT = 2


# In[ ]:


from struct import unpack
X = int(SKIP_LINES * WIDTH * sample_width - DELAY)
x1 = X * channels * LEN_SHORT 
x2 = int(sample_width + SKIP_LINES * WIDTH * sample_width + out.shape[0] * out.shape[1] * sample_width - DELAY) * channels * LEN_SHORT 
samplesOrg = unpack(f"{int((x2-x1) / LEN_SHORT)}h",samplesBin[x1:x2])
samples = [float(val) / pow(2, 15) for val in samplesOrg]
samples = np.array(samples)
if channels != 1:
  samples = samples[np.mod(np.arange(samples.size),channels)==0]
samples=samples.astype(np.float32)
print(f"samples.shape: {samples.shape}")


# In[ ]:


print(f"sample_width: {sample_width}")
print(f"out pixels: {out.shape[0]*out.shape[1]}")
print(f"samples.shape: {samples.shape}")
print(f"samples/pixels: {sample_width}")


# In[ ]:


sample_chunks = []
p = 0.0
fixed_sample_width = int(sample_width)
print(f"fixed_sample_width: {fixed_sample_width}")
ln = out.shape[0]*out.shape[1]
while p <= samples.shape[0]:
    ip = int(p)
    s = samples[ip:ip+fixed_sample_width]
    if(s.shape[0]!=fixed_sample_width):
        break
    sample_chunks.append(s)
    if(len(sample_chunks)>ln):
        break
    p += sample_width
sample_chunks = np.array(sample_chunks)
print(f"sample_chunks.shape:{sample_chunks.shape}")
sample_chunks = sample_chunks[:ln]
print(f"sample_chunks.shape:{sample_chunks.shape}")


# In[ ]:


kernel_size=int(sample_width-5)
X_train,X_test,y_train,y_test = train_test_split(sample_chunks,outf2,shuffle=True,test_size=0.25)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))


# In[ ]:


print(f"X_train.shape: {X_train.shape}")
print(f"X_test.shape: {X_test.shape}")


# # Timesink graph

# In[ ]:


plt.plot(np.abs(samples[0:int(20*sample_width)]),figure=plt.figure(figsize=(32 , 5)))
plt.title('Part #' ,color='#ff0000')
plt.ylabel('DB',color='#ff0000')
plt.xlabel('T',color='#ff0000')
plt.legend(['#'], loc='upper left')
plt.show()


# # Part images

# In[ ]:


for i in range(0,10):
     plt.plot(np.abs(X_train[i]),figure=plt.figure(figsize=(4, 2)))
     plt.title('Part #%d %d MAX:%f ARGMAX:%d' % (i,y_train[i][0],np.max(np.absolute(X_train[i])),np.argmax(np.absolute(X_train[i]))),color='#ff0000')
     plt.ylabel('DB',color='#ff0000')
     plt.xlabel('T',color='#ff0000')
     plt.legend(['#%d %d' % (i,y_train[i][0])], loc='upper left')
     plt.show()


# Sometimes the sample do not match the window ... sample rate might not be 100% correct.
# Not shure what causes this, but does not seem to have big impact.

# In[ ]:


X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2]))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], X_test.shape[2]))


# In[ ]:


input_shape = (X_train.shape[1],X_train.shape[2])
print(f"y_train.shape: {y_train.shape}")
print(f"y_test.shape: {y_test.shape}")
print(f"X_train.shape: {X_train.shape}")
print(f"X_test.shape: {X_test.shape}")
print(f"input_shape.shape: {input_shape}")


# In[ ]:


model = models.Sequential()
model.add(Conv1D(filters=4,input_shape=input_shape, kernel_size=kernel_size, activation='relu'))
model.add(Conv1D(filters=CHANNELS, kernel_size=kernel_size, activation='relu'))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(CHANNELS, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer=keras.optimizers.Adam(lr = lr), metrics=['sparse_categorical_accuracy'])


# # Model summary

# In[ ]:


model.build((None,X_train.shape[1],X_train.shape[2]))
model.summary()


# In[ ]:


print("KAGGLE_KERNEL_RUN_TYPE: %s" % KERNEL_RUN_TYPE)
if RUN_TYPE_INTERACTIVE:
    epochs  = 25
else:
    epochs  = 60
if LIVE_PLOT_AVAILABLE:
    cb = [PlotLossesKeras()]
else:
    cb = []

history = model.fit(np.absolute(X_train), y_train, epochs=epochs, shuffle=True, validation_split=0.5, batch_size=256,callbacks=cb)


# # Progress charts

# Don't get confused about bad accuracy ... the result picture is still close enough to the original (even with only 25 epochs) ;-)

# In[ ]:


fig = plt.figure(figsize=(20, 10))
# Plot training & validation accuracy values
if 'acc' in history.history:
    plt.plot(history.history['acc'],figure=fig)
if 'categorical_accuracy' in history.history:
    plt.plot(history.history['categorical_accuracy'],figure=fig)
if 'sparse_categorical_accuracy' in history.history:
    plt.plot(history.history['sparse_categorical_accuracy'],figure=fig)
if 'val_acc' in history.history:
    plt.plot(history.history['val_acc'],figure=fig)
if 'val_categorical_accuracy' in history.history:
    plt.plot(history.history['val_categorical_accuracy'],figure=fig)
if 'val_sparse_categorical_accuracy' in history.history:
    plt.plot(history.history['val_sparse_categorical_accuracy'],figure=fig)
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train Acc', 'Test Acc'], loc='upper left')
plt.show()

fig = plt.figure(figsize=(20, 10))
# Plot training & validation loss values
plt.plot(history.history['loss'],figure=fig)
plt.plot(history.history['val_loss'],figure=fig)
plt.title('Model loss')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train Loss', 'Test Loss'], loc='upper left')
plt.show()


# # Read full data

# In[ ]:


samplesOrg = unpack(f"{int(len(samplesBin) / LEN_SHORT)}h",samplesBin)
samples = [float(val) / pow(2, 15) for val in samplesOrg]
samples = np.array(samples)
if channels != 1:
  samples = samples[np.mod(np.arange(samples.size),channels)==0]
samples = samples.astype(np.float32)


# # Cleanup

# ... because we need the memory

# In[ ]:


samplesOrg = None
samplesBin = None
out3 = None
outf = None
X_train = None
X_trainC = None
y_train = None


# In[ ]:


import gc
gc.collect()


# # Generate chunks

# In[ ]:


sample_chunks = []
p = 0.0
n = 0
border = int(sample_width/6)
fixed_sample_width = int(sample_width)
print(f"samples.shape: {samples.shape}")
while p <= samples.shape[0]:
    ip = int(p)
    sw = fixed_sample_width
    h = 0
    if (n % (10)) == 0:
        s = np.absolute(samples[ip+border:ip+sw-border])
        #reduce doppler shift, wish this would work better :-/
        if np.argmax(s) == int(s.shape[0]/2):
            pass
        elif np.argmax(s) > s.shape[0]/2:
            h = 0.025
        else:
            h = -0.025
            sw = int(sample_width - 0.025)
    ss = samples[ip:ip + sw]
    if(ss.shape[0] != sw):
        break
    s = np.append(ss,np.zeros(fixed_sample_width - ss.shape[0]))
    sample_chunks.append(s)
    p += sample_width + h
    n += 1
sample_chunks = np.array(sample_chunks)
ln = len(sample_chunks)
ln = int(ln - (ln % WIDTH))
sample_chunks = sample_chunks[:ln]
train_samples = sample_chunks
print(f"sample_chunks.shape: {sample_chunks.shape}")


# # Reshaping

# In[ ]:


sample_chunks = np.reshape(sample_chunks,(sample_chunks.shape[0],train_samples.shape[1],1))
print(f"sample_chunks.shape: {sample_chunks.shape}")
input_shape = (sample_chunks.shape[1],1)


# # Create complete predicted image

# In[ ]:


pred = model.predict(np.absolute(sample_chunks),use_multiprocessing=True)
img = np.array([np.argmax(p) for p in pred])


# In[ ]:


ln = img.shape[0]
ln -= int(ln % WIDTH)
subimg = img[0:ln]
img2 = subimg.reshape(subimg.shape[0],1)
print(f"out.shape: {out.shape}")
outf3 = img2.reshape((int(ln/WIDTH),int(WIDTH),1))*(1,1,1)
print(f"outf3.shape:{outf3.shape}")
showarray(outf3)


# # Create model files

# In[ ]:


model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("model.h5")


# # Create confusion matrix

# In[ ]:


y_true = pd.Series(outf2.flatten(), name="Actual")
y_pred = pd.Series(img.flatten(), name="Predicted")
df_confusion = pd.crosstab(y_true, y_pred)
df_confusion.to_csv('confusion_matrix.csv')


# # Create download links

# In[ ]:


if GOOGLE_DRIVE:
  get_ipython().system('mkdir ../input/output 2>/dev/null')
  get_ipython().system('cp model* ../input/output/')
  get_ipython().system('cp confusion_matrix.csv ../input/output/')
  sys.exit(0)


# In[ ]:


from IPython.display import FileLink, display
display(FileLink('model.h5'))
display(FileLink('model.json'))
display(FileLink('confusion_matrix.csv'))

