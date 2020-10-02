#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf
from tensorflow.keras import callbacks
from tensorflow.keras.layers import Dropout, Flatten, Dense, LSTM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import Model
from tensorflow.keras.initializers import Constant
import pandas as pd
import os
from pathlib import Path
from tensorflow.keras.callbacks import Callback

import numpy as np
import pandas as pd
import numpy as np
import librosa.display
from scipy import signal
import scipy.io.wavfile
import IPython.display as ipd
import matplotlib.pyplot as plt
import os

import math
from scipy.io.wavfile import write
from tensorflow.keras import activations
from keras.models import model_from_json
from keras.regularizers import l2

from math import pi
from math import cos
from math import floor
from keras import backend


# In[ ]:


def dataretreival(filename1,filename2,filename3):
    # Data1

    with open("../input/data-full/"+str(filename1), "r") as read_it: 
         data1 = json.load(read_it) 
    X1=np.array(data1["features"])
    y1=np.array(data1["classes"])

    # Data2
    with open("../input/data-full/"+str(filename2), "r") as read_it: 
         data2 = json.load(read_it) 
    X2=np.array(data2["features"])
    y2=np.array(data2["classes"])

    '''
    # Data2
    with open("../input/data-full/"+str(filename3), "r") as read_it: 
         data3 = json.load(read_it) 
    X2=np.array(data3["features"])
    y2=np.array(data3["classes"])
    '''

    
    X=np.concatenate((X1,X2),axis=0)
    y=np.concatenate((y1,y2),axis=0)
    print(X.shape)
    print(y.shape)
    return X,y


def prepare_data(test_size,model_type="base"):
    
    # Load the data
    X,y=dataretreival("data.json","data_2.json","data_3.json")
    
    #print(np.max(y))

    # Create train/Test Split 
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=test_size)
    
    plt.figure()    
    plt.hist(y_train,bins=100)
    plt.show()
    
    y_train=to_categorical(y_train)
    y_test=to_categorical(y_test)
    
    #print(y_test.shape)
    # Currently 3D array for each sample (N, S1, S2)--> 4D Array (N, S1, S2,Depth)
    if model_type=="base":
        X_train=X_train[...,np.newaxis] # 4D Array 
        X_test=X_test[...,np.newaxis]
        return X_train,X_test, y_train, y_test, np.max(y)
    elif model_type=="vgg":
        inp_shape=(48,48,3)
        X_train=np.repeat(X_train[..., np.newaxis], 3, -1)
        X_test=np.repeat(X_train[..., np.newaxis], 3, -1)
        X_train_resize=[]
        X_test_resize=[]
        for img in X_train:
            X_train_resize.append(np.resize(img,inp_shape))
        X_train_resize=np.array(X_train_resize)
        for img in X_test:
            X_test_resize.append(np.resize(img,inp_shape))
        X_test_resize=np.array(X_test_resize)
        return X_train_resize,X_test_resize, y_train, y_test, np.max(y)
    else:
        return X_train,X_test, y_train, y_test, np.max(y)
    

def build_model_cnn(input_shape,class_labels):
    
    # Create Model
    model=keras.Sequential()
    
    # 1st Conv Layer
    model.add(keras.layers.Conv2D(16,(7,7),padding='same',                                  input_shape=input_shape))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation(activations.relu))
    model.add(keras.layers.MaxPool2D((3,3),strides=(2,2),padding="same"))
    
    # 2nd Conv Layer
    model.add(keras.layers.Conv2D(32,(5,5),padding='same'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation(activations.relu))
    model.add(keras.layers.MaxPool2D((3,3),strides=(2,2),padding="same"))
    
    # 3rd Conv Layer
    model.add(keras.layers.Conv2D(32,(3,3),padding='same'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation(activations.relu))
    model.add(keras.layers.MaxPool2D((3,3),strides=(2,2),padding="same"))
    
    # 4th Conv Layer
    model.add(keras.layers.Conv2D(64,(3,3),padding='same'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation(activations.relu))
    model.add(keras.layers.MaxPool2D((3,3),strides=(2,2),padding="same"))
    
    # 5th Conv Layer
    model.add(keras.layers.Conv2D(64,(3,3),padding='same'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation(activations.relu))
    model.add(keras.layers.MaxPool2D((3,3),strides=(2,2),padding="same"))
    
    # 6th Conv Layer
    model.add(keras.layers.Conv2D(128,(3,3),padding='same'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation(activations.relu))
    model.add(keras.layers.MaxPool2D((3,3),strides=(2,2),padding="same"))  

   
    # Flatten
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(1028))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation(activations.relu))
    model.add(keras.layers.Dropout(0.5))
    
  
    # Output
    model.add(keras.layers.Dense(class_labels,activation='softmax'))    
    return model

def rnn(input_shape,class_labels):
    
    # Create Model
    model=keras.Sequential()
    
    model.add(keras.layers.LSTM(64,input_shape,return_sequences=True))
    model.add(keras.layers.LSTM(64))
    model.add(keras.layers.Dense(64,activation='relu'))
    model.add(keras.layers.Dropout(0.5))  
    
    # Output
    model.add(keras.layers.Dense(class_labels,activation='softmax'))    
    return model



def vgg_model(input_shape,class_labels):
    base_model=VGG16(weights='imagenet',include_top=False,input_shape=input_shape)
    
    for layer in base_model.layers:
        if layer.name ==  'block5_conv1':
            break
        layer.trainable=False
        print('Layer '+layer.name+'frozen.')
    last=base_model.layers[-1].output
    x=Flatten()(last)
    x=Dense(1000,activation='relu',name='fc1')(x)
    x=Dropout(0.3)(x)
    x=Dense(class_labels,activation='softmax',name='predictions')(x)
    model=Model(base_model.input,x)
    return model


    


# In[ ]:


# IF CNN, type="base" & RNN, type="rnn"
# Split the data into train & test 
model_type="rnn"
X_train,X_test, y_train, y_test,label_size=prepare_data(test_size=0.2,model_type=model_type)
label_size=label_size+1
print(X_train.shape,X_test.shape,label_size)


# # Cosine Annealing for CNN only

# In[ ]:



class CosineAnnealingLearningrateschedule(Callback):
    def __init__(self,n_epochs,n_cycles,lrate_max,verbose=0):
        self.epochs=n_epochs
        self.cycles=n_cycles
        self.lr_max=lrate_max
        self.lrates=list()
        
    # calculate learning rate for each epoch
    def cosineannealing(self,epoch,n_epochs,n_cycles,lrate_max):
        epochs_per_cycle=floor(n_epochs/n_cycles)
        cos_inner=(pi*(epoch%epochs_per_cycle))/epochs_per_cycle
        return lrate_max/2*(cos(cos_inner)+1)
    
    # calculate and set learning rate at the start of the epoch
    def on_epoch_begin(self,epoch,logs=None):
        # calculate learning rate
        lr=self.cosineannealing(epoch,self.epochs,self.cycles,self.lr_max)
        # set learningrate
        backend.set_value(self.model.optimizer.lr,lr)
        # log value
        self.lrates.append(lr)


# # CNN - RNN

# In[ ]:


import tensorflow.keras as keras
#print(X_test.shape,y_test.shape,np.max(X_test[0]))


# Build the CNN 
if ((model_type == "base") or (model_type=="vgg")):
    input_shape=(X_train.shape[1],X_train.shape[2],X_train.shape[3])
    model=build_model_cnn(input_shape,label_size) # Model from scratch
else:
    input_shape=(X_train.shape[1],X_train.shape[2])
    model=rnn(input_shape,label_size) # Model from scratch

#model=vgg_model(input_shape,label_size)

model.summary()

'''
# Compile the model 
optimizer=keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer,loss="categorical_crossentropy",metrics=["accuracy"])
#model.load_weights("../input/model-best/weights-improvement-177-0.57.hdf5")

# Network Parameters
n_epochs=500
batch_Size=128
epochs_per_cycle=50

# Callbacks defintion
filepath="weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5"
checkpointer=ModelCheckpoint(filepath,monitor='val_accuracy',verbose=1,
                             save_best_only=True,mode='max')

n_cycles=n_epochs/epochs_per_cycle
ca=CosineAnnealingLearningrateschedule(n_epochs,n_cycles,0.01)
callbacks_list=[ca,checkpointer]

# Train the model
history=model.fit(X_train,y_train,validation_data=(X_test,y_test),batch_size=batch_Size,\
                  epochs=n_epochs,callbacks=callbacks_list,verbose=2) 


# Generate generalization metrics
score = model.evaluate(X_test, y_test, verbose=0)
print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')

# Saving the model
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
#print("Saved model to disk")

# Visualize history
# Plot history: Loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('loss history')
plt.ylabel('Loss value')
plt.xlabel('No. epoch')
plt.show()

# Plot history: Accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('accuracy history')
plt.ylabel('Accuracy value (%)')
plt.xlabel('No. epoch')
plt.show()
'''


# # RNN

# In[ ]:


import tensorflow.keras as keras
#print(X_test.shape,y_test.shape,np.max(X_test[0]))


# Build the CNN 
input_shape=(X_train.shape[1],X_train.shape[2])
model=rnn_cnn(input_shape,label_size) # Model from scratch

model.summary()


# Compile the model 
optimizer=keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer,loss="categorical_crossentropy",metrics=["accuracy"])

# Network Parameters
n_epochs=500
batch_Size=128
epochs_per_cycle=50

# Callbacks defintion
filepath="weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5"
checkpointer=ModelCheckpoint(filepath,monitor='val_accuracy',verbose=1,
                             save_best_only=True,mode='max')

n_cycles=n_epochs/epochs_per_cycle
ca=CosineAnnealingLearningrateschedule(n_epochs,n_cycles,0.01)
callbacks_list=[ca,checkpointer]

# Train the model
history=model.fit(X_train,y_train,validation_data=(X_test,y_test),batch_size=batch_Size,                  epochs=n_epochs,callbacks=callbacks_list,verbose=2) 


# Generate generalization metrics
score = model.evaluate(X_test, y_test, verbose=0)
print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')

# Saving the model
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
#print("Saved model to disk")

# Visualize history
# Plot history: Loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('loss history')
plt.ylabel('Loss value')
plt.xlabel('No. epoch')
plt.show()

# Plot history: Accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('accuracy history')
plt.ylabel('Accuracy value (%)')
plt.xlabel('No. epoch')
plt.show()


# In[ ]:



#optimizer=keras.optimizers.Adam(learning_rate=0.001)
#model.compile(optimizer=optimizer,loss="categorical_crossentropy",metrics=["accuracy"])
#model.load_weights("weights-improvement-177-0.57.hdf5")
score = model.evaluate(X_test,y_test, verbose=0)
print(score)
y_pred=model.predict(X_test).argmax(axis=1)
y_test1=y_test.argmax(axis=1)
print(y_pred,y_test1)


# In[ ]:


from sklearn.metrics import confusion_matrix
array=confusion_matrix(y_test1, y_pred)


# In[ ]:


from sklearn.metrics import classification_report
print (classification_report(y_test1, y_pred))


# In[ ]:


import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

for k in range(0,91,10):
    df_cm = pd.DataFrame(array[k:k+10,k:k+10], index = [i for i in range(k,k+10)],
                      columns = [i for i in range(k,k+10)])
    plt.figure(figsize = (8,3))
    sn.heatmap(df_cm, annot=True)
    plt.title(f'Confusion ratio for classes {k}-{k+10}')
    plt.show()
    plt.close()
    



# # submission sample

# In[ ]:


# Laoding & Compiling the model 
model.load_weights("weights-improvement-47-0.95.hdf5")
optimizer=keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer,loss="categorical_crossentropy",metrics=["accuracy"])

# Loading the Sample test data 
BASE_TEST_DIR = '../input/birdsong-recognition' if os.path.exists('../input/birdsong-recognition/test_audio') else '../input/birdcall-check'
df_test = pd.read_csv(f'{BASE_TEST_DIR}/test.csv')
sub_test_12 = df_test[df_test.site.isin(['site_1', 'site_2'])]
sub_test_3 = df_test[df_test.site.isin(['site_3'])]
TEST_FOLDER = f'{BASE_TEST_DIR}/test_audio'

# Defining the files for predictions
X_Valid=[]
submission = {'row_id': [], 'birds': []}
birds_all={'amepip':0, 'blkpho':1,'comgra':2,'comred':3,'comter':4,           'gockin':5,'herthr':6,'logshr':7, 'mouchi':8, 'rebwoo':9}

# getting respective bird names from the predictions
def get_key(val): 
    for key, value in birds_all.items(): 
         if val == value: 
             return key  
    return "key doesn't exist"

def predchec(arr):
    # Should store array value
    for ind in len(arr):
        if np.max(arr[ind]) > 0.80:
            return np.argwhere(arr[ind]>0.80)
        elif np.max(arr[] < 0.80 & np.max(arr[] > 0.5)):
            return top two
        else:
            return 99
            
            
# Looping through Site1/site2 & Site3 for predictions
# For Site1 & Site2 (with duration data)
for index, row in sub_test_12.iterrows():
    # Get test row information
    site = row['site']
    start_time = row['seconds'] - 5
    end_time=row['seconds']
    row_id = row['row_id']
    audio_id = row['audio_id']
    DURATION=end_time-start_time
    #print(DURATION)

    # Parameters for MFCC
    hop_length=512
    n_fft=2048
    n_mfcc=13
    #expected_no_mfccvec_per_segment=700

    # Getting the file
    librosa_audio, librosa_sample_rate = librosa.load(f'{TEST_FOLDER}/{audio_id}.mp3')
    SAMPLE_RATE=librosa_sample_rate
    DURATION=DURATION
    SAMPLES_PER_SIGNAL=SAMPLE_RATE*DURATION # Varies as a function of duration if sample rate is fixed
    segment_duration=5 # seconds
    total_samples_reqd_segment=SAMPLE_RATE*segment_duration # Total samples for 5 secs
    #print(SAMPLES_PER_SIGNAL,total_samples_reqd_segment)
    num_samples_per_segment=total_samples_reqd_segment # Should be based on the train set
    expected_no_mfccvec_per_segment=math.ceil(num_samples_per_segment/hop_length)
    #print(expected_no_mfccvec_per_segment)
    num_segments=math.ceil(int(SAMPLES_PER_SIGNAL)/num_samples_per_segment)

    # Get MFCC Features for each of the data
    start_sample_indx=int(start_time*SAMPLE_RATE)
    end_sample_indx=int(start_sample_indx+SAMPLES_PER_SIGNAL)
    mfcc=librosa.feature.mfcc(librosa_audio[start_sample_indx:end_sample_indx],hop_length=hop_length,n_fft=n_fft,n_mfcc=n_mfcc) # Analyzing a slice of a signal 
    mfcc=mfcc.T

    #librosa.display.specshow(mfcc,sr=SAMPLE_RATE,hop_length=hop_length)
    #print(mfcc.shape)
    X_Check=mfcc
    X_Ch=X_Check[...,np.newaxis]
    X_Ch_mod=np.reshape(X_Ch,(-1,X_Ch.shape[0],X_Ch.shape[1],X_Ch.shape[2]))
    y_pred=model.predict(X_Ch_mod)
    
    
    y_pred_max=np.argmax(y_pred,axis=1)
    submission['row_id'].append(row_id)
    submission['birds'].append(get_key(y_pred_max))
    


# For Site3 (No duration data)
for index, row in sub_test_3.iterrows():
    # Get test row information
    site = row['site']
    start_time = 0#row['seconds'] - 5
    end_time=5#row['seconds']
    row_id = row['row_id']
    audio_id = row['audio_id']
    DURATION=end_time-start_time
    #print(DURATION)

    # Parameters for MFCC
    hop_length=512
    n_fft=2048
    n_mfcc=13
    #expected_no_mfccvec_per_segment=700

    # Getting the file
    librosa_audio, librosa_sample_rate = librosa.load(f'{TEST_FOLDER}/{audio_id}.mp3')
    SAMPLE_RATE=librosa_sample_rate
    DURATION=DURATION
    SAMPLES_PER_SIGNAL=SAMPLE_RATE*DURATION # Varies as a function of duration if sample rate is fixed
    segment_duration=5 # seconds
    total_samples_reqd_segment=SAMPLE_RATE*segment_duration # Total samples for 5 secs
    #print(SAMPLES_PER_SIGNAL,total_samples_reqd_segment)
    num_samples_per_segment=total_samples_reqd_segment # Should be based on the train set
    expected_no_mfccvec_per_segment=math.ceil(num_samples_per_segment/hop_length)
    #print(expected_no_mfccvec_per_segment)
    num_segments=math.ceil(int(SAMPLES_PER_SIGNAL)/num_samples_per_segment)

    # Get MFCC Features for each of the data
    start_sample_indx=int(start_time*SAMPLE_RATE)
    end_sample_indx=int(start_sample_indx+SAMPLES_PER_SIGNAL)
    mfcc=librosa.feature.mfcc(librosa_audio[start_sample_indx:end_sample_indx],hop_length=hop_length,n_fft=n_fft,n_mfcc=n_mfcc) # Analyzing a slice of a signal 
    mfcc=mfcc.T
    
    #librosa.display.specshow(mfcc,sr=SAMPLE_RATE,hop_length=hop_length)
    X_Check=mfcc
    X_Ch=X_Check[...,np.newaxis]
    X_Ch_mod=np.reshape(X_Ch,(-1,X_Ch.shape[0],X_Ch.shape[1],X_Ch.shape[2]))
    y_pred=model.predict(X_Ch_mod)
    y_pred_max=np.argmax(y_pred,axis=1)
    submission['row_id'].append(row_id)
    submission['birds'].append(get_key(y_pred_max))
    
submission=pd.DataFrame(submission)
submission.head()
submission.to_csv('submission.csv', index=False)


# In[ ]:


submission.shape


# In[ ]:




