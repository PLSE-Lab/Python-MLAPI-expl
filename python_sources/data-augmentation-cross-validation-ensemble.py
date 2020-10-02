#!/usr/bin/env python
# coding: utf-8

# # Libraries
# * Import necessary libraries and packages

# In[57]:


import numpy as np
import os
import glob
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import keras
from keras.models import Sequential
from keras.utils import to_categorical
from keras.layers import Dense, Input, Conv2D, Flatten, MaxPooling2D, Activation, Dropout, Average
from keras.models import Model
from keras.models import load_model
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.callbacks import ModelCheckpoint
from keras import callbacks
from keras import backend as K
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from keras.preprocessing.image import ImageDataGenerator
import gc


# # Data Concatenation 
# * Made modular loops to concat all datas in separate train, test and label form

# In[5]:


data_dir=os.path.join('..','input')

arr_train = ['a','b','c','d','e']
iterator_train = len(arr_train)
paths_train_all=[]

for i in range(iterator_train):
    #print (arr_train[i])
    dirx= 'training-'+arr_train[i]
    paths_train_x=glob.glob(os.path.join(data_dir,dirx,'*.png'))
    paths_train_all=paths_train_all+paths_train_x

arr_test = ['a','b','c','d','e','f','auga','augc']
iterator_test = len(arr_test)
paths_test_all=[]

for i in range(iterator_test):
    dirx= 'testing-'+arr_test[i]
    paths_test_x=glob.glob(os.path.join(data_dir,dirx,'*.png'))
    paths_test_all=paths_test_all+paths_test_x
    if arr_test[i]=='f':
        paths_test_f=glob.glob(os.path.join(data_dir,dirx,'*.JPG'))
        paths_test_all=paths_test_all+paths_test_f


path_label_train_all=[]
for i in range(iterator_train):
    dirx= 'training-'+arr_train[i] + '.csv'
    paths_label_train = glob.glob(os.path.join(data_dir,dirx))
    
    path_label_train_all= path_label_train_all + paths_label_train
print (path_label_train_all)


# # Data Preparation
# * Necessary function for data preparation and loading
# 

# In[6]:


def get_key(path):
    # seperates the key of an image from the filepath
    key=path.split(sep=os.sep)[-1]
    return key

def get_data(paths_img,path_label=None,resize_dim=None):
    '''reads images from the filepaths, resizes them (if given), and returns them in a numpy array
    Args:
        paths_img: image filepaths
        path_label: pass image label filepaths while processing training data, defaults to None while processing testing data
        resize_dim: if given, the image is resized to resize_dim x resize_dim (optional)
    Returns:
        X: group of images
        y: categorical true labels
    '''
    X=[] # initialize empty list for resized images
    for i,path in enumerate(paths_img):
        img=cv2.imread(path,cv2.IMREAD_COLOR) # images loaded in color (BGR)
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        if resize_dim is not None:
            img=cv2.resize(img,(resize_dim,resize_dim),interpolation=cv2.INTER_AREA) # resize image to 28x28
#         X.append(np.expand_dims(img,axis=2)) # expand image to 28x28x1 and append to the list.
        X.append(img) # expand image to 28x28x1 and append to the list
        # display progress
        if i==len(paths_img)-1:
            end='\n'
        else: end='\r'
        print('processed {}/{}'.format(i+1,len(paths_img)),end=end)
        
    X=np.array(X) # tranform list to numpy array
    if  path_label is None:
        return X
    else:
        

        # Concatenate all data into one DataFrame
        df = pd.DataFrame()
        l = []
        for file_ in path_label:
            df_x = pd.read_csv(file_,index_col=None, header=0)
            l.append(df_x)
        df = pd.concat(l)
        
        #df = pd.read_csv(path_label[i]) # read labels
        df=df.set_index('filename') 
        y_label=[df.loc[get_key(path)]['digit'] for path in  paths_img] # get the labels corresponding to the images
        y=to_categorical(y_label,10) # transfrom integer value to categorical variable

        return X, y

def create_submission(predictions,keys,path):
    result = pd.DataFrame(
        predictions,
        columns=['label'],
        index=keys
        )
    result.index.name='key'
    result.to_csv(path, index=True)


# # Obtain the processed training data and label

# In[7]:


img_size = 32
X_train_all,y_train_all=get_data(paths_train_all,path_label_train_all,resize_dim=img_size)
print (X_train_all.shape)
print (y_train_all.shape)


# # Data Augmentation
# * CAUTION ! Run it locally for avoiding free memory error
# * Experimental setup for augmentation
# * Normalization is done as part of augmentation
# * Also has shifts, rotation, zoom-out, zoom-in and flips
# * Train for prolonged  hours and long epochs to get best results

# In[8]:


def data_aug(X_train,X_test,y_train,y_test,train_batch_size,test_batch_size):
    train_datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1.0/255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')
    test_datagen = ImageDataGenerator(rescale=1.0/255)
    train_batch = train_datagen.flow(X_train,y_train,batch_size=train_batch_size)
    test_batch = test_datagen.flow(X_test,y_test,batch_size=test_batch_size)
    return (train_batch,test_batch)


# # Model
# * Model was designed from scratch

# In[9]:


def create_model(img_size=32,channels=3):
    model = Sequential()
    input_shape = (img_size,img_size,channels)
    model.add(Conv2D(32, (3, 3), input_shape=input_shape, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3),padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3),padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(10))
    model.add(Activation('softmax'))
    
    model.compile(Adam(lr=.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    # UNCOMMENT THIS TO VIEW THE ARCHITECTURE
    #model.summary()
    
    return model


# # Callbacks
# * Contains 4 callbacks
# * Ensure necessary variables are placed in the function call to utilize the function

# In[10]:


def callback(tf_log_dir_name='./tf-log/',patience_lr=10):
    cb = []
    """
    Tensorboard log callback
    """
    tb = callbacks.TensorBoard(log_dir=tf_log_dir_name, histogram_freq=0)
    cb.append(tb)
    
    
    """
    Model-Checkpoint
    """
    #m = callbacks.ModelCheckpoint(filepath=model_name,monitor='val_loss',mode='auto')
    #cb.append(m)
    
    """
    Reduce Learning Rate
    """
    #reduce_lr_loss = callbacks.ReduceLROnPlateau(monitor='loss', factor=0.1, patience=patience_lr, verbose=1, epsilon=1e-4, mode='min')
    #cb.append(reduce_lr_loss)
    
    """
    Early Stopping callback
    """
    #Uncomment for usage
    # early_stop = callbacks.EarlyStopping(monitor='val_acc', min_delta=0, patience=5, verbose=1, mode='auto',save_best_only=True)
    # cb.apppend(early_stop)
    
    
    
    return cb


# # Obtain the processed test data

# In[11]:


X_test_all=get_data(paths_test_all,resize_dim=img_size)


# # Cross validation
# * Currently 5 folds are used. Increase folds to divide training data set into train and val splits
# * Also uncomment/comment for training with Data Augmentation and Without.
# * CAUTION ! Memory exhaustion might happen while training in kernel with Augmentation. Better to  run it locally. 
# *  Normalized the data points to 0-1 floats from 0-255 integer values.
# * Change hyperparameters like Learnin_rate using callback -> ReduceLROnPlateau ( Advanced Users)
# *  We evaluated  the score on both train and val splits from the training data.
# * Then we saved the score for all test data as csv files for each fold ( total 5 folds)
# *  It will generate 5 csv files and 5 .h5 files.
# * Lastly standard deviation and mean accuracy was shown for all folds as output 
# 
# 

# In[13]:


kfold = KFold(n_splits=5, shuffle=True, random_state=42)
cvscores = []
Fold = 1
for train, val in kfold.split(X_train_all, y_train_all):
    gc.collect()
    K.clear_session()
    print ('Fold: ',Fold)
    
    X_train = X_train_all[train]
    X_val = X_train_all[val]
    X_train = X_train.astype('float32')
    X_val = X_val.astype('float32')
    y_train = y_train_all[train]
    y_val = y_train_all[val]
    
    # Data Augmentation and Normalization(OPTIONAL) UNCOMMENT THIS FOR AUGMENTATION !!
    #batch_size = 16
    #train_batch, val_batch = data_aug(X_train,X_val,y_train,y_val, batch_size, batch_size)
    
    # Data Normalization only - COMMENT THIS OUT FOR DATA AUGMENTATION
    X_train /= 255
    X_val /= 255
    
    
    # If model checkpoint is used UNCOMMENT THIS
    #model_name = 'cnn_keras_Fold_'+str(Fold)+'.h5'
    
    cb = callback()
    
    # create model
    model = create_model(img_size,3)
    
    # Fit the model for without Data Augmentation - COMMENT THIS OUT FOR DATA AUGMENTATION
    batch_size=16
    epochs = 5
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size, callbacks=cb, verbose=2)
  
    
    # Fit generator for Data Augmentation - UNCOMMENT THIS FOR DATA AUGMENTATION
    #batch_size = 16
    #epochs = 5 
    #model.fit_generator(train_batch, validation_data=val_batch, epochs=epochs, validation_steps= X_val.shape[0] // batch_size, 
    #                    steps_per_epoch= X_train.shape[0] // batch_size, callbacks=cb, verbose=2)
    
    # Save each fold model
    model_name = 'cnn_keras_aug_Fold_'+str(Fold)+'.h5'
    model.save(model_name)
    
    
    
    # evaluate the model
    scores = model.evaluate(X_val, y_val, verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    cvscores.append(scores[1] * 100)
    
    # save the probability prediction of each fold in separate csv file
    proba = model.predict(X_test_all,batch_size=None,steps=1)
    labels=[np.argmax(pred) for pred in proba]
    keys=[get_key(path) for path in paths_test_all ]
    csv_name= 'submission_CNN_keras_aug_Fold'+str(Fold)+'.csv'
    create_submission(predictions=labels,keys=keys,path=csv_name)
    
    
    Fold = Fold +1

print("%s: %.2f%%" % ("Mean Accuracy: ",np.mean(cvscores)))
print("%s: %.2f%%" % ("Standard Deviation: +/-", np.std(cvscores)))


# # Ensemble 
# Define the ensemble fucntion first
# * Here, we are iterating over all the models to get the last layers as output
# * Then we are adding an merge layer (average)  to compute the average output scores of all the models. 
# * Compile the model
# 

# In[89]:


def ensemble(models, model_input):
    
    Models_output=[model(model_input) for model in models]
    Avg = keras.layers.average(Models_output)
    
    modelEnsemble = Model(inputs=model_input, outputs=Avg, name='ensemble')
    modelEnsemble.summary()
    modelEnsemble.compile(Adam(lr=.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    return modelEnsemble


# ## Ensembling the models
# * First create the architectures using create_model function. We can also load different models. Here, we are using 1 architecture, which were trained on  3 different folds of training data with the cross-validation scheme.
# * If you want use another model. Use pretrained models (Resnet_50, VGG_19, DenseNet.  Link : https://keras.io/applications/#densenet) or write an architecture from scratch and call the appropiate function for compiling it. ** Follow our  create_model function**.
# * Load all the weights.
# * Call the ensemble function,

# In[90]:


model_1 = create_model(img_size,3) 
model_4 = create_model(img_size,3) 
model_5 = create_model(img_size,3) 

models = []

# Load weights 
model_1.load_weights('cnn_keras_aug_Fold_1.h5')
model_1.name = 'model_1'
models.append(model_1)

model_4.load_weights('cnn_keras_aug_Fold_4.h5')
model_4.name = 'model_4'
models.append(model_4)

model_5.load_weights('cnn_keras_aug_Fold_5.h5')
model_5.name = 'model_5'
models.append(model_5)

model_input = Input(shape=models[0].input_shape[1:])
ensemble_model = ensemble(models, model_input)


# ## Evaluate Ensemble model
# * First split the training data to get the validation split.
# * Evaluate to get the score (Accuracy).

# In[92]:


X_train, X_val, y_train, y_val = train_test_split(X_train_all, y_train_all, test_size=0.2, random_state=42)
scores = ensemble_model.evaluate(X_val, y_val, verbose=0)
print("%s: %.2f%%" % (ensemble_model.metrics_names[1], scores[1]*100))


# ## Save the Ensemble Model

# In[93]:


model_name = 'cnn_keras_ensebmle.h5'
ensemble_model.save(model_name)


# # Generate Test data Prediction
# * Save the generated test data predictions in a .csv file.
# * Submit the file to the competitions to see where you stand in the leaderboard.

# In[95]:


proba = ensemble_model.predict(X_test_all,batch_size=None,steps=1)
labels=[np.argmax(pred) for pred in proba]
keys=[get_key(path) for path in paths_test_all ]
csv_name= 'submission_CNN_keras_ensemble.csv'
create_submission(predictions=labels,keys=keys,path=csv_name)


# In[ ]:




