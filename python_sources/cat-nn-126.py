# -*- coding: utf-8 -*-
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
# Any results you write to the current directory are saved as output.
import os, cv2, random
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
from matplotlib import ticker
from keras.models import Sequential
from keras.layers import Input, Dropout, Flatten, Convolution2D, MaxPooling2D, Dense, Activation, ZeroPadding2D, Merge
from keras.optimizers import RMSprop, SGD, Adadelta, Adam
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from keras.utils import np_utils
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.layers.normalization import BatchNormalization
import keras
print(keras.__version__)
import theano
print(theano.__version__)
from keras import backend as K
np.random.seed(55)

LR = 0.0001
  
TRAIN_DIR = '../input/train/'
TEST_DIR = '../input/test/'

#TRAIN_DIR = 'data/train/'
#TEST_DIR = 'data/test/'


TRIM_DATA_COUNT = 4000
#VAL_PART = 400
PRED_TEST = 0

IMG_ROWS = 128 #374 // 2 #128 # fx ?
IMG_COLS = 128 #172 #500 // 2 #128 # fy
IMG_CHANNELS = 3

NB_CLASSES = 1

nb_epoch = 50
batch_size = 64

objective = 'binary_crossentropy' #'binary_crossentropy'

def center_normalize(x):
    """
    Custom activation for online sample-wise center and std. normalization
    """
    return (x - K.mean(x)) / K.std(x)
    
def xscale(x):
    """
    0-1 scale
    """
    x -= K.min(x)
    return x / K.max(x)  

def center_random(x):
    """
    Custom activation for online sample-shape centered random
    """
    return K.random_uniform(K.shape(x)) - .5

#import numpy as np
#newim = np.zeros((height, width, channels))
#for x in xrange(channels):
#    newim[:,:,x] = im[x,:,:]

## Callback for loss logging per epoch
class LossHistory(Callback):
    
    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_losses = []
        self.aucs = []
        
    def on_train_end(self, logs={}):
        return
    
    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

        
    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        y_pred = self.model.predict(self.model.validation_data[0])
        self.aucs.append(roc_auc_score(self.model.validation_data[1], y_pred))

# ===================================================

try:
    print (X.shape)
except:

    #train_images = [TRAIN_DIR + i for i in os.listdir(TRAIN_DIR)] # use this for full dataset
    train_dogs =   [TRAIN_DIR + i for i in os.listdir(TRAIN_DIR) if 'dog' in i]
    train_cats =   [TRAIN_DIR + i for i in os.listdir(TRAIN_DIR) if 'cat' in i]
    
    #val_cats = train_cats[-VAL_PART // 2:]
    #val_dogs = train_dogs[-VAL_PART // 2:]
    #train_cats = train_cats[:-VAL_PART // 2]
    #train_dogs = train_dogs[:-VAL_PART // 2]
    
    if PRED_TEST:
        test_images =  [TEST_DIR+i for i in os.listdir(TEST_DIR)]
    
    
    if TRIM_DATA_COUNT:
        train_images = train_dogs[:TRIM_DATA_COUNT // 2] + train_cats[:TRIM_DATA_COUNT // 2]        
        if PRED_TEST:
            test_images =  test_images[:TRIM_DATA_COUNT]
    else:
        train_images = train_dogs + train_cats
        
    
    random.shuffle(train_images)
    
    
    def read_image(file_path):
        if IMG_CHANNELS == 1:
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE) #cv2.IMREAD_GRAYSCALE
        else:
            img = cv2.imread(file_path, cv2.IMREAD_COLOR) #cv2.IMREAD_GRAYSCALE
            
        h = min(img.shape[0], img.shape[1])
        h_max = max(img.shape[0], img.shape[1])
        
        if IMG_CHANNELS == 1:
            img2 = np.zeros((h_max, h_max), dtype = np.uint8)
        else:
            img2 = np.zeros((h_max, h_max, IMG_CHANNELS), dtype = np.uint8)
    
        h = img.shape[0]# / 2
        w = img.shape[1]# / 2
        h2 = img.shape[0] / 2
        w2 = img.shape[1] / 2
    
        xc = img2.shape[1] / 2
        yc = img2.shape[0] / 2
    
        img2[yc - h2 : yc - h2 + h, xc - w2 : xc - w2 + w] = img
        img = img2
    
        return cv2.resize(img, (IMG_ROWS, IMG_COLS), interpolation = cv2.INTER_CUBIC) # (fx, fy)
    
    
    def prep_data(images):
        count = len(images)
        data = np.ndarray((count, IMG_CHANNELS, IMG_ROWS, IMG_COLS), dtype=np.uint8)
        #data = np.ndarray((count, IMG_ROWS, IMG_COLS, IMG_CHANNELS), dtype=np.uint8)
    
        for i, image_file in enumerate(images):
            image = read_image(image_file)
            data[i] = image.T
            #print(image.shape)
            #data[i] = image#.transpose(1, 2, 0)
            if i % 200 == 0: 
                print('Processed {} of {}'.format(i, count))
        
        return data
    
    X = prep_data(train_images).astype('float32') #/ 255.
    
    mean = np.mean(X)  # mean for data centering
    std = np.std(X)  # std for data normalization

    X -= mean
    X /= std
    

    if PRED_TEST:
        X_test = prep_data(test_images).astype('float32') #/ 255.
        X_test -= mean
        X_test /= std
    
    labels = [int('dog' in img) for img in train_images]
    
    if NB_CLASSES > 1:
        y = np_utils.to_categorical(labels, NB_CLASSES)
    else:
        y = np.array(labels, dtype = np.int32)
        
    print('X:', X.shape, X.min(), X.max(), np.median(X))


print("Train shape: {}".format(X.shape))
if PRED_TEST:
    print("Test shape: {}".format(X_test.shape))






def make_model_1():
    
    model = Sequential()

    model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=(IMG_CHANNELS, IMG_ROWS, IMG_COLS), activation='relu'))
    model.add(Convolution2D(32, 3, 3, border_mode='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu'))
    model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Convolution2D(128, 3, 3, border_mode='same', activation='relu'))
    model.add(Convolution2D(128, 3, 3, border_mode='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Convolution2D(256, 3, 3, border_mode='same', activation='relu'))
    model.add(Convolution2D(256, 3, 3, border_mode='same', activation='relu'))
#     model.add(Convolution2D(256, 3, 3, border_mode='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

#     model.add(Convolution2D(256, 3, 3, border_mode='same', activation='relu'))
#     model.add(Convolution2D(256, 3, 3, border_mode='same', activation='relu'))
#     model.add(Convolution2D(256, 3, 3, border_mode='same', activation='relu'))
#     model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(NB_CLASSES))
    model.add(Activation('sigmoid'))

    model.compile(loss = objective, optimizer=optimizer, metrics=['accuracy'])
    
    return model

def make_model_111():
    
    border_mode = 'same' #'valid' # 'same'
    s = 3
    
    model = Sequential()

    model.add(ZeroPadding2D(padding=(1, 1), input_shape=(IMG_CHANNELS, IMG_ROWS, IMG_COLS)))
    model.add(Convolution2D(32, s, s, border_mode= border_mode, activation='relu'))
    model.add(ZeroPadding2D(padding=(1, 1)))
    model.add(Convolution2D(32, s, s, border_mode= border_mode, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(ZeroPadding2D(padding=(1, 1)))
    model.add(Convolution2D(64, s, s, border_mode= border_mode, activation='relu'))
    model.add(ZeroPadding2D(padding=(1, 1)))
    model.add(Convolution2D(64, s, s, border_mode= border_mode, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(ZeroPadding2D(padding=(1, 1)))
    model.add(Convolution2D(128, s, s, border_mode= border_mode, activation='relu'))
    model.add(ZeroPadding2D(padding=(1, 1)))
    model.add(Convolution2D(128, s, s, border_mode= border_mode, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(ZeroPadding2D(padding=(1, 1)))
    model.add(Convolution2D(256, s, s, border_mode= border_mode, activation='relu'))
    model.add(ZeroPadding2D(padding=(1, 1)))
    model.add(Convolution2D(256, s, s, border_mode= border_mode, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

#     model.add(Convolution2D(256, 3, 3, border_mode= border_mode, activation='relu'))
#     model.add(Convolution2D(256, 3, 3, border_mode= border_mode, activation='relu'))
#     model.add(Convolution2D(256, 3, 3, border_mode= border_mode, activation='relu'))
#     model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(NB_CLASSES))
    model.add(Activation('sigmoid'))
    
    #optimizer = RMSprop(lr=1e-3)
    
    optimizer = SGD(lr= LR, decay = 1e-6, momentum = 0.95, nesterov = True)    
    ##opt = Adadelta(lr= .1, decay=0.995, epsilon=1e-5)  
    optimizer = Adam(lr = LR)
    
    model.compile(loss = objective, optimizer = optimizer, metrics=['accuracy'])
    
    return model


############################################################################################


def make_model_1112():
    
    border_mode = 'same' #'valid' # 'same'
    s = 3
    
    model = Sequential()

    model.add(ZeroPadding2D(padding=(1, 1), input_shape=(IMG_CHANNELS, IMG_ROWS, IMG_COLS)))
    model.add(Convolution2D(32, s, s, border_mode= border_mode, activation='relu'))
    model.add(ZeroPadding2D(padding=(1, 1)))
    model.add(Convolution2D(32, s, s, border_mode= border_mode, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(ZeroPadding2D(padding=(1, 1)))
    model.add(Convolution2D(64, s, s, border_mode= border_mode, activation='relu'))
    model.add(ZeroPadding2D(padding=(1, 1)))
    model.add(Convolution2D(64, s, s, border_mode= border_mode, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(ZeroPadding2D(padding=(1, 1)))
    model.add(Convolution2D(128, s, s, border_mode= border_mode, activation='relu'))
    model.add(ZeroPadding2D(padding=(1, 1)))
    model.add(Convolution2D(128, s, s, border_mode= border_mode, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    #model.add(ZeroPadding2D(padding=(1, 1)))
    #model.add(Convolution2D(256, s, s, border_mode= border_mode, activation='relu'))
    #model.add(ZeroPadding2D(padding=(1, 1)))
    #model.add(Convolution2D(256, s, s, border_mode= border_mode, activation='relu'))
    #model.add(MaxPooling2D(pool_size=(2, 2)))

#     model.add(Convolution2D(256, 3, 3, border_mode= border_mode, activation='relu'))
#     model.add(Convolution2D(256, 3, 3, border_mode= border_mode, activation='relu'))
#     model.add(Convolution2D(256, 3, 3, border_mode= border_mode, activation='relu'))
#     model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(NB_CLASSES))
    model.add(Activation('sigmoid'))
    
    #optimizer = RMSprop(lr=1e-3)
    
    optimizer = SGD(lr= LR, decay = 1e-6, momentum = 0.95, nesterov = True)    
    ##opt = Adadelta(lr= .1, decay=0.995, epsilon=1e-5)  
    optimizer = Adam(lr = LR)
    
    model.compile(loss = objective, optimizer = optimizer, metrics=['accuracy'])
    
    return model    
    

def make_model_1113():
    
    border_mode = 'same' #'valid' # 'same'
    s = 4
    st = 2 #  subsample=(2, 2)
    
    model = Sequential()

    model.add(ZeroPadding2D(padding=(2, 2), input_shape=(IMG_CHANNELS, IMG_ROWS, IMG_COLS)))
    model.add(Convolution2D(32, s, s, subsample=(st, st), border_mode= border_mode, activation='relu'))
    model.add(ZeroPadding2D(padding=(1, 1)))
    model.add(Convolution2D(32, s, s, subsample=(st, st), border_mode= border_mode, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(ZeroPadding2D(padding=(2, 2)))
    model.add(Convolution2D(64, s, s, subsample=(st, st), border_mode= border_mode, activation='relu'))
    model.add(ZeroPadding2D(padding=(1, 1)))
    model.add(Convolution2D(64, s, s, subsample=(st, st), border_mode= border_mode, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(ZeroPadding2D(padding=(2, 2)))
    model.add(Convolution2D(128, s, s, subsample=(st, st), border_mode= border_mode, activation='relu'))
    model.add(ZeroPadding2D(padding=(1, 1)))
    model.add(Convolution2D(128, s, s, subsample=(st, st), border_mode= border_mode, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    #model.add(ZeroPadding2D(padding=(1, 1)))
    #model.add(Convolution2D(256, s, s, border_mode= border_mode, activation='relu'))
    #model.add(ZeroPadding2D(padding=(1, 1)))
    #model.add(Convolution2D(256, s, s, border_mode= border_mode, activation='relu'))
    #model.add(MaxPooling2D(pool_size=(2, 2)))

#     model.add(Convolution2D(256, 3, 3, border_mode= border_mode, activation='relu'))
#     model.add(Convolution2D(256, 3, 3, border_mode= border_mode, activation='relu'))
#     model.add(Convolution2D(256, 3, 3, border_mode= border_mode, activation='relu'))
#     model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(NB_CLASSES))
    model.add(Activation('sigmoid'))
    
    #optimizer = RMSprop(lr=1e-3)
    
    optimizer = SGD(lr= LR, decay = 1e-6, momentum = 0.95, nesterov = True)    
    ##opt = Adadelta(lr= .1, decay=0.995, epsilon=1e-5)  
    optimizer = Adam(lr = LR)
    
    model.compile(loss = objective, optimizer = optimizer, metrics=['accuracy'])
    
    return model   
    

##################################################################################################################
##################################################################################################################
##################################################################################################################

def make_model_01():
    
    # Best epoch:  
    
    border_mode = 'same' #'valid' # 'same'
    s = 4
    s1 = 3
    s2 = 3
    
    model = Sequential()

    model.add(ZeroPadding2D(padding=(2, 2), input_shape=(IMG_CHANNELS, IMG_ROWS, IMG_COLS)))
    
    model.add(Convolution2D(32, s, s, border_mode= border_mode, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(ZeroPadding2D(padding=(2, 2)))
    model.add(Convolution2D(64, s1, s1, border_mode= border_mode, activation='relu'))
    #model.add(ZeroPadding2D(padding=(1, 1)))
    #model.add(Convolution2D(32, s, s, border_mode= border_mode, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(ZeroPadding2D(padding=(2, 2)))
    model.add(Convolution2D(32, s2, s2, border_mode= border_mode, activation='relu'))
    #model.add(ZeroPadding2D(padding=(1, 1)))
    #model.add(Convolution2D(64, s, s, border_mode= border_mode, activation='relu'))
    #model.add(Convolution2D(64, s, s, border_mode= border_mode, activation='relu'))
    #model.add(ZeroPadding2D(padding=(1, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    #model.add(Flatten(input_shape=(IMG_CHANNELS, IMG_ROWS, IMG_COLS)))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(NB_CLASSES))
    model.add(Activation('sigmoid'))
    
    #optimizer = RMSprop(lr=1e-3)
    
    optimizer = SGD(lr= LR, decay = 1e-6, momentum = 0.95, nesterov = True)    
    ##opt = Adadelta(lr= .1, decay=0.995, epsilon=1e-5)  
    #optimizer = Adam(lr = LR)
    
    model.compile(loss = objective, optimizer = optimizer, metrics=['accuracy'])
    
    return model
    
def make_model_01__():
    
    # Best epoch:  29 0.622336828709 .001
    
    border_mode = 'same' #'valid' # 'same'
    s = 5
    s1 = 3
    
    model = Sequential()

    model.add(ZeroPadding2D(padding=(3, 3), input_shape=(IMG_CHANNELS, IMG_ROWS, IMG_COLS)))
    
    model.add(Convolution2D(64, s, s, border_mode= border_mode, activation='relu'))
    #model.add(ZeroPadding2D(padding=(1, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(128, s1, s1, border_mode= border_mode, activation='relu'))
    #model.add(ZeroPadding2D(padding=(1, 1)))
    #model.add(Convolution2D(32, s, s, border_mode= border_mode, activation='relu'))
    #model.add(MaxPooling2D(pool_size=(2, 2)))
    
    #model.add(ZeroPadding2D(padding=(1, 1)))
    #model.add(Convolution2D(64, s, s, border_mode= border_mode, activation='relu'))
    #model.add(ZeroPadding2D(padding=(1, 1)))
    #model.add(Convolution2D(64, s, s, border_mode= border_mode, activation='relu'))
    #model.add(Convolution2D(64, s, s, border_mode= border_mode, activation='relu'))
    #model.add(ZeroPadding2D(padding=(1, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    #model.add(Flatten(input_shape=(IMG_CHANNELS, IMG_ROWS, IMG_COLS)))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(NB_CLASSES))
    model.add(Activation('sigmoid'))
    
    #optimizer = RMSprop(lr=1e-3)
    
    optimizer = SGD(lr= LR, decay = 1e-6, momentum = 0.95, nesterov = True)    
    ##opt = Adadelta(lr= .1, decay=0.995, epsilon=1e-5)  
    #optimizer = Adam(lr = LR)
    
    model.compile(loss = objective, optimizer = optimizer, metrics=['accuracy'])
    
    return model    
    
def make_model_01_():
    
    # Best epoch:  26 0.6257030797
    
    border_mode = 'same' #'valid' # 'same'
    s = 3
    s1 = 5
    
    model = Sequential()

    model.add(ZeroPadding2D(padding=(1, 1), input_shape=(IMG_CHANNELS, IMG_ROWS, IMG_COLS)))
    
    model.add(Convolution2D(64, s1, s1, border_mode= border_mode, activation='relu'))
    #model.add(ZeroPadding2D(padding=(1, 1)))
    model.add(Convolution2D(128, s1, s1, border_mode= border_mode, activation='relu'))
    #model.add(ZeroPadding2D(padding=(1, 1)))
    #model.add(Convolution2D(32, s, s, border_mode= border_mode, activation='relu'))
    #model.add(MaxPooling2D(pool_size=(2, 2)))
    
    #model.add(ZeroPadding2D(padding=(1, 1)))
    #model.add(Convolution2D(64, s, s, border_mode= border_mode, activation='relu'))
    #model.add(ZeroPadding2D(padding=(1, 1)))
    #model.add(Convolution2D(64, s, s, border_mode= border_mode, activation='relu'))
    #model.add(Convolution2D(64, s, s, border_mode= border_mode, activation='relu'))
    #model.add(ZeroPadding2D(padding=(1, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    #model.add(Flatten(input_shape=(IMG_CHANNELS, IMG_ROWS, IMG_COLS)))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(NB_CLASSES))
    model.add(Activation('sigmoid'))
    
    #optimizer = RMSprop(lr=1e-3)
    
    optimizer = SGD(lr= LR, decay = 1e-6, momentum = 0.95, nesterov = True)    
    ##opt = Adadelta(lr= .1, decay=0.995, epsilon=1e-5)  
    #optimizer = Adam(lr = LR)
    
    model.compile(loss = objective, optimizer = optimizer, metrics=['accuracy'])
    
    return model
    
    
def make_model_1115():
    
    border_mode = 'same' #'valid' # 'same'
    s = 3
    st = 1 #  subsample=(2, 2)
    
    model = Sequential()

    model.add(ZeroPadding2D(padding=(1, 1), input_shape=(IMG_CHANNELS, IMG_ROWS, IMG_COLS)))
    model.add(Convolution2D(64, s, s, subsample=(st, st), border_mode= border_mode, activation='relu'))
    #model.add(ZeroPadding2D(padding=(1, 1)))
    model.add(Convolution2D(96, s, s, subsample=(st, st), border_mode= border_mode, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

#    model.add(ZeroPadding2D(padding=(1, 1)))
#    model.add(Convolution2D(128, s, s, subsample=(st, st), border_mode= border_mode, activation='relu'))
#    model.add(ZeroPadding2D(padding=(1, 1)))
#    model.add(Convolution2D(128, s, s, subsample=(st, st), border_mode= border_mode, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
#    model.add(ZeroPadding2D(padding=(1, 1)))
#    model.add(Convolution2D(48, s, s, subsample=(st, st), border_mode= border_mode, activation='relu'))
#    model.add(ZeroPadding2D(padding=(1, 1)))
#    model.add(Convolution2D(64, s, s, subsample=(st, st), border_mode= border_mode, activation='relu'))
#    model.add(MaxPooling2D(pool_size=(2, 2)))
    
#    model.add(ZeroPadding2D(padding=(1, 1)))
#    model.add(Convolution2D(64, s, s, border_mode= border_mode, activation='relu'))
#    model.add(ZeroPadding2D(padding=(1, 1)))
#    model.add(Convolution2D(96, s, s, border_mode= border_mode, activation='relu'))
#    model.add(MaxPooling2D(pool_size=(2, 2)))

#     model.add(Convolution2D(256, 3, 3, border_mode= border_mode, activation='relu'))
#     model.add(Convolution2D(256, 3, 3, border_mode= border_mode, activation='relu'))
#     model.add(Convolution2D(256, 3, 3, border_mode= border_mode, activation='relu'))
#     model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(NB_CLASSES))
    model.add(Activation('sigmoid'))
    
    #optimizer = RMSprop(lr=1e-3)
    
    optimizer = SGD(lr= LR, decay = 1e-6, momentum = 0.95, nesterov = True)    
    #optimizer = Adadelta(lr= LR, decay=0.995, epsilon=1e-5)  
    #optimizer = Adam(lr = LR)
    
    model.compile(loss = objective, optimizer = optimizer, metrics=['accuracy'])
    
    return model   


def make_model_0(img_rows = IMG_ROWS, img_cols = IMG_COLS, color_type = IMG_CHANNELS):
    
    model = Sequential()
    
    model.add(ZeroPadding2D(padding=(1, 1), input_shape=(IMG_CHANNELS, IMG_ROWS, IMG_COLS)))
    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(ZeroPadding2D(padding=(1, 1)))
    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(ZeroPadding2D(padding=(1, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(ZeroPadding2D(padding=(1, 1)))
    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(ZeroPadding2D(padding=(1, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    
    return model
 
    
  

def make_model_20(img_rows = IMG_ROWS, img_cols = IMG_COLS, color_type = IMG_CHANNELS):
    
    model = Sequential()

    #model.add(Activation(activation=center_normalize, input_shape= (IMG_CHANNELS, IMG_ROWS, IMG_COLS)))
    #model.add(Activation(activation = '', input_shape= (IMG_CHANNELS, IMG_ROWS, IMG_COLS)))
    model.add(Activation(activation= center_normalize, input_shape= (IMG_CHANNELS, IMG_ROWS, IMG_COLS)))
    
    #model.add(ZeroPadding2D(padding=(1, 1), input_shape=(IMG_CHANNELS, IMG_ROWS, IMG_COLS)))

    model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu'))
    #model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    #model.add(ZeroPadding2D(padding=(1, 1)))
    model.add(Convolution2D(64, 3, 3, border_mode='valid', activation='relu'))
    model.add(ZeroPadding2D(padding=(1, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
#    model.add(Dropout(0.2))

    model.add(ZeroPadding2D(padding=(2, 2)))
    model.add(Convolution2D(64, 4, 4, border_mode='valid', activation='relu'))
    
#    model.add(ZeroPadding2D(padding=(1, 1)))
#    model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu'))
    #model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    #model.add(ZeroPadding2D(padding=(1, 1)))
#    model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu'))
#    model.add(ZeroPadding2D(padding=(1, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.2))

#    model.add(ZeroPadding2D(padding=(1, 1)))
#    model.add(Convolution2D(128, 2, 2, border_mode='same', activation='relu'))
#    model.add(Convolution2D(128, 2, 2, border_mode='same', activation='relu'))
#    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
#    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))#, W_regularizer=l2(1e-3)))
    model.add(Dropout(0.5))
#    model.add(Dense(512, activation='relu'))#, W_regularizer=l2(1e-3)))
#    model.add(Dropout(0.5))
    model.add(Dense(NB_CLASSES, activation='sigmoid'))

    optimizer = Adam(lr = LR)
    #model.compile(optimizer=adam, loss=root_mean_squared_error)
    model.compile(loss = objective, optimizer = optimizer, metrics=['accuracy'])
    
    return model

def make_model_m_1(img_rows = IMG_ROWS, img_cols = IMG_COLS, color_type = IMG_CHANNELS):    
    
    model1 = Sequential()
    model1.add(Activation(activation= center_normalize, input_shape= (IMG_CHANNELS, IMG_ROWS, IMG_COLS)))
    #model1.add(ZeroPadding2D(padding=(1, 1), input_shape=(IMG_CHANNELS, IMG_ROWS, IMG_COLS)))
    model1.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu'))
    #model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    #model.add(ZeroPadding2D(padding=(1, 1)))
    model1.add(Convolution2D(64, 3, 3, border_mode='valid', activation='relu'))
    model1.add(ZeroPadding2D(padding=(1, 1)))
    model1.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model1.add(Dropout(0.3))
    model1.add(Flatten())
    
    model2 = Sequential()
    model2.add(Activation(activation= center_normalize, input_shape= (IMG_CHANNELS, IMG_ROWS, IMG_COLS)))
    model2.add(ZeroPadding2D(padding=(2, 2), input_shape=(IMG_CHANNELS, IMG_ROWS, IMG_COLS)))
    model2.add(Convolution2D(64, 4, 4, border_mode='same', activation='relu'))
    model2.add(Convolution2D(64, 4, 4, border_mode='valid', activation='relu'))
    model2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model2.add(Dropout(0.3))
    model2.add(Flatten())
    
def make_model_m_13(img_rows = IMG_ROWS, img_cols = IMG_COLS, color_type = IMG_CHANNELS):   
    
    model1 = Sequential()
    model1.add(Activation(activation= center_normalize, input_shape= (IMG_CHANNELS, IMG_ROWS, IMG_COLS)))
    #model1.add(ZeroPadding2D(padding=(2, 2), input_shape=(IMG_CHANNELS, IMG_ROWS, IMG_COLS)))
    model1.add(Convolution2D(32, 3, 3, border_mode='same', activation='relu'))
    #model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    #model.add(ZeroPadding2D(padding=(1, 1)))
    model1.add(Convolution2D(64, 3, 3, border_mode='valid', activation='relu'))
    model1.add(ZeroPadding2D(padding=(1, 1)))
    model1.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model1.add(Dropout(0.3))
    model1.add(Flatten())
    
    model2 = Sequential()
    model2.add(Activation(activation= center_normalize, input_shape= (IMG_CHANNELS, IMG_ROWS, IMG_COLS)))
    model2.add(ZeroPadding2D(padding=(2, 2), input_shape=(IMG_CHANNELS, IMG_ROWS, IMG_COLS)))
    model2.add(Convolution2D(32, 4, 4, border_mode='same', activation='relu'))
    model2.add(Convolution2D(64, 4, 4, border_mode='valid', activation='relu'))
    model2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model2.add(Dropout(0.3))
    model2.add(Flatten())    
    
    model3 = Sequential()
    model3.add(Activation(activation= center_normalize, input_shape= (IMG_CHANNELS, IMG_ROWS, IMG_COLS)))
    #model1.add(ZeroPadding2D(padding=(1, 1), input_shape=(IMG_CHANNELS, IMG_ROWS, IMG_COLS)))
    model3.add(Convolution2D(32, 1, 1, border_mode='same', activation='relu'))
    #model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    #model.add(ZeroPadding2D(padding=(1, 1)))
    #model1.add(Convolution2D(64, 1, 1, border_mode='valid', activation='relu'))
    #model1.add(ZeroPadding2D(padding=(1, 1)))
    #model1.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model3.add(Dropout(0.3))
    model3.add(Flatten())
    
    model4 = Sequential()
    model4.add(Activation(activation= center_normalize, input_shape= (IMG_CHANNELS, IMG_ROWS, IMG_COLS)))
    #model4.add(ZeroPadding2D(padding=(2, 2), input_shape=(IMG_CHANNELS, IMG_ROWS, IMG_COLS)))
    model4.add(Convolution2D(32, 2, 2, border_mode='same', activation='relu'))
    model4.add(Convolution2D(64, 2, 2, border_mode='valid', activation='relu'))
    model4.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model4.add(Dropout(0.3))
    model4.add(Flatten())   
    
    
    model = Sequential()
    model.add(Merge([model1, model2, model3, model4], mode='concat'))
   
    #model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(.5))
    
    model.add(Dense(NB_CLASSES))
    model.add(Activation('sigmoid'))
    
    #optimizer = RMSprop(lr=1e-3)
    
    optimizer = SGD(lr= LR, decay = 1e-6, momentum = 0.95, nesterov = True)    
    ##opt = Adadelta(lr= .1, decay=0.995, epsilon=1e-5)  
    optimizer = Adam(lr = LR)
    
    model.compile(loss = objective, optimizer = optimizer, metrics=['accuracy'])
    
    return model, 4
    
def make_model_m_14(img_rows = IMG_ROWS, img_cols = IMG_COLS, color_type = IMG_CHANNELS):   
    
    model1 = Sequential()
    model1.add(Activation(activation= center_normalize, input_shape= (IMG_CHANNELS, IMG_ROWS, IMG_COLS)))
    #model1.add(ZeroPadding2D(padding=(2, 2), input_shape=(IMG_CHANNELS, IMG_ROWS, IMG_COLS)))
    model1.add(Convolution2D(32, 3, 3, border_mode='same', activation='relu'))
    #model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    #model.add(ZeroPadding2D(padding=(1, 1)))
    model1.add(Convolution2D(64, 3, 3, border_mode='valid', activation='relu'))
    model1.add(ZeroPadding2D(padding=(1, 1)))
    model1.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model1.add(ZeroPadding2D(padding=(1, 1)))
    model1.add(Convolution2D(64, 3, 3, border_mode='valid', activation='relu'))
    model1.add(ZeroPadding2D(padding=(1, 1)))
    model1.add(Convolution2D(128, 3, 3, border_mode='valid', activation='relu'))
    model1.add(ZeroPadding2D(padding=(1, 1)))
    model1.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model1.add(Dropout(0.3))
    model1.add(Flatten())
    model1.add(BatchNormalization())
    
    model2 = Sequential()
    model2.add(Activation(activation= center_normalize, input_shape= (IMG_CHANNELS, IMG_ROWS, IMG_COLS)))
    model2.add(ZeroPadding2D(padding=(2, 2)))
    model2.add(Convolution2D(32, 4, 4, border_mode='same', activation='relu'))
    model2.add(Convolution2D(48, 4, 4, border_mode='valid', activation='relu'))
    model2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model2.add(ZeroPadding2D(padding=(2, 2)))
    model2.add(Convolution2D(64, 4, 4, border_mode='valid', activation='relu'))
    #model2.add(ZeroPadding2D(padding=(1, 1)))
    #model2.add(Convolution2D(96, 4, 4, border_mode='same', activation='relu'))
    model2.add(Dropout(0.3))
    model2.add(Flatten()) 
    model2.add(BatchNormalization())
    
    
    model3 = Sequential()
    model3.add(Activation(activation= center_normalize, input_shape= (IMG_CHANNELS, IMG_ROWS, IMG_COLS)))
    #model1.add(ZeroPadding2D(padding=(1, 1), input_shape=(IMG_CHANNELS, IMG_ROWS, IMG_COLS)))
    model3.add(Convolution2D(64, 1, 1, border_mode='same', activation='relu'))
    #model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    #model.add(ZeroPadding2D(padding=(1, 1)))
    #model1.add(Convolution2D(64, 1, 1, border_mode='valid', activation='relu'))
    #model1.add(ZeroPadding2D(padding=(1, 1)))
    #model1.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model3.add(Dropout(0.3))
    model3.add(Flatten())
    model3.add(BatchNormalization())
    
    model4 = Sequential()
    model4.add(Activation(activation= center_normalize, input_shape= (IMG_CHANNELS, IMG_ROWS, IMG_COLS)))
    #model4.add(ZeroPadding2D(padding=(2, 2), input_shape=(IMG_CHANNELS, IMG_ROWS, IMG_COLS)))
    model4.add(Convolution2D(32, 2, 2, border_mode='same', activation='relu'))
    model4.add(Convolution2D(64, 2, 2, border_mode='valid', activation='relu'))
    model4.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model4.add(ZeroPadding2D(padding=(2, 2)))
    #model4.add(Convolution2D(96, 2, 2, border_mode='valid', activation='relu'))
    #model4.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    
    model4.add(Dropout(0.3))
    model4.add(Flatten())
    model4.add(BatchNormalization())
    
    
    model5 = Sequential()
    model5.add(Activation(activation= center_normalize, input_shape= (IMG_CHANNELS, IMG_ROWS, IMG_COLS)))
    model5.add(ZeroPadding2D(padding=(2, 2), input_shape=(IMG_CHANNELS, IMG_ROWS, IMG_COLS)))
    model5.add(Convolution2D(16, 5, 5, border_mode='same', activation='relu'))
    model5.add(Convolution2D(24, 5, 5, border_mode='valid', activation='relu'))
    model5.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    #model5.add(ZeroPadding2D(padding=(2, 2)))
    #model4.add(Convolution2D(96, 2, 2, border_mode='valid', activation='relu'))
    #model4.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    
    model5.add(Dropout(0.3))
    model5.add(Flatten())
    model5.add(BatchNormalization())
    
    
    model = Sequential()
    model.add(Merge([model1, model2, model3, model4, model5], mode='concat'))
   
    #model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(.5))
    
    model.add(Dense(NB_CLASSES))
    model.add(Activation('sigmoid'))
    
    #optimizer = RMSprop(lr=1e-3)
    
    optimizer = SGD(lr= LR, decay = 1e-6, momentum = 0.95, nesterov = True)    
    ##opt = Adadelta(lr= .1, decay=0.995, epsilon=1e-5)  
    optimizer = Adam(lr = LR)
    
    model.compile(loss = objective, optimizer = optimizer, metrics=['accuracy'])
    
    return model, 5


def make_model_m_14_1(img_rows = IMG_ROWS, img_cols = IMG_COLS, color_type = IMG_CHANNELS):   
    
    model1 = Sequential()
    model1.add(Activation(activation= center_normalize, input_shape= (IMG_CHANNELS, IMG_ROWS, IMG_COLS)))
    #model1.add(ZeroPadding2D(padding=(2, 2), input_shape=(IMG_CHANNELS, IMG_ROWS, IMG_COLS)))
    model1.add(Convolution2D(32, 3, 3, border_mode='same', activation='relu'))
    #model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    #model.add(ZeroPadding2D(padding=(1, 1)))
    model1.add(Convolution2D(64, 3, 3, border_mode='valid', activation='relu'))
    model1.add(ZeroPadding2D(padding=(1, 1)))
    model1.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model1.add(ZeroPadding2D(padding=(1, 1)))
    model1.add(Convolution2D(64, 3, 3, border_mode='valid', activation='relu'))
    model1.add(ZeroPadding2D(padding=(1, 1)))
    model1.add(Convolution2D(128, 3, 3, border_mode='valid', activation='relu'))
    model1.add(ZeroPadding2D(padding=(1, 1)))
    model1.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model1.add(Dropout(0.3))
    model1.add(Flatten())
    model1.add(BatchNormalization())
    
    
    
    model = model1
    
    #model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(.5))
    #model.add(BatchNormalization())
    
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(.5))
    #model.add(BatchNormalization())
    
    model.add(Dense(NB_CLASSES))
    model.add(Activation('sigmoid'))
    
    #optimizer = RMSprop(lr=1e-3)
    
    optimizer = SGD(lr= LR, decay = 1e-6, momentum = 0.95, nesterov = True)    
    ##opt = Adadelta(lr= .1, decay=0.995, epsilon=1e-5)  
    optimizer = Adam(lr = LR)
    
    model.compile(loss = objective, optimizer = optimizer, metrics=['accuracy'])
    
    return model, 1

def make_model_m_14_2(img_rows = IMG_ROWS, img_cols = IMG_COLS, color_type = IMG_CHANNELS):   
    
    model2 = Sequential()
    model2.add(Activation(activation= center_normalize, input_shape= (IMG_CHANNELS, IMG_ROWS, IMG_COLS)))
    model2.add(ZeroPadding2D(padding=(2, 2)))
    model2.add(Convolution2D(32, 4, 4, border_mode='same', activation='relu'))
    model2.add(Convolution2D(48, 4, 4, border_mode='valid', activation='relu'))
    model2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model2.add(ZeroPadding2D(padding=(2, 2)))
    model2.add(Convolution2D(64, 4, 4, border_mode='valid', activation='relu'))
    #model2.add(ZeroPadding2D(padding=(1, 1)))
    #model2.add(Convolution2D(96, 4, 4, border_mode='same', activation='relu'))
    model2.add(Dropout(0.3))
    model2.add(Flatten()) 
    model2.add(BatchNormalization())
    
    
    
    model = model2
    
    #model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(.5))

    model.add(Dense(256, activation='relu'))
    model.add(Dropout(.5))

    model.add(Dense(NB_CLASSES))
    model.add(Activation('sigmoid'))
    
    #optimizer = RMSprop(lr=1e-3)
    
    optimizer = SGD(lr= LR, decay = 1e-6, momentum = 0.95, nesterov = True)    
    ##opt = Adadelta(lr= .1, decay=0.995, epsilon=1e-5)  
    optimizer = Adam(lr = LR)
    
    model.compile(loss = objective, optimizer = optimizer, metrics=['accuracy'])
    
    return model, 1
    
def make_model_m_14_4(img_rows = IMG_ROWS, img_cols = IMG_COLS, color_type = IMG_CHANNELS):   
    
    model4 = Sequential()
    model4.add(Activation(activation= center_normalize, input_shape= (IMG_CHANNELS, IMG_ROWS, IMG_COLS)))
    #model4.add(ZeroPadding2D(padding=(2, 2), input_shape=(IMG_CHANNELS, IMG_ROWS, IMG_COLS)))
    model4.add(Convolution2D(32, 2, 2, border_mode='same', activation='relu'))
    model4.add(Convolution2D(64, 2, 2, border_mode='valid', activation='relu'))
    model4.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model4.add(ZeroPadding2D(padding=(2, 2)))
    #model4.add(Convolution2D(96, 2, 2, border_mode='valid', activation='relu'))
    #model4.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    
    model4.add(Dropout(0.3))
    model4.add(Flatten())
    model4.add(BatchNormalization())
    
    model = model4
    #model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(.5))

    model.add(Dense(NB_CLASSES))
    model.add(Activation('sigmoid'))
    
    #optimizer = RMSprop(lr=1e-3)
    
    optimizer = SGD(lr= LR, decay = 1e-6, momentum = 0.95, nesterov = True)    
    ##opt = Adadelta(lr= .1, decay=0.995, epsilon=1e-5)  
    optimizer = Adam(lr = LR)
    
    model.compile(loss = objective, optimizer = optimizer, metrics=['accuracy'])
    
    return model, 1    
    
def make_model_m_14_3(img_rows = IMG_ROWS, img_cols = IMG_COLS, color_type = IMG_CHANNELS):   
    
    model3 = Sequential()
    model3.add(Activation(activation= center_normalize, input_shape= (IMG_CHANNELS, IMG_ROWS, IMG_COLS)))
    #model1.add(ZeroPadding2D(padding=(1, 1), input_shape=(IMG_CHANNELS, IMG_ROWS, IMG_COLS)))
    model3.add(Convolution2D(64, 1, 1, border_mode='same', activation='relu'))
    #model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    #model.add(ZeroPadding2D(padding=(1, 1)))
    #model1.add(Convolution2D(64, 1, 1, border_mode='valid', activation='relu'))
    #model1.add(ZeroPadding2D(padding=(1, 1)))
    #model1.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model3.add(Dropout(0.3))
    model3.add(Flatten())
    model3.add(BatchNormalization())
    
    model = model3
    #model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(.5))

    model.add(Dense(NB_CLASSES))
    model.add(Activation('sigmoid'))
    
    #optimizer = RMSprop(lr=1e-3)
    
    optimizer = SGD(lr= LR, decay = 1e-6, momentum = 0.95, nesterov = True)    
    ##opt = Adadelta(lr= .1, decay=0.995, epsilon=1e-5)  
    optimizer = Adam(lr = LR)
    
    model.compile(loss = objective, optimizer = optimizer, metrics=['accuracy'])
    
    return model, 1     
    
def make_model_m_14_5(img_rows = IMG_ROWS, img_cols = IMG_COLS, color_type = IMG_CHANNELS):   
    
    model5 = Sequential()
    model5.add(Activation(activation= center_normalize, input_shape= (IMG_CHANNELS, IMG_ROWS, IMG_COLS)))
    model5.add(ZeroPadding2D(padding=(2, 2), input_shape=(IMG_CHANNELS, IMG_ROWS, IMG_COLS)))
    model5.add(Convolution2D(16, 5, 5, border_mode='same', activation='relu'))
    model5.add(Convolution2D(24, 5, 5, border_mode='valid', activation='relu'))
    model5.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    #model5.add(ZeroPadding2D(padding=(2, 2)))
    #model4.add(Convolution2D(96, 2, 2, border_mode='valid', activation='relu'))
    #model4.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    
    model5.add(Dropout(0.3))
    model5.add(Flatten())
    model5.add(BatchNormalization())
    
    model = model5
    #model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(.5))

    model.add(Dense(NB_CLASSES))
    model.add(Activation('sigmoid'))
    
    #optimizer = RMSprop(lr=1e-3)
    
    optimizer = SGD(lr= LR, decay = 1e-6, momentum = 0.95, nesterov = True)    
    ##opt = Adadelta(lr= .1, decay=0.995, epsilon=1e-5)  
    optimizer = Adam(lr = LR)
    
    model.compile(loss = objective, optimizer = optimizer, metrics=['accuracy'])
    
    return model, 1       


def make_model_m_15(img_rows = IMG_ROWS, img_cols = IMG_COLS, color_type = IMG_CHANNELS):   
    
    model1 = Sequential()
    model1.add(Activation(activation= center_normalize, input_shape= (IMG_CHANNELS, IMG_ROWS, IMG_COLS)))
    #model1.add(ZeroPadding2D(padding=(2, 2), input_shape=(IMG_CHANNELS, IMG_ROWS, IMG_COLS)))
    model1.add(Convolution2D(32, 3, 3, border_mode='same', activation='relu'))
    #model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    #model.add(ZeroPadding2D(padding=(1, 1)))
    model1.add(Convolution2D(64, 3, 3, border_mode='valid', activation='relu'))
    model1.add(ZeroPadding2D(padding=(1, 1)))
    model1.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model1.add(ZeroPadding2D(padding=(1, 1)))
    model1.add(Convolution2D(64, 3, 3, border_mode='valid', activation='relu'))
    model1.add(ZeroPadding2D(padding=(1, 1)))
    model1.add(Convolution2D(128, 3, 3, border_mode='valid', activation='relu'))
    model1.add(ZeroPadding2D(padding=(1, 1)))
    model1.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model1.add(Dropout(0.3))
    model1.add(Flatten())
    model1.add(BatchNormalization())
    
    model2 = Sequential()
    model2.add(Activation(activation= center_normalize, input_shape= (IMG_CHANNELS, IMG_ROWS, IMG_COLS)))
    model2.add(ZeroPadding2D(padding=(2, 2)))
    model2.add(Convolution2D(32, 4, 4, border_mode='same', activation='relu'))
    model2.add(Convolution2D(48, 4, 4, border_mode='valid', activation='relu'))
    model2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model2.add(ZeroPadding2D(padding=(2, 2)))
    model2.add(Convolution2D(64, 4, 4, border_mode='valid', activation='relu'))
    #model2.add(ZeroPadding2D(padding=(1, 1)))
    #model2.add(Convolution2D(96, 4, 4, border_mode='same', activation='relu'))
    model2.add(Dropout(0.3))
    model2.add(Flatten()) 
    model2.add(BatchNormalization())
    
    
    model3 = Sequential()
    model3.add(Activation(activation= center_normalize, input_shape= (IMG_CHANNELS, IMG_ROWS, IMG_COLS)))
    #model1.add(ZeroPadding2D(padding=(1, 1), input_shape=(IMG_CHANNELS, IMG_ROWS, IMG_COLS)))
    model3.add(Convolution2D(64, 1, 1, border_mode='same', activation='relu'))
    #model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    #model.add(ZeroPadding2D(padding=(1, 1)))
    #model1.add(Convolution2D(64, 1, 1, border_mode='valid', activation='relu'))
    #model1.add(ZeroPadding2D(padding=(1, 1)))
    #model1.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model3.add(Dropout(0.3))
    model3.add(Flatten())
    model3.add(BatchNormalization())
    
    model4 = Sequential()
    model4.add(Activation(activation= center_normalize, input_shape= (IMG_CHANNELS, IMG_ROWS, IMG_COLS)))
    #model4.add(ZeroPadding2D(padding=(2, 2), input_shape=(IMG_CHANNELS, IMG_ROWS, IMG_COLS)))
    model4.add(Convolution2D(32, 2, 2, border_mode='same', activation='relu'))
    model4.add(Convolution2D(64, 2, 2, border_mode='valid', activation='relu'))
    model4.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model4.add(ZeroPadding2D(padding=(2, 2)))
    #model4.add(Convolution2D(96, 2, 2, border_mode='valid', activation='relu'))
    #model4.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    
    model4.add(Dropout(0.3))
    model4.add(Flatten())
    model4.add(BatchNormalization())
    
    
    model5 = Sequential()
    model5.add(Activation(activation= center_normalize, input_shape= (IMG_CHANNELS, IMG_ROWS, IMG_COLS)))
    model5.add(ZeroPadding2D(padding=(2, 2), input_shape=(IMG_CHANNELS, IMG_ROWS, IMG_COLS)))
    model5.add(Convolution2D(16, 5, 5, border_mode='same', activation='relu'))
    model5.add(Convolution2D(24, 5, 5, border_mode='valid', activation='relu'))
    model5.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    #model5.add(ZeroPadding2D(padding=(2, 2)))
    #model4.add(Convolution2D(96, 2, 2, border_mode='valid', activation='relu'))
    #model4.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    
    model5.add(Dropout(0.3))
    model5.add(Flatten())
    model5.add(BatchNormalization())
    
    models = [model1, model2, model3, model4, model5]
    models = [model1, model2, model4, model5]
    
    model = Sequential()
    model.add(Merge(models, mode='concat'))
   
    #model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(.5))
    
    model.add(Dense(NB_CLASSES))
    model.add(Activation('sigmoid'))
    
    #optimizer = RMSprop(lr=1e-3)
    
    optimizer = SGD(lr= LR, decay = 1e-6, momentum = 0.95, nesterov = True)    
    ##opt = Adadelta(lr= .1, decay=0.995, epsilon=1e-5)  
    optimizer = Adam(lr = LR)
    
    model.compile(loss = objective, optimizer = optimizer, metrics=['accuracy'])
    
    return model, len(models)
    

def make_model_m_15_63(img_rows = IMG_ROWS, img_cols = IMG_COLS, color_type = IMG_CHANNELS):   
    
    model6 = Sequential()
    model6.add(Activation(activation= center_normalize, input_shape= (IMG_CHANNELS, IMG_ROWS, IMG_COLS)))
    model6.add(ZeroPadding2D(padding=(3, 3)))
    model6.add(Convolution2D(32, 6, 6, subsample=(3, 3), border_mode='same', activation='relu'))
    model6.add(ZeroPadding2D(padding=(2, 2)))
    model6.add(Convolution2D(64, 6, 6, subsample=(3, 3), border_mode='valid', activation='relu'))
    model6.add(ZeroPadding2D(padding=(1, 1)))
    model6.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    #model5.add(ZeroPadding2D(padding=(2, 2)))
    #model4.add(Convolution2D(96, 2, 2, border_mode='valid', activation='relu'))
    #model4.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    
    model6.add(Dropout(0.3))
    model6.add(Flatten())
    model6.add(BatchNormalization())
    
    model = model6
    #model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(.5))

    model.add(Dense(NB_CLASSES))
    model.add(Activation('sigmoid'))
    
    #optimizer = RMSprop(lr=1e-3)
    
    optimizer = SGD(lr= LR, decay = 1e-6, momentum = 0.95, nesterov = True)    
    ##opt = Adadelta(lr= .1, decay=0.995, epsilon=1e-5)  
    optimizer = Adam(lr = LR)
    
    model.compile(loss = objective, optimizer = optimizer, metrics=['accuracy'])
    
    return model, 1  

def make_model_m_15_6(img_rows = IMG_ROWS, img_cols = IMG_COLS, color_type = IMG_CHANNELS):   
    # 5000 epoch 30 16s - loss: 0.5092 - acc: 0.7440 - val_loss: 0.5882 - val_acc: 0.6930
    model6 = Sequential()
    model6.add(Activation(activation= center_normalize, input_shape= (IMG_CHANNELS, IMG_ROWS, IMG_COLS)))
    model6.add(ZeroPadding2D(padding=(2, 2)))
    model6.add(Convolution2D(32, 6, 6, subsample=(2, 2), border_mode='same', activation='relu'))
    model6.add(ZeroPadding2D(padding=(2, 2)))
    model6.add(Convolution2D(64, 6, 6, subsample=(2, 2), border_mode='valid', activation='relu'))
    #model6.add(ZeroPadding2D(padding=(1, 1)))
    model6.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    #model5.add(ZeroPadding2D(padding=(2, 2)))
    #model4.add(Convolution2D(96, 2, 2, border_mode='valid', activation='relu'))
    #model4.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    
    model6.add(Dropout(0.3))
    model6.add(Flatten())
    model6.add(BatchNormalization())
    
    model = model6
    #model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(.5))

    model.add(Dense(NB_CLASSES))
    model.add(Activation('sigmoid'))
    
    #optimizer = RMSprop(lr=1e-3)
    
    optimizer = SGD(lr= LR, decay = 1e-6, momentum = 0.95, nesterov = True)    
    ##opt = Adadelta(lr= .1, decay=0.995, epsilon=1e-5)  
    optimizer = Adam(lr = LR)
    
    model.compile(loss = objective, optimizer = optimizer, metrics=['accuracy'])
    
    return model, 1
    
def make_model_m_15_7(img_rows = IMG_ROWS, img_cols = IMG_COLS, color_type = IMG_CHANNELS):   
    # 5000 epoch 30 16s - loss: 0.5092 - acc: 0.7440 - val_loss: 0.5882 - val_acc: 0.6930
    model7 = Sequential()
    model7.add(Activation(activation= center_normalize, input_shape= (IMG_CHANNELS, IMG_ROWS, IMG_COLS)))
    model7.add(ZeroPadding2D(padding=(3, 3)))
    model7.add(Convolution2D(24, 5, 5, subsample=(1, 1), border_mode='same', activation='relu'))
    #model7.add(ZeroPadding2D(padding=(2, 2)))
    model7.add(Convolution2D(24, 5, 5, subsample=(1, 1), border_mode='valid', activation='relu'))
    #model7.add(ZeroPadding2D(padding=(2, 2)))
    model7.add(Convolution2D(32, 3, 3, subsample=(2, 2), border_mode='valid', activation='relu'))
    #model7.add(ZeroPadding2D(padding=(1, 1)))
    #model7.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    #model7.add(ZeroPadding2D(padding=(2, 2)))
    #model7.add(Convolution2D(96, 2, 2, border_mode='valid', activation='relu'))
    model7.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    
    model7.add(Dropout(0.3))
    model7.add(Flatten())
    model7.add(BatchNormalization())
    
    model = model7
    #model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(.5))

    model.add(Dense(NB_CLASSES))
    model.add(Activation('sigmoid'))
    
    #optimizer = RMSprop(lr=1e-3)
    
    optimizer = SGD(lr= LR, decay = 1e-6, momentum = 0.95, nesterov = True)    
    ##opt = Adadelta(lr= .1, decay=0.995, epsilon=1e-5)  
    optimizer = Adam(lr = LR)
    
    model.compile(loss = objective, optimizer = optimizer, metrics=['accuracy'])
    
    return model, 1 
    
def make_model_m_15_1(img_rows = IMG_ROWS, img_cols = IMG_COLS, color_type = IMG_CHANNELS):   
    
    model1 = Sequential()
    model1.add(Activation(activation= center_normalize, input_shape= (IMG_CHANNELS, IMG_ROWS, IMG_COLS)))
    model1.add(ZeroPadding2D(padding=(1, 1), input_shape=(IMG_CHANNELS, IMG_ROWS, IMG_COLS)))
    model1.add(Convolution2D(32, 3, 3, border_mode='valid', activation='relu'))
    #model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model1.add(ZeroPadding2D(padding=(1, 1)))
    model1.add(Convolution2D(64, 3, 3, border_mode='valid', activation='relu'))
    #model1.add(ZeroPadding2D(padding=(1, 1)))
    model1.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    
    model1.add(ZeroPadding2D(padding=(1, 1)))
    model1.add(Convolution2D(128, 3, 3, border_mode='valid', activation='relu'))
    model1.add(ZeroPadding2D(padding=(2, 2)))
    model1.add(Convolution2D(256, 3, 3, border_mode='valid', activation='relu'))
    model1.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    
    model1.add(ZeroPadding2D(padding=(2, 2)))
    model1.add(Convolution2D(512, 3, 3, border_mode='valid', activation='relu'))
    #model1.add(ZeroPadding2D(padding=(2, 2)))
    #model1.add(Convolution2D(256, 3, 3, border_mode='valid', activation='relu'))
    model1.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    
    model1.add(Dropout(0.3))
    model1.add(Flatten())
    model1.add(BatchNormalization())
    
    
    
    model = model1
    
    #model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(.5))
    #model.add(BatchNormalization())
    
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(.5))
    #model.add(BatchNormalization())
    
    model.add(Dense(NB_CLASSES))
    model.add(Activation('sigmoid'))
    
    #optimizer = RMSprop(lr=1e-3)
    
    optimizer = SGD(lr= LR, decay = 1e-6, momentum = 0.95, nesterov = True)    
    ##opt = Adadelta(lr= .1, decay=0.995, epsilon=1e-5)  
    optimizer = Adam(lr = LR)
    
    model.compile(loss = objective, optimizer = optimizer, metrics=['accuracy'])
    
    return model, 1

def make_model_1114_0231():
    
    border_mode = 'same' #'valid' # 'same'
    s = 3
    st = 1 #  subsample=(2, 2)
    
    model = Sequential()

    model.add(Activation(activation= center_normalize, input_shape= (IMG_CHANNELS, IMG_ROWS, IMG_COLS)))
    #model.add(Activation(activation= center_normalize, input_shape= (IMG_ROWS, IMG_COLS, IMG_CHANNELS)))
    model.add(ZeroPadding2D(padding=(1, 1)))#, input_shape=(IMG_CHANNELS, IMG_ROWS, IMG_COLS)))
    model.add(Convolution2D(24, s, s, subsample=(st, st), border_mode= border_mode, activation='relu'))
    model.add(ZeroPadding2D(padding=(1, 1)))
    model.add(Convolution2D(32, s, s, subsample=(st, st), border_mode= border_mode, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(ZeroPadding2D(padding=(1, 1)))
    model.add(Convolution2D(32, s, s, subsample=(st, st), border_mode= border_mode, activation='relu'))
    model.add(ZeroPadding2D(padding=(2, 2)))
    model.add(Convolution2D(48, s, s, subsample=(st, st), border_mode= border_mode, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(ZeroPadding2D(padding=(1, 1)))
    model.add(Convolution2D(48, s, s, subsample=(st, st), border_mode= border_mode, activation='relu'))
    model.add(ZeroPadding2D(padding=(2, 2)))
    model.add(Convolution2D(64, s, s, subsample=(st, st), border_mode= border_mode, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(ZeroPadding2D(padding=(1, 1)))
    model.add(Convolution2D(64, s, s, border_mode= border_mode, activation='relu'))
    model.add(ZeroPadding2D(padding=(2, 2)))
    model.add(Convolution2D(96, s, s, border_mode= border_mode, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(ZeroPadding2D(padding=(1, 1)))
    model.add(Convolution2D(96, s, s, border_mode= border_mode, activation='relu'))
    model.add(ZeroPadding2D(padding=(2, 2)))
    model.add(Convolution2D(128, s, s, border_mode= border_mode, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    

    model.add(Flatten())
    #model.add(Dense(128))
    model.add(Dense(128, activation='relu', init='he_normal'))

    #model.add(Activation('relu'))
    model.add(Dropout(0.5))
    
    #model.add(Dense(128, activation='relu'))
    #model.add(Dropout(0.5))

    model.add(Dense(NB_CLASSES))
    model.add(Activation('sigmoid'))
    
    #optimizer = RMSprop(lr=1e-3)
    
    optimizer = SGD(lr= LR, decay = 1e-6, momentum = 0.95, nesterov = True)    
    ##opt = Adadelta(lr= .1, decay=0.995, epsilon=1e-5)  
    optimizer = Adam(lr = LR)
    
    model.compile(loss = objective, optimizer = optimizer, metrics=['accuracy'])
    
    return model, 1 

    
#============================================================================

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = .2)

print ('Model building...')

model, N_MERGED = make_model_1114_0231()
#N_MERGED = 4

print(model.optimizer.get_config())


early_stopping = EarlyStopping(monitor='val_loss', patience = 5, verbose = 1, mode = 'auto')        
        

print ('Model training...')
    
history = LossHistory()


        
#model.save_weights('first_try.h5')  # always save your weights after training or during training

res = model.fit([X_train]*N_MERGED, y_train, batch_size = batch_size, nb_epoch = nb_epoch, validation_data=([X_val]*N_MERGED, y_val), 
            verbose = 2, shuffle = True, callbacks = [early_stopping])#callbacks = [history, early_stopping])
#predictions = model.predict(X_test, verbose = 0) if PRED_TEST else None

val_loss = res.history['val_loss']#[0]#log_loss(y_val, pred_val)
loss = res.history['loss']
        
#loss = history.losses
#val_loss = history.val_losses
i = np.argmin(val_loss)
print()
print('Best epoch: ', i + 1, val_loss[i])



