import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
# -*- coding: utf-8 -*-
"""

"""
LR = .001
ROWS = 128
COLS = 128
CHANNELS = 3
NB_CLASSES = 1
TRIM_DATA_COUNT = 500
PRED_TEST = 0
objective = 'binary_crossentropy' #'binary_crossentropy'

import os, cv2, random
import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score

import matplotlib.pyplot as plt
#from matplotlib import ticker
#import seaborn as sns
#%matplotlib inline 

from keras.models import Sequential
from keras.layers import Input, Dropout, Flatten, Convolution2D, MaxPooling2D, Dense, Activation, ZeroPadding2D
from keras.optimizers import RMSprop, SGD, Adadelta, Adam
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from keras.utils import np_utils
from keras.layers.advanced_activations import LeakyReLU, PReLU

TRAIN_DIR = '../input/train/'
TEST_DIR = '../input/test/'

#TRAIN_DIR = 'data/train/'
#TEST_DIR = 'data/test/'







## Callback for loss logging per epoch
class LossHistory(Callback):
    
    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_losses = []
        self.aucs = []
        self.lrs = []
        
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
        try:
            self.lrs.append(self.model.optimizer.lr.get_value())
        except:
            pass

def read_image(file_path):
    img = cv2.imread(file_path, cv2.IMREAD_COLOR) #cv2.IMREAD_GRAYSCALE
    return cv2.resize(img, (ROWS, COLS), interpolation = cv2.INTER_CUBIC)
    
    
def prep_data(images):
    count = len(images)
    data = np.ndarray((count, CHANNELS, ROWS, COLS), dtype = np.uint8)
    
    for i, image_file in enumerate(images):
        image = read_image(image_file)
        data[i] = image.T
        if i % 200 == 0: 
            print('Processed {} of {}'.format(i, count))
        
    return data
        
def prep_data1(images):
    count = len(images)
    data = [] #np.ndarray((count, CHANNELS * ROWS * COLS), dtype = np.uint8)
    
    for i, image_file in enumerate(images):
        image = read_image(image_file)#.flatten()
        data.append(image)
        if i % 200 == 0: 
            print('Processed {} of {}'.format(i, count))
        
    data = np.array(data, dtype=np.float32) # uint8
    data = data.transpose((0, 3, 1, 2))
    return data
        
        
# ===================================================

try:
    print (X.shape)
except:

    #train_images = [TRAIN_DIR + i for i in os.listdir(TRAIN_DIR)] # use this for full dataset
    train_dogs =   [TRAIN_DIR + f for f in os.listdir(TRAIN_DIR) if 'dog' in f]
    train_cats =   [TRAIN_DIR + f for f in os.listdir(TRAIN_DIR) if 'cat' in f]
    
    if PRED_TEST:
        test_images =  [TEST_DIR + f for f in os.listdir(TEST_DIR)]
    
    
    if TRIM_DATA_COUNT:
        train_images = train_dogs[:TRIM_DATA_COUNT] + train_cats[:TRIM_DATA_COUNT]
        if PRED_TEST:
            test_images =  test_images[:TRIM_DATA_COUNT]
    else:
        train_images = [TRAIN_DIR + f for f in os.listdir(TRAIN_DIR)] # use this for full dataset
            
    random.shuffle(train_images)
    
    
    
    X = (prep_data(train_images) / 255.).astype(np.float32)
    if PRED_TEST:
        X_test = (prep_data(test_images) / 255.).astype(np.float32)
    
    labels = [int('cat' in img) for img in train_images]
    
    if NB_CLASSES > 1:
        y = np_utils.to_categorical(labels, NB_CLASSES)
    else:
        y = labels
    
    print(X.shape, X.min(), X.max(), np.median(X))


print("Train shape: {}".format(X.shape))
if PRED_TEST:
    print("Test shape: {}".format(X_test.shape))

#print(X[0])
print(y)




def make_model_0():
    
    model = Sequential()
    
    model.add(ZeroPadding2D((1, 1), input_shape=(3, ROWS, COLS)))
    model.add(Convolution2D(32, 3, 3))
    #model.add(Convolution2D(32, 3, 3, input_shape=(3, ROWS, COLS)))
    model.add(Activation('relu'))
    #model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(32, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Flatten())  
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    
    return model


def make_model_01():
    
    model = Sequential()
    
    model.add(ZeroPadding2D((1, 1), input_shape=(3, ROWS, COLS)))
    
    model.add(Flatten())  
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    
    return model
    
    
def make_model_1():
    
    model = Sequential()

    model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=(3, ROWS, COLS), activation='relu'))
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

def make_model_11(rows = ROWS, cols = COLS):
    
    border_mode = 'same' #'valid' # 'same'
    objective = 'binary_crossentropy' #'binary_crossentropy'
    s = 4
    n = 16
    
    model = Sequential()

    model.add(ZeroPadding2D((1, 1), input_shape=(3, rows, cols)))
    model.add(Convolution2D(n, 3, 3, border_mode= border_mode, activation='relu'))
    #model.add(Convolution2D(n, 3, 3, border_mode= border_mode, input_shape=(3, ROWS, COLS), activation='relu'))
    
    #model.add(Convolution2D(32, s, s, border_mode= border_mode, input_shape=(3, ROWS, COLS)))
    #model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=(3, ROWS, COLS), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(n, s, s, border_mode= border_mode, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(n*2, s, s, border_mode= border_mode, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(n*2, s, s, border_mode= border_mode, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(n*4, s, s, border_mode= border_mode, activation='relu'))
    model.add(Convolution2D(n*4, s, s, border_mode= border_mode, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    

#     model.add(ZeroPadding2D((1, 1)))
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
    
    #optimizer = RMSprop(lr=5e-4)
    #optimizer = SGD(lr= LR, decay = 1e-6, momentum = 0.9, nesterov = True) 
    optimizer = 'rmsprop' #'Adam'
    optimizer = Adadelta(lr = LR, decay = 0.995, epsilon = 1e-5)    
    model.compile(loss = objective, optimizer = optimizer, metrics = ['accuracy'])
    
    return model

def make_model_16(img_rows = ROWS, img_cols = COLS, color_type = 3):
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(color_type,
                                                 img_rows, img_cols)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='softmax'))

    #model.load_weights('../input/vgg16_weights.h5')

    # Code above loads pre-trained data and
    model.layers.pop()
    model.add(Dense(NB_CLASSES, activation='softmax'))
    # Learning rate is changed to 0.001
    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss= objective) # 'categorical_crossentropy'
    
    return model
    
def create_model_31(img_rows = ROWS, img_cols = COLS, color_type = 3):

    nb_classes = NB_CLASSES    
    nb_filters = 16    # number of convolutional filters to use
    nb_pool = 2    # size of pooling area for max pooling
    nb_conv = 3 # convolution kernel size
    n = 512
    alpha = 0.3
    LR = .02
    
    model = Sequential()
    
    model.add(Convolution2D(nb_filters, nb_conv, nb_conv, border_mode='valid', input_shape=(color_type, img_rows, img_cols)))
    #model.add(Activation('relu'))
    model.add(LeakyReLU(alpha= alpha))   #keras.layers.advanced_activations.LeakyReLU(alpha=0.3)  PReLU(init='zero', weights=None)
    model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
    #model.add(Activation('relu'))
    model.add(LeakyReLU(alpha= alpha))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    model.add(Dropout(0.5))
    
    model.add(Convolution2D(nb_filters * 2, nb_conv, nb_conv,  border_mode='valid',
                            input_shape=(color_type, img_rows, img_cols)))
    #model.add(Activation('relu'))
    model.add(LeakyReLU(alpha= alpha))
    model.add(Convolution2D(nb_filters * 2, nb_conv, nb_conv))
    #model.add(Activation('relu'))
    model.add(LeakyReLU(alpha= alpha))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    
    model.add(Dense(n))
    #model.add(Activation('relu'))    
    model.add(LeakyReLU(alpha= alpha))
    #model.add(BatchNormalization())
    model.add(Dropout(0.5))
    
    model.add(Dense(n))
    #model.add(Activation('relu'))
    model.add(LeakyReLU(alpha= alpha))
    #model.add(BatchNormalization())
    model.add(Dropout(0.5))
    
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    #model.compile(loss='categorical_crossentropy', optimizer='adadelta')
    opt = SGD(lr= LR, decay = 1e-6, momentum = 0.95, nesterov = True)    
    #opt = Adadelta(lr= .1, decay=0.995, epsilon=1e-5)
    
    model.compile(optimizer= opt, loss = objective)
    
    return model   
    

#============================================================================

print ('Model building...')

model = make_model_11() #(rows = ROWS, cols = COLS)
#model = create_model_31(img_rows = ROWS, img_cols = COLS, color_type = 3)


nb_epoch = 20
batch_size = 32

early_stopping = EarlyStopping(monitor='val_loss', patience = 5, verbose = 1, mode = 'auto')        
        

print ('Model training...')
    
history = LossHistory()
model.fit(X, y, batch_size = batch_size, nb_epoch = nb_epoch,
          validation_split = 0.2, verbose = 2, shuffle = True, callbacks = [history, early_stopping])

predictions = model.predict(X_test, verbose=0) if PRED_TEST else None
    



loss = history.losses
val_loss = history.val_losses
print(history.aucs)
print(history.lrs)
#print(model.get_config())
print(model.optimizer.get_config())

plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Trend')
plt.plot(loss, 'blue', label='Training Loss')
plt.plot(val_loss, 'green', label='Validation Loss')
plt.xticks(range(0, nb_epoch)[0::2])
plt.legend()
plt.show()

