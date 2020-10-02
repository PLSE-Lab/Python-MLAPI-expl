#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Ignore  the warnings
import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')

# data visualisation and manipulation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns
 
#configure
# sets matplotlib to inline and displays graphs below the corressponding cell.
get_ipython().run_line_magic('matplotlib', 'inline')
style.use('fivethirtyeight')
sns.set(style='whitegrid',color_codes=True)

#model selection
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score,precision_score,recall_score,confusion_matrix,roc_curve,roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder

#preprocess.
from keras.preprocessing.image import ImageDataGenerator

#dl libraraies
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense,Input
from keras.optimizers import Adam,SGD,Adagrad,Adadelta,RMSprop
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint,EarlyStopping,TensorBoard,CSVLogger,ReduceLROnPlateau,LearningRateScheduler

# specifically for cnn
from keras.layers import Dropout, Flatten,Activation
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization,GlobalAveragePooling2D
import tensorflow as tf
import random as rn

# specifically for manipulating zipped images and getting numpy arrays of pixel values of images.
import cv2                  
import numpy as np  
from tqdm import tqdm
import os                   
from random import shuffle  
from zipfile import ZipFile
from PIL import Image

# hyper parameter 
import optuna 
import keras.backend as K
from keras.models import Model


# In[ ]:


IMG_SIZE = 28
NUM_CLASSES = 10
TEST_SIZE = 0.2
RANDOM_STATE = 1234
#Model
NO_EPOCHS = 50
BATCH_SIZE = 128

PATH="../input/japanese handwritten digits/Japanese Handwritten Digits/"
print(os.listdir(PATH))


# In[ ]:


# Create a dictionary for each type of label 
labels = {0 : "00", 1: "01", 2: "02", 3: "03", 4: "04",
          5: "05", 6: "06", 7: "07", 8: "08", 9: "09"}


# In[ ]:


X=[]
Z=[]


# In[ ]:


def assign_label(img,label):
    return label


# In[ ]:


def make_train_data(label,DIR):
    for img in tqdm(os.listdir(DIR)):
        label=assign_label(img,label)
        path = os.path.join(DIR,img)
        img = cv2.imread(path,cv2.IMREAD_COLOR)
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        
        X.append(np.array(img))
        Z.append(str(label))


# In[ ]:


for label in labels.values():
    make_train_data(label,PATH + label)


# In[ ]:


# check some image
fig,ax=plt.subplots(5,2)
fig.set_size_inches(15,15)
for i in range(5):
    for j in range (2):
        l=rn.randint(0,len(Z))
        ax[i,j].imshow(X[l])
        ax[i,j].set_title(Z[l])
        
plt.tight_layout()


# In[ ]:


le=LabelEncoder()
Y=le.fit_transform(Z)
Y=to_categorical(Y)
X=np.array(X)
X=X/255


# In[ ]:


# separate data
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.3,random_state=42)
x_test,x_valid,y_test,y_valid=train_test_split(x_test,y_test,test_size=0.20,random_state=42)


# In[ ]:


# fix random seed
np.random.seed(42)
rn.seed(42)
tf.set_random_seed(42)


# In[ ]:


x_train.shape


# In[ ]:


def create_model(num_layer, mid_units, num_filters):
    
    model = Sequential()
    model.add(Conv2D(filters = num_filters[0], kernel_size = (5,5),padding = 'Same',activation ='relu', input_shape = (28,28,3)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    for i in range(1,num_layer):
        model.add(Conv2D(filters = num_filters[i], kernel_size = (3,3),padding = 'Same',activation ='relu'))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    
    model.add(Flatten())
    model.add(Dense(mid_units))
    model.add(Activation('relu'))
    model.add(Dense(10, activation = "softmax"))
    
    return model

def objective(trial):
    
    K.clear_session()

    num_layer = trial.suggest_int("num_layer", 3 , 4)

    mid_units = int(trial.suggest_discrete_uniform("mid_units", 128, 512, 128))

    num_filters = [int(trial.suggest_discrete_uniform("num_filter_"+str(i), 32, 128, 32)) for i in range(num_layer)]

    model = create_model(num_layer, mid_units, num_filters)
    model.compile(optimizer=Adam(lr=0.001),
          loss="categorical_crossentropy",
          metrics=["accuracy"])
    
    history = model.fit(x=x_train,y=y_train, batch_size=128,
                        epochs = 20, validation_data = (x_valid,y_valid))

    return 1 - history.history["val_acc"][-1]


# In[ ]:


study = optuna.create_study()
study.optimize(objective, n_trials=100)


# In[ ]:


study.best_params


# In[ ]:


from keras.callbacks import ReduceLROnPlateau
reduce = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.1,
    patience=3,
    verbose=1, 
    mode='auto'
)
earlystop = EarlyStopping(
    monitor='val_loss',
    min_delta=0.001,
    patience=30,
    verbose=1,
    mode='auto'
)
callbacks = [reduce,earlystop]


# In[ ]:


K.clear_session()
num_layer = int(study.best_params['num_layer'])
mid_units = int(study.best_params['mid_units'])
num_filters = [int(study.best_params['num_filter_{}'.format(i)]) for i in range(num_layer)]
model = create_model(num_layer, mid_units, num_filters)
model.compile(optimizer=Adam(lr=0.001),
      loss="categorical_crossentropy",
      metrics=["accuracy"])


# In[ ]:


model.summary()


# In[ ]:


datagen = ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
    zoom_range = 0.1, # Randomly zoom image 
    width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=True,  # randomly flip images
    vertical_flip=False)  # randomly flip images

datagen.fit(x_train)


# In[ ]:


history = model.fit_generator(datagen.flow(x_train,y_train, batch_size=128),
                              epochs = 100, validation_data = (x_valid,y_valid),
                              steps_per_epoch=x_train.shape[0] // 128 ,
                              verbose = 1,
                              callbacks=callbacks)


# In[ ]:


# getting predictions on val set.
pred=model.predict(x_test)
pred_digits=np.argmax(pred,axis=1)


# In[ ]:


accuracy_score(y_test.argmax(1),pred_digits)


# In[ ]:




