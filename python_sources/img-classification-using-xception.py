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


train_df= pd.read_csv('/kaggle/input/gala-party-image-dataset/dataset/train.csv')
test_df= pd.read_csv('/kaggle/input/gala-party-image-dataset/dataset/test.csv')


# In[ ]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from keras.preprocessing import image
import numpy as np
import pandas as pd
import sys
import pickle
from PIL import Image, ImageFilter
from tqdm import tqdm_notebook
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, mean_squared_error, log_loss, confusion_matrix
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')
import os
import seaborn as sns


# In[ ]:


from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop, Adam
from keras.preprocessing.image import ImageDataGenerator


# In[ ]:


#dl libraraies
from keras import backend as K
from keras.optimizers import Adam,SGD,Adagrad,Adadelta,RMSprop
from keras.callbacks import ReduceLROnPlateau
# specifically for cnn
from keras.layers import Dropout, Flatten,Activation
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
 
import tensorflow as tf
import random as rn


# In[ ]:


train_image = []
for i in tqdm(range(train_df.shape[0])):
    print(i)
    img = image.load_img('/kaggle/input/gala-party-image-dataset/dataset/Train Images/'+train_df['Image'][i], target_size=(28,28,1), grayscale=False)
    img = image.img_to_array(img)
    img = img/255
    train_image.append(img)
Xtrain = np.array(train_image)


# In[ ]:


test_image = []
for i in tqdm(range(test_df.shape[0])):
    print(i)
    img = image.load_img('/kaggle/input/gala-party-image-dataset/dataset/Test Images/'+test_df['Image'][i], target_size=(28,28,1), grayscale=False)
    img = image.img_to_array(img)
    img = img/255
    train_image.append(img)
Xtest = np.array(train_image)


# In[ ]:


Xtrain.shape


# In[ ]:


'''
def read_all(folder_path, key_prefix=""):
    '''
    #It returns a dictionary with 'file names' as keys and 'flattened image arrays' as values.
    '''
    print("Reading:")
    images = {}
    files = os.listdir(folder_path)
    for i, file_name in tqdm_notebook(enumerate(files), total=len(files)):
        file_path = os.path.join(folder_path, file_name)
        image_index = key_prefix + file_name[:]
        image = Image.open(file_path)
        image = image.convert("L")
        image = image.resize((32,32),Image.BICUBIC)
        images[image_index] = np.array(image.copy()).flatten()
        image.close()
    return images
    '''


# In[ ]:


Ytrain= train_df['Class']


# In[ ]:


from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
Ytrain = labelencoder.fit_transform(Ytrain)


# In[ ]:


Ytrain= Ytrain.reshape(-1,1)
from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder()
Ytrain = onehotencoder.fit_transform(Ytrain).toarray()


# In[ ]:


from keras.applications.xception import Xception
from keras.layers import Activation, Dense,GlobalAveragePooling2D, Dropout
from keras.models import Model
from keras.optimizers import SGD
from keras.metrics import categorical_accuracy, top_k_categorical_accuracy

base_model = Xception(weights='imagenet', include_top=False )

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(512, activation='relu')(x)

x = Dropout(0.3)(x)
# and a logistic layer -- let's say we have 200 classes
predictions = Dense(4, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

model.summary()


# In[ ]:


batch_size= 32
epochs= 25

'''from keras.callbacks import EarlyStopping, ReduceLROnPlateau
# red_lr= ReduceLROnPlateau(monitor='val_acc',patience=3,verbose=1,factor=0.1)


earlystop = EarlyStopping(monitor='val_loss',
                          min_delta=0,
                          patience=epochs,
                          verbose=1,
                          restore_best_weights=True
                          )

reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                              factor=0.2,
                              patience=5,
                              verbose=1,
                              min_delta=0.0001)

callbacks = [earlystop,reduce_lr]'''


# In[ ]:


X_train, X_val, Y_train, Y_val= train_test_split(Xtrain, Ytrain, test_size= 0.2, random_state= 2) 


# In[ ]:


datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=20,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images


datagen.fit(X_train)


# In[ ]:


from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())


# In[ ]:


model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])


# In[ ]:


import tensorflow as tf
with tf.device('/device:GPU:0'):
    history = model.fit_generator(datagen.flow(X_train,Y_train, batch_size=batch_size),
                              epochs = epochs, 
                              validation_data = (X_val, Y_val),
                              verbose = 1)


# In[ ]:


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[ ]:


import cvbbd


# In[ ]:


images_train


# In[ ]:


images_test = read_all("../input/gala-party-image-dataset/dataset"+"/Test Images")


# In[ ]:


train = pd.DataFrame.from_dict(images_train, orient='index')
test = pd.DataFrame.from_dict(images_test, orient='index')


# In[ ]:


train['Image']= train.index


# In[ ]:


train.dropna(axis=1, how='all')


# In[ ]:


train= pd.merge(train_df, train, on= 'Image')


# In[ ]:


train.set_index('Image', inplace= True)


# In[ ]:


train


# In[ ]:


test['Image']= test.index


# In[ ]:


test= pd.merge(test_df, test, on= 'Image')


# In[ ]:


test.set_index('Image', inplace= True)


# In[ ]:





# In[ ]:


Xtrain= train.drop('Class', axis=1)


# In[ ]:


p= plt.hist(Ytrain)


# In[ ]:


Xtrain


# In[ ]:


Xtrain[5][0]


# In[ ]:


df= pd.DataFrame()


# In[ ]:


df['Pixels']= ''
for i in range(Xtrain.shape[0]): 
    x= []
    x= [int(Xtrain[j][i]) for j in range(0,1023)]
    x= np.asarray(x).reshape(32,32)
    df['Pixels'].append(x)
    print(x)


# In[ ]:


df.shape


# In[ ]:


import alkaoptyur


# In[ ]:


Xtrain = Xtrain.values.reshape(-1,32,32,1)
test = test.values.reshape(-1,32,32,1)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


X_train, X_val, Y_train, Y_val= train_test_split(Xtrain, Ytrain, test_size= 0.2, random_state= 2) 


# In[ ]:


Xtrain.shape


# In[ ]:


fig,ax=plt.subplots(5,2)
fig.set_size_inches(15,15)
for i in range(5):
    for j in range (2):
        l=rn.randint(0,len(Ytrain))
        ax[i,j].imshow(Xtrain[l])
        ax[i,j].set_title('Flower: '+Ytrain[l])
        
plt.tight_layout()
        


# In[ ]:


np.random.seed(42)
rn.seed(42)
tf.set_random_seed(42)


# In[ ]:


base_model= VGG16(include_top= 'False', weights= 'imagenet', input_shape= )


# In[ ]:


from keras.preprocessing import image
gen = image.ImageDataGenerator(rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')
batches = gen.flow(X_train, Y_train, batch_size=64)
val_batches=gen.flow(X_val, Y_val, batch_size=64)


# In[ ]:


history = model.fit_generator(generator= batches,
                              epochs = epochs, validation_data = (X_val,Y_val),
                              verbose = 1, steps_per_epoch=X_train.shape[0] // batch_size
                              )


# In[ ]:




