#!/usr/bin/env python
# coding: utf-8

# **MNIST digit recognition using CNN with Keras**

# In[ ]:


# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Handle table-like data and matrices
import numpy as np
import pandas as pd
import math 
# Modelling Algorithms
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier
from sklearn.metrics import accuracy_score

# Modelling Helpers
from sklearn.preprocessing import Imputer , Normalizer , scale
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFECV
import datetime as dt

# Visualisation
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns
import keras
from keras import backend as K

# Configure visualisations
get_ipython().run_line_magic('matplotlib', 'inline')
mpl.style.use( 'ggplot' )
sns.set_style( 'white' )
pylab.rcParams[ 'figure.figsize' ] = 8 , 6
from random import randint


# In[ ]:


train = pd.read_csv('../input/digit-recognizer/train.csv')
test = pd.read_csv('../input/digit-recognizer/test.csv')


# In[ ]:


df = train.copy()


# In[ ]:


df.head()


# In[ ]:


print(df.shape , test.shape)


# In[ ]:


df.columns


# In[ ]:


y = df.label.values.astype('int32')
df = df[df.columns[1:]].values.astype('float32')

X_train , X_test , y_train , y_test = train_test_split(df , y , test_size = 0.2 , random_state = 100)


# In[ ]:


print ( X_train.shape , y_train.shape)


# In[ ]:


test = test.values.astype('float32')


# In[ ]:


X_train = X_train.reshape( -1 , 28 , 28 , 1)
X_test = X_test.reshape( -1 , 28 , 28 , 1)
test = test.reshape( -1 , 28 , 28 , 1)


# In[ ]:


X_train.shape


# In[ ]:


new = pd.read_csv('../input/digit-recognizer/train.csv')


# In[ ]:


new.describe()


# In[ ]:


label_counter = new.label.value_counts()
print (label_counter)


# In[ ]:


plt.subplots(figsize = (8,5))
plt.title('Count of different digits as labeled in the datset')
sns.countplot(x=new.label , data=new)
plt.show()


# **# Normalizing**

# Now we have 33,600 images with dimensions of 28 x 28 x 1 in X_train .

# In[ ]:


X_train = X_train / 255
X_test = X_test / 255
test = test / 255


# 
# Converting y_train and y_test into one hot encoder , eg. 3--> [0,0,0,1,0,0,0,0,0,0]

# In[ ]:


y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)  #  10 is used because we have to classify images in 10 groups


# In[ ]:


from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau


# Making the model with convolutional layers , maxpool layers and dropout layers to enhance regularization and generalization with further avoidance to overfitting .

# In[ ]:


model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', padding='same',input_shape=(28 , 28 , 1)))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))      #  10 is used because we have to classify images in 10 groups


# In[ ]:


print (model.summary())


# In[ ]:


model.compile(optimizer = RMSprop(lr=0.001),loss='categorical_crossentropy', metrics=['accuracy'])


# In[ ]:


datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.0, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images

datagen.fit(X_train)


# In[ ]:


batch_size = 64
epochs = 10
lr_reduce = ReduceLROnPlateau(monitor='val_acc', factor=0.1, epsilon=0.0001, patience=1, verbose=1)


# In[ ]:


model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size),
                    steps_per_epoch=X_train.shape[0] // batch_size,
                    callbacks=[lr_reduce],
                    validation_data=(X_test, y_test),
                    epochs = epochs, verbose = 2)


# In[ ]:


score = model.evaluate(X_test, y_test, verbose=0)
print('valid loss:', score[0])
print('valid accuracy:', score[1])


# In[ ]:


pred = model.predict(test)


# In[ ]:


pred.shape


# In[ ]:


pred


# In[ ]:


pred_digits = np.argmax(pred , axis = 1)
ImageId = range( 1 , len(pred_digits)+1 )


# pred_digits contains the predicted labels of the digits .
# argmax(axis = 1) is converting the one hot encoder back to the labels along the row axis .

# In[ ]:


pred_digits


# In[ ]:


len(ImageId)


# In[ ]:


context = {"ImageId" : ImageId , "Label" : pred_digits }
ans = pd.DataFrame(context)


# ans is the predicted dataframe by CNN

# In[ ]:


ans.head()


# Converting the dataframe to csv file

# In[ ]:


ans.to_csv("Predictions by CNN.csv", index=None)

