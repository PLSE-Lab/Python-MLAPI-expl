#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os,cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pylab import rcParams
rcParams['figure.figsize'] = 20, 10

from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split

import keras

from keras.utils import np_utils

from keras import backend as K

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD,RMSprop,adam, Adadelta
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd


# In[ ]:


import os
for dirname, _, filenames in os.walk('../input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


# get the data
filname = '../input/facial-expression/fer2013/fer2013.csv'
label_map = ['Anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
names=['emotion','pixels','usage']
df=pd.read_csv('../input/facial-expression/fer2013/fer2013.csv',names=names, na_filter=False)
im=df['pixels']
df.head(10)


# In[ ]:


a = df['emotion']
idx = pd.Index(a)
count = idx.value_counts()
print(count)


# In[ ]:


from keras.models import Sequential
from keras.layers import Dense , Activation , Dropout ,Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.metrics import categorical_accuracy
from keras.models import model_from_json
from keras.callbacks import ModelCheckpoint
from keras.optimizers import *
from keras.layers.normalization import BatchNormalization


# In[ ]:


def getData(filname):
    # images are 48x48
    # N = 35887
    Y = []
    X = []
    first = True
    for line in open(filname):
        #This condition skips the first condition
        if first:
            first = False
        else:
            row = line.split(',')
            Y.append(int(row[0]))
            X.append([int(p) for p in row[1].split()])

    #X, Y = np.array(X) / 255.0, np.array(Y)
    X, Y = np.array(X)/255.0 , np.array(Y)
    return X, Y


# In[ ]:


X, Y = getData(filname)
num_class = len(set(Y))
print(num_class)


# Preprocess Image to detect only face

# In[ ]:


X.shape


# In[ ]:


# keras with tensorflow backend
N,D = X.shape
X = X.reshape(N, 48, 48, 1)


# **Split Train Test data**

# In[ ]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
y_train = (np.arange(num_class) == y_train[:, None]).astype(np.float32)
y_test = (np.arange(num_class) == y_test[:, None]).astype(np.float32)


# In[ ]:


img = X_train[780].reshape(48,48)
plt.imshow(img, interpolation='nearest')
plt.show()


# In[ ]:


from keras.applications import VGG19
#Load the VGG model
vgg_conv = VGG19(weights=None, include_top=False, input_shape=(48, 48,1))


# In[ ]:


def vgg_custom():
    model = Sequential()
    #add vgg conv model
    model.add(vgg_conv)
    
    #add new layers
    model.add(Flatten())
    model.add(Dense(7,  kernel_initializer='normal'))
    model.compile(loss='mean_squared_error', optimizer=Adadelta())
    #model.compile(loss='mean_squared_error', optimizer=Adam())
    
    return model


# In[ ]:


model = vgg_custom()
model.summary()


# In[ ]:


from keras import callbacks
filename='model_train_new.csv'
filepath="Best-weights-my_model-{epoch:03d}-{loss:.4f}-{acc:.4f}.hdf5"

csv_log=callbacks.CSVLogger(filename, separator=',', append=False)
checkpoint = callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [csv_log,checkpoint]
callbacks_list = [csv_log]


# Image Augmentation

# In[ ]:


aug = ImageDataGenerator(
    rotation_range=25, width_shift_range=0.1,
    height_shift_range=0.1, shear_range=0.2, 
    zoom_range=0.2,horizontal_flip=True, 
    fill_mode="nearest")


# Define Batch Size and Epoch iteration

# In[ ]:


BS =8
EPOCHS = 50


# In[ ]:


len(X_train)


# Fit augmentated Data into model

# In[ ]:


history = model.fit(
    X_train, y_train, batch_size=BS,
    validation_data=(X_test, y_test),
    epochs=EPOCHS, verbose=1,
    callbacks = callbacks_list 
    ,shuffle = True
    )


# In[ ]:


#Model Save
model.save_weights('model_weights.h5')
model.save('model_keras.h5')


# In[ ]:


hist = history


# In[ ]:


# visualizing losses and accuracy
# %matplotlib inline

train_loss=hist.history['loss']
val_loss=hist.history['val_loss']
#train_acc=hist.history['acc']
#val_acc=hist.history['val_acc']

epochs = range(len(val_loss))

plt.plot(epochs,train_loss,'r-o', label='train_loss')
plt.plot(epochs,val_loss,'b', label='val_loss')
plt.title('train_loss vs val_loss')
#plt.plot(epochs,train_loss,'r-o', label='train_acc')
#plt.plot(epochs,val_loss,'b', label='val_acc')
#plt.title('train_acc vs val_acc')
plt.title('train_loss vs val_loss')
plt.legend()
plt.figure()
#plt.savefig('train_test_acc.png')
plt.savefig('train_test.png')


# In[ ]:


train_acc=hist.history['acc']
val_acc=hist.history['val_acc']
epochs=range(len(val_acc))
plt.plot(epochs,train_acc,'r-o', label='train_acc')
plt.plot(epochs, val_acc,'b', label='val_acc')
plt.title('train_acc vs val_acc')
plt.legend()
plt.figure()
plt.savefig('Acc_Train_Val.png')


# In[ ]:


print(train_loss)


# In[ ]:


print(val_loss)


# In[ ]:


# Evaluating the model
score = model.evaluate(X_test, y_test, verbose=0)
print('Test Loss:', score)

test_image = X_test[0:1]
print (test_image.shape)

#predict
y_pred = model.predict(X_test) 

print(model.predict(test_image))
print(model.predict_classes(test_image))
print(y_test[0:1])

res = model.predict_classes(X_test[9:18])
plt.figure(figsize=(10, 10))


# In[ ]:


from sklearn.metrics import confusion_matrix
results = model.predict_classes(X_test)
cm = confusion_matrix(np.where(y_test == 1)[1], results)
#cm = cm.astype(np.float) / cm.sum(axis=1)[:, np.newaxis]


# Confusion Matrix

# In[ ]:


import seaborn as sns


# In[ ]:


label_mapdisgust = ['Anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']


# In[ ]:


#Transform to df for easier plotting
cm_df = pd.DataFrame(cm, index = label_mapdisgust,
                     columns = label_mapdisgust
                    )


# In[ ]:


final_cm = cm_df.drop('Disgust',axis=0)
final_cm = final_cm.drop('Disgust',axis=1)


# In[ ]:


final_cm


# In[ ]:


plt.figure(figsize = (5,5))
sns.heatmap(final_cm, annot = True,cmap='Greys',cbar=False,linewidth=2,fmt='d')
plt.title('CNN Emotion Classify')
plt.ylabel('True class')
plt.xlabel('Prediction class')
plt.show()


# ROC Curve

# In[ ]:


from sklearn.metrics import roc_curve,auc
from itertools import cycle


# In[ ]:


new_label = ['Anger', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
final_label = new_label
new_class = 6


# In[ ]:


#ravel flatten the array into single vector
y_pred_ravel = y_pred.ravel()
lw = 2


# In[ ]:


fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(new_class):
    fpr[i], tpr[i], _ = roc_curve(y_test[:,i], y_pred[:,i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    
#colors = cycle(['red', 'green','black'])
colors = cycle(['red', 'green','black','blue', 'yellow','purple'])
for i, color in zip(range(new_class), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0}'''.format(final_label[i]))
    

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()


# In[ ]:





# In[ ]:


#keras.optimizers.SGD(learning_rate=0.01, momentum=0.0, nesterov=False)
#keras.optimizers.RMSprop(learning_rate=0.001, rho=0.9)
#keras.optimizers.Adagrad(learning_rate=0.01)
#keras.optimizers.Adadelta(learning_rate=1.0, rho=0.95)
#keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
#keras.optimizers.Adamax(learning_rate=0.002, beta_1=0.9, beta_2=0.999)
#keras.optimizers.Nadam(learning_rate=0.002, beta_1=0.9, beta_2=0.999)


# 
