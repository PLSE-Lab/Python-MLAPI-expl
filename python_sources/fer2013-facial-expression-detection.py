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

from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.datasets import make_classification
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


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
    #model.compile(loss='mean_squared_error', optimizer=adam())
    model.compile(loss='mean_squared_error', optimizer=RMSprop())
    
    return model


# In[ ]:


model = vgg_custom()
model.summary()


# Image Augmentation

# In[ ]:


aug = ImageDataGenerator(
    rotation_range=25, width_shift_range=0.1,
    height_shift_range=0.1, shear_range=0.2, 
    zoom_range=0.2,horizontal_flip=True, 
    fill_mode="nearest")


# Define Batch Size and Epoch iteration

# In[ ]:


BS = 8 
EPOCHS = 200


# In[ ]:


len(X_train)


# Fit augmentated Data into model

# In[ ]:


from sklearn.model_selection import KFold


# In[ ]:


kf = KFold(n_splits=5, shuffle=False)


# In[ ]:


result = []
scores_acc = []
k_no = 0
for train_index, test_index in kf.split(X):
    X_Train_ = X[train_index]
    Y_Train = Y[train_index]
    X_Test_ = X[test_index]
    Y_Test = Y[test_index]
    
    Y_Train = (np.arange(num_class) == Y_Train[:, None]).astype(np.float32)
    Y_Test = (np.arange(num_class) == Y_Test[:, None]).astype(np.float32)

    file_path = "/kaggle/working/weights_best_"+str(k_no)+".hdf5"
    checkpoint = ModelCheckpoint(file_path, monitor='loss', verbose=0, save_best_only=True, mode='min')
    early = EarlyStopping(monitor="loss", mode="min", patience=8)

    callbacks_list = [checkpoint, early]

    model = model
    hist = model.fit_generator(aug.flow(X_Train_, Y_Train),steps_per_epoch=len(X_Train_) // BS, epochs=EPOCHS,validation_data=(X_Test_, Y_Test), callbacks=callbacks_list, verbose=1)
    # model.fit(X_Train, Y_Train, batch_size=batch_size, epochs=epochs, validation_data=(X_Test, Y_Test), verbose=1)
    model.load_weights(file_path)
    result.append(model.predict(X_Test_))
    score = model.evaluate(X_Test_,Y_Test, verbose=0)
    scores_acc.append(score)
    k_no+=1


# In[ ]:


print(scores_acc)


# In[ ]:


value_max = max(scores_loss)
value_index = scores_acc.index(value_max)
print(value_index)


# In[ ]:


model.load_weights("/kaggle/working/weights_best_"+str(value_index)+".hdf5")


# In[ ]:


#Model Save
model.save_weights('model_weights.h5')
model.save('model_keras.h5')


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


print(train_loss)


# In[ ]:


print('accuracy: {} loss: {}'.format((1-train_loss[-1])*100,(train_loss[-1])*100))


# In[ ]:


print(val_loss)


# In[ ]:


print('accuracy: {} loss: {}'.format((1-val_loss[-1])*100,(val_loss[-1])*100))


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
