#!/usr/bin/env python
# coding: utf-8

# **DATASET**
# 
# The HAM10000 dataset,is a large collection of multi-source dermatoscopic images of common pigmented skin lesions.
# 
# It contains 10015 dermatoscopic images of 7 different skin cancer categories:
# * 1.Actinic keratoses and intraepithelial carcinoma / Bowen's disease (akiec)
# * 2.Basal cell carcinoma (bcc)
# * 3.Benign keratosis-like lesions (solar lentigines / seborrheic keratoses and lichen-planus like keratoses), (bkl)
# * 4.Dermatofibroma (df)
# * 5.Melanoma (mel)
# * 6.Melanocytic nevi (nv)
# * 7.Vascular lesions (angiomas, angiokeratomas, pyogenic granulomas and hemorrhage, vasc).
# 
# you can download the dataset from: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T
# 
# 

# **DATA TREE**
# 
# All the images were arranged in the following way:
# 
# (train:60%)  (validation 20%)  (test:20%)
# * TRAIN:
#          * akiec
#          * bcc
#          * bkl
#          * df
#          * mel
#          * nv
#          * vasc
#          
# * VALIDATION:
#          * akiec
#          * bcc
#          * bkl
#          * df
#          * mel
#          * nv
#          * vasc
# * TEST:
#          * akiec
#          * bcc
#          * bkl
#          * df
#          * mel
#          * nv
#          * vasc         
#          
#          
#          
#  check out this repo to create data tree: https://github.com/tsaideepak7/skin-cancer

# ****INITIALIZATION****

# In[ ]:


import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D,BatchNormalization,regularizers
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
from IPython.display import clear_output

import time
import datetime

import shutil


# **RESIZING TRAIN AND VALIDATION DATA**

# In[ ]:


print('number of train samples')
print(len(os.listdir('../input/datatree/datatree/train/nv')))
print(len(os.listdir('../input/datatree/datatree/train/mel')))
print(len(os.listdir('../input/datatree/datatree/train/bkl')))
print(len(os.listdir('../input/datatree/datatree/train/bcc')))
print(len(os.listdir('../input/datatree/datatree/train/akiec')))
print(len(os.listdir('../input/datatree/datatree/train/vasc')))
print(len(os.listdir('../input/datatree/datatree/train/df')))


# In[ ]:



train_datagen = ImageDataGenerator(rescale=1./255,rotation_range=1,zoom_range=0.1)

x_train = train_datagen.flow_from_directory(
    directory=r'../input/datatree/datatree/train/',
    batch_size=40,
    target_size=(75,100),
    class_mode="categorical",
    shuffle=True,
    seed=42
)

validation_datagen = ImageDataGenerator(rescale=1./255,rotation_range=1,zoom_range=0.1)

x_validation = validation_datagen.flow_from_directory(
    directory=r'../input/datatree/datatree/validation/',
    batch_size=31,
    target_size=(75,100),
    class_mode="categorical",
    shuffle=True,
    seed=42
)

test_datagen = ImageDataGenerator(rescale=1./255,rotation_range=1,zoom_range=0.1)

x_test = test_datagen.flow_from_directory(
    directory=r'../input/datatree/datatree/test/',
    batch_size=20,
    target_size=(75,100),
    class_mode="categorical",
    shuffle=False,
    seed=42
)


# In[ ]:


#ploting one image

p = x_train.next()
print((p[0][0]).shape)
(plt.imshow(p[0][0][:,:,:]) )



# CODE FOR LIVE PLOTTING LOSS AND ACCURACY

# In[ ]:



class PlotLearning(keras.callbacks.Callback):
    
    
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        self.acc = []
        self.val_acc = []
        self.fig = plt.figure()
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.acc.append(logs.get('acc'))
        self.val_acc.append(logs.get('val_acc'))
        self.i += 1
        f, (ax1, ax2) = plt.subplots(1, 2, sharex=True)
        
        clear_output(wait=True)
        
        ax1.set_yscale('log')
        ax1.plot(self.x, self.losses, label="train loss")
        ax1.plot(self.x, self.val_losses, label="val loss")
        ax1.legend()
        ax2.plot(self.x, self.acc, label="train acc")
        ax2.plot(self.x, self.val_acc, label="validation acc")
        ax2.legend()
        
        plt.show();
        
plot = PlotLearning()


# **CNN**
# 
# Building the model

# In[ ]:



classes_count= 7

model = Sequential()
model.add(Conv2D(32,(3, 3),activation='relu',input_shape=(75,100,3)))

model.add(Conv2D(32,(3, 3),activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.2))

model.add(Conv2D(64,(3, 3),activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.3))

model.add(Flatten())


model.add(Dense(128, activation='relu'))
 
model.add(Dropout(0.1))

model.add(Dense(classes_count,activation='softmax'))

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

model.summary()


# TRAINING THE MODEL

# In[ ]:



start=time.time()

cnn=model.fit_generator(x_train,steps_per_epoch=630,validation_data=x_validation,validation_steps=65,callbacks=[plot],epochs=30,verbose=2)

end=time.time()
print('training time: '+str(datetime.timedelta(seconds=(end-start))))


# In[ ]:



print('train accuracy     : '+str(cnn.history['acc'][-1]))
print('train loss         : '+str(cnn.history['loss'][-1]))
print('validation accuracy: '+str(cnn.history['val_acc'][-1]))
print('validation loss    : '+str(cnn.history['val_loss'][-1]))


# In[ ]:


name='model_'+str(cnn.history['acc'][-1])
model.save('model.h5')


# In[ ]:



predictions=model.predict_generator(x_test,steps=100,verbose=1)
print(predictions.shape)


# In[ ]:


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    
    


# In[ ]:


test_labels = x_test.classes
print(test_labels.shape)


# In[ ]:



import numpy as np
import matplotlib.pyplot as plt
import itertools


from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
cm = confusion_matrix(test_labels, predictions.argmax(axis=1))
cm_plot_labels = ['akiec', 'bcc', 'bkl', 'df', 'mel','nv', 'vasc']

plot_confusion_matrix(cm, cm_plot_labels, title='Confusion Matrix')


# In[ ]:


# Get the index of the class with the highest probability score
y_pred = np.argmax(predictions, axis=1)

# Get the labels of the test images.
y_true = x_test.classes

from sklearn.metrics import classification_report

# Generate a classification report
report = classification_report(y_true, y_pred, target_names=cm_plot_labels)

print(report)


# In[ ]:


label_frac_correct=np.diag(cm)/np.sum(cm,axis=1)
plt.bar(cm_plot_labels,label_frac_correct)
plt.xlabel('True Label')
plt.ylabel('Fraction classified correctly')


# **OVERFITTING**
# 
# The following methods can be used to prevent overfitting:
# * adding weights to classes
# * applying random crop to categories with less images
# * using a deeper predifined model like Mobilenet
