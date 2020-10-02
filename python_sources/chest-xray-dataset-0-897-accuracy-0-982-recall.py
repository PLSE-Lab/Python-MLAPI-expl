#!/usr/bin/env python
# coding: utf-8

# ## Chest X-ray Images Dataset
# 
# Ben Sturm <br/>
# 11/22/19
# 
# The Chest X-ray Images Dataset is avaliable [here](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia).  It consists of 5,863 Chest X-ray images consisting of two classes (Pneumonia/Normal).  The dataset is broken up into a train/validation/test sets.  My task is to develop a classifier that takes in the images and outputs 1 (Pneumonia) or 0 (Normal).
# 
# The problem with this dataset is that the validation set is very small (18 examples total), which isn't enough to use to validate the model during training.  So, what I chose to do is split the training data using a 90/10 split of training/validation, which I will implement in this notebook.  I also plan to use data augmentation when training my model.  

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
#    for filename in filenames:
#        print(os.path.join(dirname, filename))
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


os.listdir('../input/chest-xray-pneumonia/chest_xray/chest_xray')


# ### Copying images to training, validation, and test directories
# 
# Here I'm specifying the original dataset directory and the base directory where the postprocessed data will be saved.  The base dictory is located in the working directory of the kernel.  

# In[ ]:


original_dataset_dir = '../input/chest-xray-pneumonia/chest_xray/chest_xray'

base_dir ='./chest_xray_postprocessed'
os.mkdir(base_dir)


# In[ ]:


train_dir = os.path.join(base_dir, 'train')
os.mkdir(train_dir)
val_dir = os.path.join(base_dir, 'val')
os.mkdir(val_dir)
test_dir = os.path.join(base_dir, 'test')
os.mkdir(test_dir)

train_pneumonia_dir = os.path.join(train_dir, 'pneumonia')
os.mkdir(train_pneumonia_dir)

train_normal_dir = os.path.join(train_dir, 'normal')
os.mkdir(train_normal_dir)

val_pneumonia_dir = os.path.join(val_dir, 'pneumonia')
os.mkdir(val_pneumonia_dir)

val_normal_dir = os.path.join(val_dir, 'normal')
os.mkdir(val_normal_dir)

test_pneumonia_dir = os.path.join(test_dir, 'pneumonia')
os.mkdir(test_pneumonia_dir)

test_normal_dir = os.path.join(test_dir, 'normal')
os.mkdir(test_normal_dir)


# I'm going to do a 90/10 split of training and validation.  This means of the positive class (Pneumonia), we'll have 3489 Train and 387 Validation.  Of the negative class (Normal), we'll have 1208 Train and 134 Validation.

# In[ ]:


import glob

paths_pneumonia = glob.glob(original_dataset_dir+'/train/PNEUMONIA/*')
paths_normal = glob.glob(original_dataset_dir+'/train/NORMAL/*')


# In[ ]:


print(len(paths_pneumonia))
print(len(paths_normal))


# While we're at it, let's quickly display a barplot to show this distribution.  This is obviously an unbalanced dataset, weighted heavily to the chest X-rays with pneumonia.

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8,6))
sns.barplot(['pneumonia','normal'],[len(paths_pneumonia),len(paths_normal)])
plt.title('Class distribution')


# In[ ]:


fname_pneumonia = [x.split('/')[-1] for x in paths_pneumonia]
fname_pneumonia[:2]


# In[ ]:


fname_normal = [x.split('/')[-1] for x in paths_normal]
fname_normal[:2]


# Much of this code was stolen from [Deep Learning with Python](https://www.manning.com/books/deep-learning-with-python).

# In[ ]:


import shutil

def train_val_data_split(pneumonia_list, normal_list, validation_split=0.1):
    
    n_pneumonia = len(pneumonia_list)
    n_normal = len(normal_list)
    
    shuffle_idx_pneumonia = [x for x in range(n_pneumonia)]
    shuffle_idx_normal = [x for x in range(n_normal)]
    
    num_train_pneumonia = int((1-validation_split)*n_pneumonia)
    num_val_pneumonia = n_pneumonia - num_train_pneumonia
    
    num_train_normal = int((1-validation_split)*n_normal)
    num_val_normal = n_normal - num_train_normal
    
    for i in range(num_train_pneumonia):
        src = os.path.join(original_dataset_dir+'/train/PNEUMONIA/', pneumonia_list[i])
        dst = os.path.join(train_pneumonia_dir, pneumonia_list[i])
        shutil.copyfile(src,dst)
        
    for i in range(num_train_pneumonia,n_pneumonia):
        src = os.path.join(original_dataset_dir+'/train/PNEUMONIA/', pneumonia_list[i])
        dst = os.path.join(val_pneumonia_dir, pneumonia_list[i])
        shutil.copyfile(src,dst)
        
    for i in range(num_train_normal):
        src = os.path.join(original_dataset_dir+'/train/NORMAL/', normal_list[i])
        dst = os.path.join(train_normal_dir, normal_list[i])
        shutil.copyfile(src,dst)
        
    for i in range(num_train_normal,n_normal):
        src = os.path.join(original_dataset_dir+'/train/NORMAL/', normal_list[i])
        dst = os.path.join(val_normal_dir, normal_list[i])
        shutil.copyfile(src,dst)


# In[ ]:


train_val_data_split(fname_pneumonia, fname_normal, validation_split=0.1)


# In[ ]:


def test_data_copy():
    
    test_data_pneumonia = glob.glob(original_dataset_dir+'/test/PNEUMONIA/*')
    test_data_normal = glob.glob(original_dataset_dir+'/test/NORMAL/*')
    
    test_fname_pneumonia = [x.split('/')[-1] for x in test_data_pneumonia]
    test_fname_normal = [x.split('/')[-1] for x in test_data_normal]
    
    n_pneumonia = len(test_fname_pneumonia)
    n_normal = len(test_fname_normal)
    
    for i in range(n_pneumonia):
        src = os.path.join(original_dataset_dir+'/test/PNEUMONIA/', test_fname_pneumonia[i])
        dst = os.path.join(test_pneumonia_dir, test_fname_pneumonia[i])
        shutil.copyfile(src,dst)
        
    for i in range(n_normal):
        src = os.path.join(original_dataset_dir+'/test/NORMAL/', test_fname_normal[i])
        dst = os.path.join(test_normal_dir, test_fname_normal[i])
        shutil.copyfile(src,dst)


# In[ ]:


test_data_copy()


# Now I have successfully done the 90/10 split of the training/validation data.  I also copied the test data.  All of this data now sits at './chest_xray_postprocessed'.  I will need to delete this data before committing those notebook, since Kaggle has a limit of 500 output files.
# 
# Okay, let's now work on building a model.

# ### Model building using Keras

# In[ ]:


import keras
keras.__version__


# In[ ]:


#os.listdir('./chest_xray_postprocessed/train/normal/')


# In[ ]:


from PIL import Image
import matplotlib.pyplot as plt

fname = './chest_xray_postprocessed/train/normal/IM-0382-0001.jpeg'
image = Image.open(fname).convert("L")
arr = np.asarray(image)
plt.imshow(arr, cmap='gray', vmin=0, vmax=255)
plt.title('Normal')
plt.show()


# In[ ]:


#os.listdir('./chest_xray_postprocessed/train/pneumonia/')


# In[ ]:


fname = './chest_xray_postprocessed/train/pneumonia/person1000_bacteria_2931.jpeg'
image = Image.open(fname).convert("L")
arr = np.asarray(image)
plt.imshow(arr, cmap='gray', vmin=0, vmax=255)
plt.title('Pneumonia')
plt.show()


# In[ ]:


print(arr.shape)
print(np.max(arr))


# Since the images are grayscale images, which means they consist of 1 channel, I'm going to use a simple convnet which I've stolen from the [Deep Learning with Python](https://www.manning.com/books/deep-learning-with-python) book.

# In[ ]:


from keras import layers
from keras import models
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras import metrics


# In[ ]:


model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128,128,1)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4),metrics=['acc',metrics.Recall()])


# In[ ]:


model.summary()


# In[ ]:


# Here I'm specifying values for data augmentation during training
train_datagen = ImageDataGenerator(rescale=1./255, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')

# Validation and test data should not be augmented
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory('./chest_xray_postprocessed/train', target_size=(128, 128), batch_size=16, color_mode='grayscale', class_mode='binary')

val_generator = val_datagen.flow_from_directory('./chest_xray_postprocessed/val', target_size=(128, 128), batch_size=16, color_mode='grayscale', class_mode='binary')

test_generator = test_datagen.flow_from_directory('./chest_xray_postprocessed/test', target_size=(128, 128), batch_size=16, color_mode='grayscale', class_mode='binary', shuffle=False)


# In[ ]:


history = model.fit_generator(train_generator, steps_per_epoch = 326, epochs=30, validation_data = val_generator, validation_steps=1)


# In[ ]:


# Saving the history data as a dict file

import pickle 
    
with open('./trainHistoryDict', 'wb') as file_pi:
    pickle.dump(history.history, file_pi)


# ### Model evaluation

# In[ ]:


# Displaying curves of loss and accuracy during training

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc)+1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()


# In[ ]:


model.save_weights('CNN_part2_model.h5')


# In[ ]:


model.metrics_names


# In[ ]:


# evaluate the model
scores = model.evaluate_generator(test_generator)
print("{}: {:.3f}".format(model.metrics_names[1], scores[1]))
print("{}: {:.3f}".format(model.metrics_names[2], scores[2]))


# Using predict_generator to obtain probabilities of each test example with the probability representing the likelihood of the image example belonging to the pneumonia class (class=1).

# In[ ]:


y_test_prob = model.predict_generator(test_generator, verbose=True)


# Since y_test_prob represents probabilities, we will use the standard decision boundary that a probability>0.5 belongs to the positive class (pneumonia class).

# In[ ]:


index_array = y_test_prob>0.5
index_array_1d = np.reshape(index_array,-1)
y_test_pred = np.zeros(len(y_test_prob), dtype=int)
y_test_pred[index_array_1d] = 1 #y_test_pred is a binary value (0 or 1) 


# In[ ]:


y_test_pred[-10:]


# In[ ]:


y_test_true = test_generator.classes # the test_generator classes attribute provides an array of classes for for each instance


# In[ ]:


type(y_test_true)


# In[ ]:


from sklearn.metrics import roc_auc_score,roc_curve,accuracy_score,recall_score,confusion_matrix,f1_score,precision_score


# In[ ]:


print("Accuracy on the test set: {:.3f}".format(accuracy_score(y_test_true,y_test_pred)))


# In[ ]:


print("Recall on the test set: {:.3f}".format(recall_score(y_test_true,y_test_pred)))


# In[ ]:


print("Precision on the test set: {:.3f}".format(precision_score(y_test_true,y_test_pred)))


# In[ ]:


fpr, tpr, thresholds = roc_curve(y_test_true,y_test_prob)
plt.plot(fpr, tpr, label="ROC Curve")
plt.xlabel("FPR")
plt.ylabel("TPR (recall)")
plt.legend(loc=4)


# In[ ]:


print("ROC AUC score on the test set: {:.3f}".format(roc_auc_score(y_test_true,y_test_prob)))


# In[ ]:


confusion_matrix(y_test_true, y_test_pred)


# In[ ]:


print("f1 score on the test set: {:.3f}".format(f1_score(y_test_true,y_test_pred)))


# In[ ]:


shutil.rmtree('./chest_xray_postprocessed')


# In[ ]:




