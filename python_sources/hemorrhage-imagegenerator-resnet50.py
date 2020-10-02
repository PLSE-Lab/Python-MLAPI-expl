#!/usr/bin/env python
# coding: utf-8

# # Hemorrhage prediction with Image Generator and ResNet50
# This notebook is a starter code for all beginners and easy to understand. Used is a image generator based on the template <br>
# https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
# 
# Furthermore i read the following notebooks:
# * https://www.kaggle.com/gzuidhof/full-preprocessing-tutorial
# * https://www.kaggle.com/allunia/rsna-ih-detection-eda-baseline
# 
# The model is based on ResNet50 and runs on GPU.

# # Load Libraries

# In[ ]:


import numpy as np
import pandas as pd
import random
import os
import matplotlib.pyplot as plt


# In[ ]:


import pydicom as dicom
import cv2


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer


# In[ ]:


from keras.utils import to_categorical, Sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, Activation
from keras.optimizers import RMSprop,Adam
from keras.applications import VGG19, VGG16, ResNet50


# In[ ]:


import warnings
warnings.filterwarnings("ignore")


# # Input Path
# Define the path for the subfolders with the data.

# In[ ]:


path_in = "../input/rsna-intracranial-hemorrhage-detection/rsna-intracranial-hemorrhage-detection/"
os.listdir(path_in)


# Define the sub paths with images

# In[ ]:


path_train_img = path_in + 'stage_2_train'
path_test_img = path_in + 'stage_2_test'


# Path to the pretrained data set.

# In[ ]:


path_models = '../input/models' 
os.listdir(path_models)


# # Define some parameters

# In[ ]:


q_size = 200
img_channel = 3
num_classes = 6


# # Read image name

# In[ ]:


list_train_img = os.listdir(path_train_img)
list_test_img = os.listdir(path_test_img)


# In[ ]:


list_train_img[0:2]


# In[ ]:


print('number of train images: ', len(list_train_img))
print('number of test images: ', len(list_test_img))


# # Read input data

# In[ ]:


train_data = pd.read_csv(path_in + 'stage_2_train.csv')
sub_org = pd.read_csv(path_in + 'stage_2_sample_submission.csv')


# In[ ]:


train_data.head()


# # Modify the input data

# In[ ]:


train_data['sub_type'] = train_data['ID'].str.split("_", n = 3, expand = True)[2]
train_data['PatientID'] = train_data['ID'].str.split("_", n = 3, expand = True)[1]
sub_org['sub_type'] = sub_org['ID'].str.split("_", n = 3, expand = True)[2]
sub_org['PatientID'] = sub_org['ID'].str.split("_", n = 3, expand = True)[1]


# # Group the subtypes

# In[ ]:


group_type = train_data.groupby('sub_type').sum()
fig = plt.figure(figsize=(9, 5))
ax = fig.add_subplot(111)
ax.bar(group_type.index, group_type['Label'])
ax.set_xticklabels(group_type.index, rotation=45)
plt.grid()
plt.show()


# In[ ]:


train_data_pivot = train_data[['Label', 'PatientID', 'sub_type']].drop_duplicates().pivot(index='PatientID', columns='sub_type', values='Label')
test_data_pivot = sub_org[['Label', 'PatientID', 'sub_type']].drop_duplicates().pivot(index='PatientID', columns='sub_type', values='Label')


# # Select a subset of the input data for training

# In[ ]:


percentage = 0.25
num_train_img = int(percentage*len(train_data_pivot.index))
num_test_img = len(test_data_pivot.index)
print('num_train_data:', len(list_train_img), num_train_img)
print('num_test_data:', len(list_test_img))
list_train_img = list(train_data_pivot.index)
list_test_img = list(test_data_pivot.index)
random_train_img = random.sample(list_train_img, num_train_img)


# In[ ]:


y_train_org = train_data_pivot.loc[random_train_img]


# # Split the input data for train and val

# In[ ]:


y_train, y_val = train_test_split(y_train_org, test_size=0.3)
y_test = test_data_pivot


# # Calculate class weights

# In[ ]:


class_weight = dict(zip(range(0, num_classes), y_train.sum()/y_train.sum().sum()))


# In[ ]:


class_weight


# # Build the Data Generator

# In[ ]:


#https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
class DataGenerator(Sequence):
    def __init__(self, path, list_IDs, labels, batch_size, img_size, img_channel, num_classes, shuffle=True):
        self.path = path
        self.list_IDs = list_IDs
        self.labels = labels
        self.batch_size = batch_size
        self.img_size = img_size
        self.img_channel = img_channel
        self.num_classes = num_classes
        self.shuffle = shuffle
        self.on_epoch_end()
     
    
    def __len__(self):
        return int(np.floor(len(self.list_IDs)/self.batch_size))
    
    
    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        X, y = self.__data_generation(list_IDs_temp)
        return X, y
    
    
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
    
    
    def rescale_pixelarray(self, dataset):
        image = dataset.pixel_array
        rescaled_image = image * dataset.RescaleSlope + dataset.RescaleIntercept
        rescaled_image[rescaled_image < -1024] = -1024
        return rescaled_image

    
    def __data_generation(self, list_IDs_temp):
        X = np.empty((self.batch_size, self.img_size, self.img_size))
        y = np.empty((self.batch_size, self.num_classes), dtype=int)
        for i, ID in enumerate(list_IDs_temp):
            data_file = dicom.dcmread(self.path+'/ID_'+ID+'.dcm')
            img = self.rescale_pixelarray(data_file)
            img = cv2.resize(img, (self.img_size, self.img_size))
            X[i, ] = img
            y[i, ] = self.labels.loc[ID]
        X = np.repeat(X[..., np.newaxis], 3, -1)
        X = X.astype('float32')
        X -= X.mean(axis=0)
        std = X.std(axis=0)
        X /= X.std(axis=0)
        return X, y


# # Load the pretrained model

# In[ ]:


conv_base = ResNet50(weights='../input/models/model_weights_resnet.h5',
                     include_top=False,
                     input_shape=(q_size, q_size, img_channel))
conv_base.trainable = True


# # Define train and validation data via Data Generator

# In[ ]:


batch_size = 32
train_generator = DataGenerator(path_train_img, list(y_train.index), y_train,
                                batch_size, q_size, img_channel, num_classes)
val_generator = DataGenerator(path_train_img, list(y_val.index), y_val,
                                batch_size, q_size, img_channel, num_classes)


# # Define the model

# In[ ]:


model = Sequential()
model.add(conv_base)
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(6, activation='sigmoid'))


# # Compile the model

# In[ ]:


model.compile(optimizer = RMSprop(lr=1e-5),
              loss='binary_crossentropy',
              metrics=['binary_accuracy'])


# In[ ]:


model.summary()


# In[ ]:


epochs = 5


# # Fit the model with the fit_generator method

# In[ ]:


history = model.fit_generator(generator=train_generator,
                              validation_data=val_generator,
                              epochs = epochs,
                              class_weight = class_weight,
                              workers=4)


# # Plot the loss values

# In[ ]:


loss = history.history['loss']
loss_val = history.history['val_loss']
epochs = range(1, len(loss)+1)
plt.plot(epochs, loss, 'bo', label='loss_train')
plt.plot(epochs, loss_val, 'b', label='loss_val')
plt.title('value of the loss function')
plt.xlabel('epochs')
plt.ylabel('value of the loss functio')
plt.legend()
plt.grid()
plt.show()


# # Plot the accuracy values

# In[ ]:


acc = history.history['binary_accuracy']
acc_val = history.history['val_binary_accuracy']
epochs = range(1, len(loss)+1)
plt.plot(epochs, acc, 'bo', label='accuracy_train')
plt.plot(epochs, acc_val, 'b', label='accuracy_val')
plt.title('accuracy')
plt.xlabel('epochs')
plt.ylabel('value of accuracy')
plt.legend()
plt.grid()
plt.show()


# # Define the test data via Data Generator

# In[ ]:


batch_size = 16
test_generator = DataGenerator(path_test_img, list(y_test.index), y_test,
                                batch_size, q_size, img_channel, num_classes, shuffle=False)


# # Predict the test images with the generator class

# In[ ]:


predict = model.predict_generator(test_generator, verbose=1)


# In[ ]:


assert(len(predict) == len(test_data_pivot))


# # Prepare the prediction data by the export format

# In[ ]:


submission = pd.DataFrame(predict, columns=y_train_org.columns)
submission.insert(loc=0, column='PatientID', value=test_data_pivot.index)
submission.index=submission['PatientID']
submission = submission.drop(['PatientID'], axis=1)


# In[ ]:


submission = submission.stack().reset_index()
submission = submission.rename(columns={0: 'Label'})


# In[ ]:


submission.insert(loc=0, column='ID', value='ID_'+submission['PatientID'].astype(str)+'_'+submission['sub_type'].astype(str))
submission = submission.drop(['PatientID', 'sub_type'], axis=1)


# In[ ]:


submission.index = submission['ID']
submission = submission.reindex(sub_org['ID'])
submission.index = range(len(submission))


# # Export the prediction data

# In[ ]:


submission.to_csv('submission.csv', header = True, index=False)


# In[ ]:


submission


# ### Extensions
# * Study the medical knowledge to improve the model.
# * Test and improve the model.

# In[ ]:


del model

