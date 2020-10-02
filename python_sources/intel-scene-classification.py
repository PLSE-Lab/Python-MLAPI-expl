#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import gc
import numpy as np
import pandas as pd
from PIL import Image
from skimage.transform import resize
import matplotlib.pyplot as plt
import pickle
from shutil import copyfile
from tqdm import tqdm_notebook as tqdm
import seaborn as sns

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split
import cv2

from keras.utils import Sequence
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator


# In[2]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input/trainimage"))

# Any results you write to the current directory are saved as output.


# In[3]:


data_train = pd.read_csv('../input/trainimage/train.csv')
data_train.head()


# In[4]:


sns.countplot(y='label', data=data_train)


# In[5]:


X_train, X_test, y_train, y_test = train_test_split(data_train['image_name'].values, data_train['label'].values, 
                                                    test_size=0.3, random_state=42)
print(X_train.shape, X_test.shape)


# In[6]:


print(list(zip(X_train[:5], y_train[:5])))
print(list(zip(X_test[:5], y_test[:5])))


# In[7]:


get_ipython().run_cell_magic('time', '', "\n#Building the Folder Structure\nDATA_DIR = '../input/trainimage/train'\n\nif not os.path.isdir('data'):\n        os.mkdir('data')\n\n# Create the Training data\nif not os.path.isdir(os.path.join('data', 'train')):\n        os.mkdir(os.path.join('data', 'train'))\nroot = os.path.join('data', 'train')\nfor i in tqdm(range(len(X_train))):\n    label_dir = os.path.join(root, str(y_train[i]))\n    if not os.path.isdir(label_dir):\n        os.mkdir(label_dir)\n    src = os.path.join(DATA_DIR, X_train[i])\n    dst = os.path.join(label_dir, X_train[i])\n    copyfile(src, dst)\n\n# Create the Test data\nif not os.path.isdir(os.path.join('data', 'test')):\n        os.mkdir(os.path.join('data', 'test'))\nroot = os.path.join('data', 'test')\nfor i in tqdm(range(len(X_test))):\n    label_dir = os.path.join(root, str(y_test[i]))\n    if not os.path.isdir(label_dir):\n        os.mkdir(label_dir)\n    src = os.path.join(DATA_DIR, X_test[i])\n    dst = os.path.join(label_dir, X_test[i])\n    copyfile(src, dst)")


# In[8]:


list(os.walk('data'))[0]


# # Transfer Learning using ResNet-152

# In[9]:


# transforms.CenterCrop(224),
transformations = transforms.Compose([
    transforms.Resize(255),
    transforms.RandomRotation(0.3),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# In[10]:


train_set = datasets.ImageFolder("data/train", transform = transformations)
test_set = datasets.ImageFolder("data/test", transform = transformations)


# In[11]:


train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
test_loader = DataLoader(test_set, batch_size =32, shuffle=True)


# In[12]:


get_ipython().run_cell_magic('time', '', 'model = models.resnet152(pretrained=True)\nfor param in model.parameters():\n    param.requires_grad = False')


# In[13]:


classifier_input = model.fc.in_features; num_labels = 6;
classifier = nn.Sequential(nn.Linear(classifier_input, 1024),
                           nn.ReLU(),
                           nn.Dropout(p=0.5),
                           nn.Linear(1024, 512),
                           nn.ReLU(),
                           nn.Dropout(p=0.5),
                           nn.Linear(512, 128),
                           nn.ReLU(),
                           nn.Linear(128, num_labels),
                           nn.LogSoftmax(dim=1))

model.fc = classifier
# model.classifier = classifier


# In[14]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model.to(device)


# In[15]:


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters())
# optimizer = optim.Adam(model.classifier.parameters())


# In[16]:


epochs = 10
tr_list = []; val_list = []; acc = [];
for epoch in range(epochs):
    train_loss = 0
    val_loss = 0
    accuracy = 0
    
    # Training the model
    model.train()
    counter = 0
    for inputs, labels in train_loader:
        # Move to device
        inputs, labels = inputs.to(device), labels.to(device)
        # Clear optimizers
        optimizer.zero_grad()
        # Forward pass
        output = model.forward(inputs)
        # Loss
        loss = criterion(output, labels)
        # Calculate gradients (backpropogation)
        loss.backward()
        # Adjust parameters based on gradients
        optimizer.step()
        # Add the loss to the training set's rnning loss
        train_loss += loss.item()*inputs.size(0)
        
        # Print the progress of our training
        counter += 1
        if counter % 350 == 0:
            print(counter, "/", len(train_loader))
    
    print()
    # Evaluating the model
    model.eval()
    counter = 0
    # Tell torch not to calculate gradients
    with torch.no_grad():
        for inputs, labels in test_loader:
            # Move to device
            inputs, labels = inputs.to(device), labels.to(device)
            # Forward pass
            output = model.forward(inputs)
            # Calculate Loss
            valloss = criterion(output, labels)
            # Add loss to the validation set's running loss
            val_loss += valloss.item()*inputs.size(0)
            
            # Since our model outputs a LogSoftmax, find the real 
            # percentages by reversing the log function
            output = torch.exp(output)
            # Get the top class of the output
            top_p, top_class = output.topk(1, dim=1)
            # See how many of the classes were correct?
            equals = top_class == labels.view(*top_class.shape)
            # Calculate the mean (get the accuracy for this batch)
            # and add it to the running accuracy for this epoch
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            
            # Print the progress of our evaluation
            counter += 1
            if counter % 150 == 0:
                print(counter, "/", len(test_loader))
    
    # Get the average loss for the entire epoch
    train_loss = train_loss/len(train_loader.dataset)
    valid_loss = val_loss/len(test_loader.dataset)
    # Print out the information
    print('Accuracy: ', accuracy/len(test_loader))
    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(epoch, train_loss, valid_loss))
    tr_list.append(train_loss)
    val_list.append(valid_loss)
    acc.append(accuracy/len(test_loader))


# In[17]:


plt.plot(range(1,11), tr_list, 'b-', label='Train Error')
plt.plot(range(1,11), val_list, 'g-', label='Test Error')
plt.plot(range(1,11), acc, 'r-', label='Accuracy')
plt.legend(loc='best')
plt.show()


# In[18]:


print("Training Error:", max(tr_list), min(tr_list))
print("Test Error:", max(val_list), min(val_list))
print("Accuracy:", max(acc), min(acc))


# In[19]:


del model; gc.collect()


# # Shallow CNN from Scratch

# In[20]:


# DATA_DIR = '../input/trainimage/train'

# X_train = [f'{DATA_DIR}/{x}' for x in tqdm(X_train)]
# X_test = [f'{DATA_DIR}/{x}' for x in tqdm(X_test)]


# In[21]:


# class Data_Generator(Sequence):

#     def __init__(self, image_filenames, labels, batch_size):
#         self.image_filenames, self.labels = image_filenames, labels
#         self.batch_size = batch_size

#     def __len__(self):
#         return int(np.ceil(len(self.image_filenames) / float(self.batch_size)))

#     def __getitem__(self, idx):
#         batch_x = self.image_filenames[idx * self.batch_size:(idx + 1) * self.batch_size]
#         batch_y = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]

#         return np.array([resize(plt.imread(file_name), (img_size, img_size))/255.0 for file_name in batch_x]), np.array(batch_y)


# In[22]:


# img_size = 230; batch_size = 16; epochs = 10; num_classes = 6;

# train_generator = Data_Generator(X_train, y_train, batch_size)
# test_generator = Data_Generator(X_test, y_test, batch_size)


# In[23]:


# x_train = np.empty((len(X_train), img_size, img_size, 3), dtype='float32')
# x_test = np.empty((len(X_test), img_size, img_size, 3), dtype='float32')

# for i in tqdm(range(1, len(X_train))):
#     img = plt.imread(f'{DATA_DIR}/{X_train[i]}')
#     x_train[i] = resize(img, (img_size, img_size), anti_aliasing=True)
    
# for i in tqdm(range(1, len(X_test))):
#     img = plt.imread(f'{DATA_DIR}/{X_test[i]}')
#     x_test[i] = resize(img, (img_size, img_size), anti_aliasing=True)

# x_train /= 255
# x_test /= 255

# print(x_train.shape, x_test.shape)


# In[28]:


img_size = 230; batch_size = 32; epochs = 15; num_classes = 6;

train_datagen = ImageDataGenerator(
    samplewise_center=True,
    samplewise_std_normalization=True,
    zca_whitening=False,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    brightness_range=[0.5, 1.5],
    rescale=1./255,
    shear_range=0.2,
    zoom_range=[0.9, 1.25],
    horizontal_flip=True,
    vertical_flip=True,
    channel_shift_range=0.2,
    fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        'data/train',
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='sparse')

test_generator = test_datagen.flow_from_directory(
        'data/test',
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='sparse')


# In[29]:


import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, GaussianNoise, BatchNormalization
# from keras.applications import InceptionV3, VGG16, ResNet50

# base_net = VGG16(include_top=False,
#                  input_shape = (img_size,img_size,3),
#                  weights = 'imagenet')
# base_net.trainable = False

model = Sequential()
# model.add(base_net)
# model.add(GaussianNoise(0.3))
model.add(Conv2D(64, kernel_size=(7, 7), activation='relu', padding='same', input_shape=(img_size,img_size,3)))
model.add(BatchNormalization())
model.add(Conv2D(64, (7, 7), activation='relu', padding='same', kernel_regularizer=keras.regularizers.l2(0.1)))
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

model.add(Conv2D(64, (5, 5), activation='relu', padding='same', kernel_regularizer=keras.regularizers.l2(0.1)))
model.add(BatchNormalization())
model.add(Conv2D(64, (5, 5), activation='relu', padding='same', kernel_regularizer=keras.regularizers.l2(0.1)))
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

model.add(Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=keras.regularizers.l2(0.1)))
model.add(BatchNormalization())
model.add(Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=keras.regularizers.l2(0.1)))
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

# model.add(Conv2D(128, (7, 7), activation='relu', padding='same', kernel_regularizer=keras.regularizers.l2(0.01)))
# model.add(BatchNormalization())
# model.add(Conv2D(64, (7, 7), activation='relu', padding='same', kernel_regularizer=keras.regularizers.l2(0.01)))
# model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

# model.add(Conv2D(128, (5, 5), activation='relu', padding='same', kernel_regularizer=keras.regularizers.l2(0.01)))
# model.add(BatchNormalization())
# model.add(Conv2D(64, (5, 5), activation='relu', padding='same', kernel_regularizer=keras.regularizers.l2(0.01)))
# model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

# model.add(Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=keras.regularizers.l2(0.01)))
# model.add(BatchNormalization())
# model.add(Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=keras.regularizers.l2(0.01)))
# model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

model.add(Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=keras.regularizers.l2(0.1)))
model.add(BatchNormalization())
model.add(Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=keras.regularizers.l2(0.1)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=keras.regularizers.l2(0.1)))
model.add(BatchNormalization())
model.add(Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=keras.regularizers.l2(0.1)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=keras.regularizers.l2(0.1)))
model.add(BatchNormalization())
model.add(Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=keras.regularizers.l2(0.1)))
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

model.add(Flatten())
model.add(Dense(2048, activation='relu', kernel_regularizer=keras.regularizers.l2(0.1)))
model.add(BatchNormalization())
model.add(Dropout(0.7))

model.add(Dense(1024, activation='relu', kernel_regularizer=keras.regularizers.l2(0.1)))
model.add(BatchNormalization())
model.add(Dropout(0.7))

model.add(Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(0.1)))
model.add(BatchNormalization())
model.add(Dropout(0.7))

model.add(Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(0.1)))
# model.add(Dropout(0.5))

model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.sparse_categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])
# print(model.summary())

history = model.fit_generator(generator=train_generator,
                    steps_per_epoch=(len(X_train) // batch_size),
                    epochs=epochs,
                    verbose=1,
                    validation_data=test_generator,
                    validation_steps=(len(X_test) // batch_size),
#                     use_multiprocessing=True,
#                     workers=16,
                    max_queue_size=32)
#                     shuffle=True)

# history = model.fit(x_train, y_train,
#           batch_size=batch_size,
#           epochs=epochs,
#           verbose=1,
#           validation_data=(x_test, y_test))

# score = model.evaluate(x_test, y_test, verbose=0)
# print('Test loss:', score[0])
# print('Test accuracy:', score[1])


# In[30]:


plt.plot(range(1,epochs+1), history.history['loss'], 'b-', label='Train Error')
plt.plot(range(1,epochs+1), history.history['val_loss'], 'g-', label='Test Error')
plt.plot(range(1,epochs+1), history.history['val_acc'], 'r-', label='Test Accuracy')
plt.legend(loc='best')
plt.show()


# In[31]:


print("Training Error:", max(history.history['loss']), min(history.history['loss']))
print("Test Error:", max(history.history['val_loss']), min(history.history['val_loss']))
print("Test Accuracy:", max(history.history['val_acc']), min(history.history['val_acc']))


# In[ ]:


del model; gc.collect()


# # Conclusion :-
# - The pretrained model is performing much better than the models from scratch.
# - Model with Data Augmentation is performing better than the Model without Augmentation.
