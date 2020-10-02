#!/usr/bin/env python
# coding: utf-8

# **This kernel is created to show the standard step-by-step process in handling image data. However, given the time limit of an hour, the kernel can only reach a low validation accuracy. Another way  of trainning the model from scratch is to run the script on a very powerful computer or using cloud computing. If you want to save time and computational power, you can also pre-process the data in the same manner and use [ImageNet pre-trained models](https://www.pyimagesearch.com/2017/03/20/imagenet-vggnet-resnet-inception-xception-keras/). Now let's begin and hope you enjoy it. **

# **Libraries that you need for image data preprocessing**

# In[ ]:


import os
import cv2 # image handling
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tqdm import tqdm

import seaborn as sns
import matplotlib.pyplot as plt 

import sklearn
from sklearn.cross_validation import train_test_split

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# **Examine the breeds, and we found out there are 120 breeds in total**

# In[ ]:


lables = pd.read_csv('../input/dog-breed-identification/labels.csv')
print (lables.head(5))
breed_count = lables['breed'].value_counts()
print (breed_count.head())
print (breed_count.shape)


# **One hot encoding the lables**

# In[ ]:


targets = pd.Series(lables['breed'])
one_hot = pd.get_dummies(targets, sparse = True)
one_hot_labels = np.asarray(one_hot)


# **Set image parameters to be used later, I'm using grayscale here so the number of channel is 1**

# In[ ]:


img_rows=128
img_cols=128
num_channel=1# 3 colour channes


# **Testing on a single image, first read in the image file in graysalce, then resize it**

# In[ ]:


img_1 = cv2.imread('../input/dog-breed-identification/train/000bec180eb18c7604dcecc8fe0dba07.jpg', 0)
plt.title('Original Image')
plt.imshow(img_1)


# In[ ]:


img_1_resize= cv2.resize(img_1, (img_rows, img_cols)) 
print (img_1_resize.shape)
plt.title('Resized Image')
plt.imshow(img_1_resize)


# **Now loop the proceedure through the train folder, and keep adding each new image data onto the existing data frame (x_feature) **

# In[ ]:


x_feature = []
y_feature = []

i = 0 # initialisation
for f, img in tqdm(lables.values): # f for format ,jpg
    train_img = cv2.imread('../input/dog-breed-identification/train/{}.jpg'.format(f),0)
    label = one_hot_labels[i]
    train_img_resize = cv2.resize(train_img, (img_rows, img_cols)) 
    x_feature.append(train_img_resize)
    y_feature.append(label)
    i += 1


# **The data frames need to be the form of arrays and normolised. Becuase I'm dealing with grayscale here, I needed to add the dimension at the end of the array else it keras would raise an exception**

# In[ ]:


x_train_data = np.array(x_feature, np.float32) / 255.   # /= 255 for normolisation
print (x_train_data.shape)
x_train_data = np.expand_dims(x_train_data, axis = 3)
print (x_train_data.shape)


# In[ ]:


y_train_data = np.array(y_feature, np.uint8)
print (y_train_data.shape)


# **Spliting the training and validation sets**

# In[ ]:


x_train, x_val, y_train, y_val = train_test_split(x_train_data, y_train_data, test_size=0.2, random_state=2)
print (x_train.shape)
print (x_val.shape)


# **After we have done the trainning data, now we are doing the test data, do the same thing to prepare the test data**

# In[ ]:


submission = pd.read_csv('../input/dog-breed-identification/sample_submission.csv')
test_img = submission['id']
print (test_img.head(5))


# In[ ]:


x_test_feature = []

i = 0 # initialisation
for f in tqdm(test_img.values): # f for format ,jpg
    img = cv2.imread('../input/dog-breed-identification/test/{}.jpg'.format(f), 0)
    img_resize = cv2.resize(img, (img_rows, img_cols)) 
    x_test_feature.append(img_resize)


# In[ ]:


x_test_data = np.array(x_test_feature, np.float32) / 255. 
print (x_test_data.shape)
x_test_data = np.expand_dims(x_test_data, axis = 3)
print (x_test_data.shape)


# **Now we have prepared: x_train, y_train, x_val, y_val and x_test. Time to build our CNN model. First import keras**

# In[ ]:


from keras.models import Sequential  # initial NN
from keras.layers import Dense, Dropout # construct each layer
from keras.layers import Convolution2D # swipe across the image by 1
from keras.layers import MaxPooling2D # swipe across by pool size
from keras.layers import Flatten


# **Initialising model**

# In[ ]:


model = Sequential()


# **I have a rather simple CNN here**
# 1. Convetional layer (detect features in image matrix)
# 2. Pooling layer (recongise features in different angle and/or size)
# 3. Convetional layer
# 4. Pooling laye
# 5. Flattening layer (flatten layers in array of imput)
# 6. Full connected layer (full connected ANN)
# 7. Output layer

# In[ ]:


# retifier ensure the non-linearity in the processing 
model.add(Convolution2D (filters = 64, kernel_size = (4,4),padding = 'Same', 
                         activation ='relu', input_shape = (img_rows, img_cols, num_channel))) 
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Convolution2D (filters = 64, kernel_size = (4,4),padding = 'Same', 
                         activation ='relu')) 
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten()) 
# fully connected ANN 
model.add(Dense(units = 120, activation = 'relu')) 
# output layer
model.add(Dense(units = 120, activation = 'softmax')) 


# **Compile the model**

# In[ ]:


model.compile(optimizer = 'adam' , loss = "categorical_crossentropy", metrics=["accuracy"]) 
model.summary()


# **Fit the model into data**

# In[ ]:


batch_size = 128 
nb_epochs = 2
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=nb_epochs,
                    verbose=2, 
                    validation_data=(x_val, y_val),
                    initial_epoch=0)


# **Plot the loss and accuracy curves for training and validation**

# In[ ]:


fig, ax = plt.subplots(2,1)
ax[0].plot(history.history['loss'], color='b', label="Training loss")
ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])
legend = ax[0].legend(loc='best', shadow=True)

ax[1].plot(history.history['acc'], color='b', label="Training accuracy")
ax[1].plot(history.history['val_acc'], color='r',label="Validation accuracy")
legend = ax[1].legend(loc='best', shadow=True)


# **Predict results**

# In[ ]:


results = model.predict(x_test_data)
prediction = pd.DataFrame(results)

# Set column names to those generated by the one-hot encoding earlier
col_names = one_hot.columns.values
prediction.columns = col_names
# Insert the column id from the sample_submission at the start of the data frame
prediction.insert(0, 'id', submission['id'])

submission = prediction
submission.to_csv('new_submission.csv', index=False)


# **Thank you for reading**
