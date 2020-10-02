#!/usr/bin/env python
# coding: utf-8

# # **[Beginner] CNN-based Honey Bee Pollen Binary Classifier**
# ---
# **By: Peter Sunga** <br/>
# This is a simple CNN model used to classify whether or not a bee is carrying pollen.   
# This model it able to obtain over 90% accuracy with 571 training images on 143 test images.     
#  
# *I will try to improve accuracy in the future with image augmentation techniques*
# 
# # Table of Contents
# 1. [Part 1: Organize the Dat](#Part 1: Organize the Data)
# 2. [Part 2: Build the CNN model](#Part 2: Build the CNN model)
# 3. [Part 3: Train the CNN mode](#Part 3: Train the CNN model)
# 4. [Part 4: Analyze the CNN mode](#Part 4: Analyze the CNN mode)
# 5. [Part 5: Retrain the Model with all the Training data](#Part 5: Retrain the Model with all the Training data)
# 5. [Part 6: Test the Model](#Part 6: Test the Model)

# <a name="Part 1: Organize the Data"></a>
# ## Part 1: Organize the Data    
# 1. Import the images into a dataset
# 2. Split the dataset into a training, validation, and testing sets

# In[ ]:


import glob, os 
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
path="../input/pollendataset/PollenDataset/images/"
imlist= glob.glob(os.path.join(path, '*.jpg'))


# In[ ]:


def dataset(file_list,size=(300,180),flattened=False):
	data = []
	for i, file in enumerate(file_list):
		image = io.imread(file)
		image = transform.resize(image, size, mode='constant')
		if flattened:
			image = image.flatten()

		data.append(image)

	labels = [1 if f.split("/")[-1][0] == 'P' else 0 for f in file_list]

	return np.array(data), np.array(labels)


# * Load the dataset into the image tensor X
# * y will contain the labels for X

# In[ ]:


# Load the dataset (may take a few seconds)
X,y=dataset(imlist)


# In[ ]:


# X has the following structure: X[imageid, y,x,channel]
print('The length of X: ',len(X))  # data
print('The shape of X: ',X.shape)  # target
print('The shape of Y', y.shape)


# As displayed above there are **714**  tensor images in dataset X

# In[ ]:


print(X[1,:,: ,0]) #let's look into the pixel data for one of the images


# We do not need to scale the pixel values as they are already between 0-1
# ### Now let's create the training, partial training,  valdiation, and testing sets </br>
# * We will split the dataset X and label matrix y into three partitions to achieve this
# * train_test_split module function from sklearn can automatically do this for us
# * Training dataset will be used to train the final model, partial training dataset is only used in conjunction with the validation dataset to ensure we do not overfit. (Determine ideal number of epoches)

# In[ ]:


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
	X, y, test_size=0.20)

partial_x_train, validation_x_train, partial_y_train, validation_y_train = train_test_split(
	x_train, y_train, test_size=0.15)


# As a sanity check, let's check the length of each set...

# In[ ]:


print('The size of the training set: ',len(x_train))
print('The size of the partial training set: ',len(partial_x_train))
print('The size of the validation training set: ',len(validation_x_train))
print('The size of the testing set: ',len(x_test))


# Now that we have the training, partial training, validation, and testing set. It's time to design the CNN model...
# <a name="Part 2: Build the CNN model"></a>
# ## Part 2: Build the CNN model

# In[ ]:


from keras import layers
from keras import models
from keras import optimizers

model = models.Sequential()
model.add(layers.Conv2D(64,(3,3), activation='relu', input_shape=(300,180,3)))  #input shape must be the match the input image tensor shape
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Conv2D(64,(3,3), activation='relu'))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Conv2D(128,(3,3), activation='relu'))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Conv2D(128,(3,3), activation='relu'))
model.add(layers.Conv2D(128,(3,3), activation='relu'))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Conv2D(256,(3,3), activation='relu'))
model.add(layers.Conv2D(256,(3,3), activation='relu'))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation = 'relu'))
model.add(layers.Dense(1, activation = 'sigmoid'))

model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4), metrics=['acc'])


# Now we are ready to begin the training!
# <a name="Part 3: Train the CNN model"></a>
# ## Part 3: Train the CNN model

# In[ ]:


history = model.fit(
    partial_x_train, 
    partial_y_train,
    validation_data=(validation_x_train, validation_y_train),
    epochs=100, 
    batch_size=15, 
    verbose =2) #hides some information while training


# <a name="Part 4: Analyze the CNN mode"></a>
# ## Part 4: Analyze the CNN mode

# In[ ]:


def smooth_curve(points, factor=0.8): #this function will make our plots more smooth
	smoothed_points = []
	for point in points:
		if smoothed_points:
			previous = smoothed_points[-1]
			smoothed_points.append(previous*factor+point*(1-factor))
		else:
			smoothed_points.append(point)
	return smoothed_points


# In[ ]:


acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']


# In[ ]:


epochs = range(1, len(acc)+1)
plt.plot(epochs, smooth_curve(acc), 'bo', label='Training acc')
plt.plot(epochs, smooth_curve(val_acc), 'r-', label='Training acc')
plt.legend()
plt.title('Training and Validation Acc')
plt.figure()

plt.plot(epochs, smooth_curve(loss), 'bo', label='Training loss')
plt.plot(epochs, smooth_curve(val_loss), 'r-', label='Training acc')
plt.legend()
plt.title('Training and Validation loss')
plt.show()


# Based on the plots above, it seems we start to overfit around 60 epochs!  
# We will use around 60 epochs for our final training model.   
# **Result may vary**

# <a name="Part 5: Retrain the Model with all the Training data"></a>
# ## Part 5: Retrain the Model with all the Training data

# In[ ]:


model1 = models.Sequential()
model1.add(layers.Conv2D(64,(3,3), activation='relu', input_shape=(300,180,3)))  #input shape must be the match the input image tensor shape
model1.add(layers.MaxPooling2D(2,2))
model1.add(layers.Conv2D(64,(3,3), activation='relu'))
model1.add(layers.MaxPooling2D(2,2))
model1.add(layers.Conv2D(128,(3,3), activation='relu'))
model1.add(layers.MaxPooling2D(2,2))
model1.add(layers.Conv2D(128,(3,3), activation='relu'))
model1.add(layers.Conv2D(128,(3,3), activation='relu'))
model1.add(layers.MaxPooling2D(2,2))
model1.add(layers.Conv2D(256,(3,3), activation='relu'))
model1.add(layers.Conv2D(256,(3,3), activation='relu'))
model1.add(layers.MaxPooling2D(2,2))
model1.add(layers.Flatten())
model1.add(layers.Dropout(0.5))
model1.add(layers.Dense(512, activation = 'relu'))
model1.add(layers.Dense(1, activation = 'sigmoid'))

model1.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4), metrics=['acc'])

history1 = model1.fit(
    x_train, 
    y_train,
    epochs=65, 
    batch_size=15, 
    verbose =2)


# <a name="Part 6: Test the Model"></a>
# ## Part 6: Test the Model

# In[ ]:


test_loss, test_acc = model1.evaluate(x_test, y_test, steps=10)
print('The final test accuracy: ',test_acc)

