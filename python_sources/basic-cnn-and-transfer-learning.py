#!/usr/bin/env python
# coding: utf-8

# # Implementing CNN for a basic image classification
# 
# The main goal of this notebook is to give an introduction to utilizing Keras library for building a basic Convolutional Neural Network. As a secondary goal I wished to highlight transfer learning using ImageNet as an example for improving a neural networks performance. Image processing, splitting testing/training data, and basic plotting of model performance will also be highlighted here.
# 
# **First we will import at the necessary libararies:**

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2 #cv2 for image processing
import random
import os
import numpy as np #NumPy for array manipulation
import matplotlib.pyplot as plt #Matplotlib for visualizing the performance of the models
import matplotlib.image as mpimg
get_ipython().run_line_magic('matplotlib', 'inline')

import keras #Keras is a library for building neural networks on top of TensorFlow
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array, load_img
from keras.applications import InceptionResNetV2
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras import layers, models, optimizers

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# **Capture all the image file paths**
# 
# Now that all the various libraries have been imported, we want to capture all the image data in this dataset. We will use the directory path to capture the images of people wearing masks and the images of those not wearing masks. Once we have the file path we use iteration to create a list of all the images in each file path.

# In[ ]:


mask_dir = '../input/face-mask-detection-data/with_mask'
no_mask_dir = '../input/face-mask-detection-data/without_mask'
mask_img = [f'{mask_dir}/{i}' for i in os.listdir(mask_dir)]
no_mask_img = [f'{no_mask_dir}/{i}' for i in os.listdir(no_mask_dir)]


# Lets look at the total number of each images in each list, and the total of all images. This is used so we can identify how we will split the dataset into training and testing sets.

# In[ ]:


#Identify how many images we have in each group and total images
print("Total number of images with mask: " + str(len(mask_img)))
print("Total number of images without mask: " + str(len(no_mask_img)))
print("Total images: " + str(len(mask_img) + len(no_mask_img)))


# Usually we want to split the data in either a 70%/30% or a 80%/20% training/testing sets. For convenience I split just below a 80%/20% split, capturing the first 1500 images in each set for training. The remaining images will make up our testing set. After we have split for training and testing, we combine the training sets with and without masks, and do the same for the testing sets.

# In[ ]:


#Split the mask and no mask into training and testing sets
tr_mask = mask_img[0:1499]
tr_no_mask = no_mask_img[0:1499]
test_mask = mask_img[1500:]
test_no_mask = no_mask_img[1500:]

#Combine the training and testing sets
train_img = tr_mask + tr_no_mask
test_img = test_mask + test_no_mask


# **Image processing**
# 
# Now that we have captured all the images and split into training and testing sets, we will do some image processing and classification labeling. This utilizes the CV2 library for resizing the images, converting the images to 3 channel BGR colors, and pulling out a NumPy array version of the images to be used in TensorFlow. We wish to resize the images so they are all same size when inputting them into the neural networks. Converting the images to 3 channel BGR colors is so they aren't as complex (as a full colored image contains quite a lot more information, oftentimes unncessary and increasing our computational complexity). Converting the information into a NumPy array is necessary as an input for TensorFlow which Keras uses.
# 
# At the end we will print 5 images just to see what our data processing has accomplished.

# In[ ]:


#Define a function to resive and convert the images to the 3 channel BGR color image
#Also creates labels for with mask = 0 and without mask = 1 for classification use in neural network
def process_imgs(imgs, width=150, height=150):
    x = []
    y = []
    for i in imgs:
        x.append(cv2.resize(cv2.imread(i, cv2.IMREAD_COLOR), (width, height), interpolation=cv2.INTER_CUBIC))
        label = 1 if 'without' in i else 0
        y.append(label)
    return np.array(x), np.array(y)

tr_x, tr_y = process_imgs(train_img)
test_x, test_y = process_imgs(test_img)

# plot 5 images just to see the results of processing the images
plt.figure(figsize=(20, 10))
cols = 5
for i in range(cols):
    plt.subplot(5 / cols+1, cols, i+1) #keras
    plt.imshow(tr_x[i])


# Now that we have processed the images, we will use ImageDataGenerator for data augmentation. This will help the model generalize better. I suggest reading the relevant information available for ImageDataGenerator to learn more. We will run the training and the testing data through this.

# In[ ]:


#Image data augmentation for use in TensorFlow with test and training sets
tr_data = ImageDataGenerator(rescale=1/255,
                            rotation_range=40,
                            width_shift_range=0.2,
                            height_shift_range=0.2,
                            shear_range=0.2,
                            zoom_range=0.2,
                            horizontal_flip=True)

tr_gen = tr_data.flow(tr_x, tr_y, batch_size=32)
test_gen = tr_data.flow(test_x, test_y, batch_size = 32)


# ### **Building our first CNN**
# 
# Now that we have done all the image processing, we can build our first neural network. I will use an input convolutional 2D layer. There are numerous resources online describing what exactly a convolutional layer is and why it should be used, I highly suggest doing some personal research on this topic.
# 
# Notice the input_shape = (150,150,3), this is because when we processed the images our new width and height are 150 and the 3 is for the BGR channel.
# 
# Our output layer contains only 2 neurons (nodes) because this is a simple binary classification problem (with mask or without mask).
# 
# When designing a neural network it's important to play with the number of nodes in each layer, along with how many layers you want in your network. Just be aware that the more layers and nodes you add, the higher your compuational cost will be (how long it'll take for your model to be trained). Also we always run the risk of overfitting if our model becomes too complex. For this tutorial I tried to keep it simple.

# In[ ]:


#Designing our first CNN for training
model = models.Sequential()
model.add(Conv2D(64, (1, 1), input_shape = (150,150,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (1, 1), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(2, activation='softmax'))


# **We can look at our model we created using a simple method summary():**

# In[ ]:


print(model.summary())


# **Training our first model**
# 
# Time to train our model! We will compile using a RMSprop optimizer with a specified learning rate. I will repeat this exact process with this exact model using a different optimizer just for comparison later on.

# In[ ]:


#Compiling and training our first CNN using RMSprop as an optimizer
batch_size = 32
epochs = 20
model.compile(loss='sparse_categorical_crossentropy',
             optimizer=optimizers.RMSprop(lr=2e-5),
             metrics=['acc'])
hist = model.fit(tr_gen, steps_per_epoch=tr_x.shape[0] // batch_size, epochs=epochs)


# **How well does our model perform on the test set?**
# 
# Now that we have trained our model, lets see how it perfoms on the testing data. This is easily done using the method .evaluate()

# In[ ]:


#Comparing the accuracy and loss of our first CNN on the test data
results = model.evaluate(test_gen, batch_size = 32)
print("Test loss and test accuracy: ", results)


# **Graphing the epochs of the model we created**
# 
# We can use matplotlib to view the training progression of our model. This is why I stored the model to hist, so we can access the loss and accuracy over each epoch.

# In[ ]:


#Graphing the loss and accuracy for our first CNN
epochs = list(range(1, len(hist.history['acc'])+1))
accuracy = hist.history['acc']
loss = hist.history['loss']


plt.subplot(2,1,1)
plt.plot(epochs, accuracy)
plt.title("CNN for Accuracy and Loss (Mask vs No Mask)")
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.subplot(2,1,2)
plt.plot(epochs, loss)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()


# Let's check out if anything changes when utilizing the ADAM optimizer.

# **Building the same neural network to use the ADAM optimizer**
# 
# Now we will simply rebuild the same neural network (same number and type of layers and nodes):

# In[ ]:


#Rebuilding the same model to compile using a different optimizer
model2 = models.Sequential()
model2.add(Conv2D(64, (1, 1), input_shape = (150,150,3), activation='relu'))
model2.add(MaxPooling2D(pool_size=(2, 2)))
model2.add(Conv2D(128, (1, 1), activation='relu'))
model2.add(layers.Flatten())
model2.add(layers.Dense(256, activation='relu'))
model2.add(layers.Dense(2, activation='softmax'))


# Same process as above but this time we will utilize the string 'adam' for our optimizer argument and train the model this way.

# In[ ]:


#Compiling and training our first CNN using ADAM as an optimizer
batch_size = 32
epochs = 20
model2.compile(loss='sparse_categorical_crossentropy',
             optimizer='adam',
             metrics=['acc'])
hist2 = model2.fit(tr_gen, steps_per_epoch=tr_x.shape[0] // batch_size, epochs=epochs)


# Lets see how this model performs on the test set:

# In[ ]:


results2 = model2.evaluate(test_gen, batch_size = 32)
print("Test loss and test accuracy: ", results2)


# And as a comparison let's check out that matplotlib graph of loss and accuracy for each epoch:

# In[ ]:


epochs2 = list(range(1, len(hist2.history['acc'])+1))
accuracy2 = hist2.history['acc']
loss2 = hist2.history['loss']

plt.subplot(2,1,1)
plt.plot(epochs2, accuracy2)
plt.title("CNN for Accuracy and Loss (Mask vs No Mask)")
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.subplot(2,1,2)
plt.plot(epochs2, loss2)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()


# ## The power of Transfer Learning
# 
# Now let's use transfer learning to help improve our performance. What we are doing is using ImageNet's pre-trained weights for images to help improve our performance. What I mean by this, is instead of training our network from scratch, let's use some already trained model, throw a few layers on top for specialization with our dataset, and see if we can get increased performance.

# In[ ]:


#Loading pre-trained weights from ImageNet to save training time, thank you Transfer Learning
base=InceptionResNetV2(weights='imagenet',
                             include_top=False,
                             input_shape=(150, 150, 3))


# Notice I don't use a convoluational layer for this model. The idea is to rely heavily upon the ImageNet's pretrained model, and add only a few basic layers on top for specialization with our dataset. We build a model with our first layer consisting of the ImageNet's model.

# In[ ]:


model3 = models.Sequential()
model3.add(base)
model3.add(layers.Flatten())
model3.add(layers.Dense(256, activation='relu'))
model3.add(layers.Dense(2, activation='softmax'))
base.trainable = False


# In[ ]:


print(model3.summary())


# Lets train the model and see if anything improves!

# In[ ]:


batch_size = 32
epochs = 20
model3.compile(loss='sparse_categorical_crossentropy',
             optimizer='adam',
             metrics=['acc'])
hist3 = model3.fit(tr_gen, steps_per_epoch=tr_x.shape[0] // batch_size, epochs=epochs)


# In[ ]:


#Comparing the accuracy and loss of our model relying on ImageNet model as the first layer
results3 = model3.evaluate(test_gen, batch_size = 32)
print("Test loss and test accuracy: ", results3)


# In[ ]:


#Graphing the loss and accuracy for our model using ImageNet model as the first layer
epochs3 = list(range(1, len(hist3.history['acc'])+1))
accuracy3 = hist3.history['acc']
loss3 = hist3.history['loss']


plt.subplot(2,1,1)
plt.plot(epochs3, accuracy3)
plt.title("ImageNet and our NN for Accuracy and Loss (Mask vs No Mask)")
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.subplot(2,1,2)
plt.plot(epochs3, loss3)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()


# That's all folks! I hope this was clear and easy to follow, and hopefully demostrated the power of transfer learning.
# 
# If you enjoyed this notebook please throw me an upvote!
# Thanks for reading!
