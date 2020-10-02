#!/usr/bin/env python
# coding: utf-8

# - In this kernel I will define all the steps required for training the CNN model to detect ASL sign language from the given input i.e., palm images.
# - Aim of this kernel is make some complex steps simple for newbies to quickly understand the CNN concepts and its implementation.
# - **Please feel free to upvote the kernel if you find it helpful**.
# - Fork the code to modify and experiment with the code a bit.
# 
# ## Step 1 - Load all the required libraries
# 
# - Added comments to explain which libraries are being used for what purpose

# In[ ]:


#for file operations
import os

#for converting lists into numpy arrays and to perform related operations on numpy arrays
import numpy as np

#for image loading and processing
import cv2
from PIL import Image

#for data/image visualization
import matplotlib.pyplot as plt

#for splitting dataset
from sklearn.model_selection import train_test_split

#for building and training a CNN model
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Dropout
from keras.layers.normalization import BatchNormalization

print("Loaded all libraries")


# ## Step 2 - Load dataset
# 
# - First check the data provided in the input folder

# In[ ]:


os.listdir("../input/")


# - We will use data from train dataset to train the CNN model which we will be building in the succeeding steps

# In[ ]:


fpath = "../input/asl_alphabet_train/asl_alphabet_train/"
categories = os.listdir(fpath)
print("No. of categories of images in the train set = ",len(categories))


# - Since this is a supervised learning approach to train the model we will have to label the images before giving it as input to the CNN model

# In[ ]:


def load_images_and_labels(categories):
    img_lst=[]
    labels=[]
    for index, category in enumerate(categories):
        n = 0
        for image_name in os.listdir(fpath+"/"+category):
            if n==100:
                break
            #load image data into an array
            img = cv2.imread(fpath+"/"+category+"/"+image_name)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_array = Image.fromarray(img, 'RGB')
            
            #data augmentation - resizing the image
            resized_img = img_array.resize((200, 200))
            
            #converting the image array to numpy array before appending it to the list
            img_lst.append(np.array(resized_img))
            
            #appending label
            labels.append(index)
            
            n+=1
    return img_lst, labels

images, labels = load_images_and_labels(categories)
print("No. of images loaded = ",len(images),"\nNo. of labels loaded = ",len(labels))
print(type(images),type(labels))


# - To give the labels and images data as input to the CNN model they have to be in numpy array format so we have to convert the lists initialized in the previous step to numpy arrays

# In[ ]:


images = np.array(images)
labels = np.array(labels)

print("Images shape = ",images.shape,"\nLabels shape = ",labels.shape)
print(type(images),type(labels))


# - Let's check how the loaded data in the numpy arrays look like by displaying them as images

# In[ ]:


def display_rand_images(images, labels):
    plt.figure(1 , figsize = (15 , 10))
    n = 0 
    for i in range(4):
        n += 1 
        r = np.random.randint(0 , images.shape[0] , 1)
        
        plt.subplot(2, 2, n)
        plt.subplots_adjust(hspace = 0.3 , wspace = 0.1)
        plt.imshow(images[r[0]])
        
        plt.title('Assigned label : {}'.format(labels[r[0]]))
        plt.xticks([])
        plt.yticks([])
        
    plt.show()
    
display_rand_images(images, labels)


# ## Step 3 - Prepare the data for training the CNN model
# 
# - In the previous step we successfully loaded the images data into numpy array and labeled them.
# - Now in this step we have to ensure that we have data not only to train the model but also to test it.

# In[ ]:


#1-step in data shuffling
random_seed = 101

#get equally spaced numbers in a given range
n = np.arange(images.shape[0])
print("'n' values before shuffling = ",n)

#shuffle all the equally spaced values in list 'n'
np.random.seed(random_seed)
np.random.shuffle(n)
print("\n'n' values after shuffling = ",n)


# In[ ]:


#2-step in data shuffling

#shuffle images and corresponding labels data in both the lists
images = images[n]
labels = labels[n]

print("Images shape after shuffling = ",images.shape,"\nLabels shape after shuffling = ",labels.shape)


# In[ ]:


#3-data normalization

images = images.astype(np.float32)
labels = labels.astype(np.int32)
images = images/255
print("Images shape after normalization = ",images.shape)


# - Display and check how the images look after the changes made in the previous steps

# In[ ]:


display_rand_images(images, labels)


# - Split the dataset into 2 parts - train set, test set

# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size = 0.2, random_state = random_seed)

print("x_train shape = ",x_train.shape)
print("y_train shape = ",y_train.shape)
print("\nx_test shape = ",x_test.shape)
print("y_test shape = ",y_test.shape)


# In[ ]:


display_rand_images(x_train, y_train)


# ## Step 4 - Build a CNN model and train it
# 
# - Define layers in the CNN model

# In[ ]:


model = Sequential()

#1 conv layer
model.add(Conv2D(filters = 16, kernel_size = 3, activation = "relu", input_shape = x_train.shape[1:]))

#1 max pool layer
model.add(MaxPooling2D(pool_size = 3))

#2 conv layer
model.add(Conv2D(filters = 32, kernel_size = 3, activation = "relu"))

#2 max pool layer
model.add(MaxPooling2D(pool_size = 3))

#3 conv layer
model.add(Conv2D(filters = 64, kernel_size = 3, activation = "relu"))

#3 max pool layer
model.add(MaxPooling2D(pool_size = 3))

model.add(BatchNormalization())

model.add(Flatten())

#1 dense layer
model.add(Dense(1000, input_shape = x_train.shape, activation = "relu"))

model.add(Dropout(0.4))

model.add(BatchNormalization())

#2 dense layer
model.add(Dense(500, activation = "relu"))

model.add(Dropout(0.4))

model.add(BatchNormalization())

#output layer
model.add(Dense(29,activation="softmax"))

model.summary()


# - Compile the CNN model that we defined in the previous step

# In[ ]:


model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])


# ## Step 5 - Train the model

# In[ ]:


get_ipython().run_cell_magic('time', '', 'model.fit(x_train, y_train, epochs=100, batch_size = 100)')


# - Check how the trained model is performing on the test dataset

# In[ ]:


loss, accuracy = model.evaluate(x_test, y_test)

print("Loss = ",loss,"\nAccuracy = ",accuracy)


# ## Step 6 - Predict values using the trained model

# In[ ]:


pred = model.predict(x_test)

pred.shape


# - Let's check what labels the CNN model has predicted by displaying the image and its labels

# In[ ]:


plt.figure(1 , figsize = (15, 10))
n = 0 

for i in range(4):
    n += 1 
    r = np.random.randint(0, x_test.shape[0], 1)
    
    plt.subplot(2, 2, n)
    plt.subplots_adjust(hspace = 0.3, wspace = 0.3)
    
    plt.imshow(x_test[r[0]])
    plt.title('Actual = {}, Predicted = {}'.format(y_test[r[0]] , y_test[r[0]]*pred[r[0]][y_test[r[0]]]) )
    plt.xticks([]) , plt.yticks([])

plt.show()

