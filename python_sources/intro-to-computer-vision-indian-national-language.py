#!/usr/bin/env python
# coding: utf-8

# # Start Note:
# ### I would like to thank the data provider for providing this hard to get by dataset.
# Source - https://web.archive.org/web/20160105230017/http://cvresearchnepal.com/wordpress/dhcd/

# 1. Introduction
# 2. Reading Images from Folder with class name
# 3. Class Distribution
# 4. Image Characteristics
# 5. Identifying Grayscale Images
# 6. Splitting Input and Target Data
# 7. Target Label Binarization
# 8. Input Image Reshaping
# 9. Image Normalisation
# 10. Splitting into Training and Testing set
# 11. Stratified split
# 12. Split Frequency check
# 13. Modelling
# 14. Evaluation

# # Introduction
# 

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2 
import copy
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import glob
import os
from tqdm import tqdm 
import matplotlib.image as mpimg
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools


# In[ ]:


dataset = pd.read_csv("/kaggle/input/devanagari-character-set/data.csv")


# In[ ]:


dataset.head()


# In[ ]:


dataset.shape


# ## A few points:
# 
# #### 1. Firstly the dataset originally has images stored in a flattened manner. Two dimensional array has been flattened into a one dimension array.
# #### 2. I have come across notebooks that take the given 1-D array as input to build models. I appreciate the approach but believe that atleast some crucial information about the spacial arrangement of pixels is lost in doing so. Hence my modelling is based on 2-D image input.
# #### 3. Though I have used the given csv from the dataset, I have added a portion briefly on how to read images from specific folders where the folder name is the class name. I hope this will act as a baseline for many others who are required to take input in this manner for other competitions and datasets as well.

# # Reading Images from Folder with class name

# In[ ]:


def read_data():

    data_set_full = [] 
    for folder_path in glob.glob('/kaggle/input/devanagari-character-set/Images/Images/*'):

        class_name = folder_path.split('/')[-1]
        print(class_name)

        for img in tqdm(os.listdir(folder_path)): 
            path = os.path.join(folder_path, img)
            img = cv2.imread(path)
            data_set_full.append([np.array(img), class_name])
            #print(data_set_full)
            #break
        
    return data_set_full


# In[ ]:


## Commented due to long run time once after the data had been generated.
##full_data = read_data()


# In[ ]:


##len(full_data)


# # Class Distribution

# In[ ]:


plt.figure(figsize = (10,20))
_ = sns.countplot(x=None, y=dataset.character)


# ### Every class is equally distributed, very well. We won't encounter the problem of class imbalance like [here](https://www.kaggle.com/shrutimechlearn/pokemon-classification-and-smote-for-imbalance)

# In[ ]:


dataset.drop(['character'], axis =1).iloc[0].max()


# # Image Characteristics

# ## Reading Images

# In[ ]:


img = (dataset.drop(['character'],axis=1).iloc[0]).to_numpy().reshape(32,32)
_ = plt.imshow(img, cmap = 'gray')


# ### This is a grayscale image and I have used cmap = 'gray' to for best suiting representation of the image. Also notice the shades of gray in the image. Had it been a binary image there would be just black and white colors. But how did I identify its type?

# # Identifying Grayscale Images

# ### Before going to the theory let's look at a plain old pythonic way of checking whether an image is grayscale

# In[ ]:


#import Image

def is_grey_scale(img_path):
    img = Image.open(img_path).convert('RGB')
    w,h = img.size
    for i in range(w):
        for j in range(h):
            r,g,b = img.getpixel((i,j))
            if r != g != b: return False
    return True


### Basically, check every pixel to see if it is grayscale (R == G == B)


# ## Theory

# ## A grayscale image is formed when the red, blue and green component of the image have the same value for each pixel. So that means we should have a multidimensional array with values like m x n x 3, 3 for representing the 3 channels. But here we can see from the dataset that each row represents a single image and has only one channel. Also the pixels have values other than 0 and 1 so its not binary. So does that mean that this image format is unknown to us and we can't identify it?
# ## No, the image format is very well known to us by now and its none other than grayscale. Since the red, blue and green channels of all grayscale images have the same value at each pixel, it is smart to store the image in a single 2-D array rather than having 3 2-D arrays having the same values. Though when the image is read, the final result is a merge of the 3 channels (having same values of pixels) basically a grayscale image.

# ## Practical Example

# In[ ]:


ones_array = np.ones([100, 100, 3], dtype=np.uint8)
_ = plt.imshow(ones_array)


# In[ ]:


red_array = copy.deepcopy(ones_array)
red_array[:,:,0] = 255
red_array[:,:,1] = 0
red_array[:,:,2] = 0

_ = plt.imshow(red_array)


# ### Now to make any shade of gray

# In[ ]:


any_gray_array = copy.deepcopy(ones_array)
any_gray_array[:,:,0] = 200
any_gray_array[:,:,1] = 200
any_gray_array[:,:,2] = 200

_ = plt.imshow(any_gray_array)


# In[ ]:


any_gray_array = copy.deepcopy(ones_array)
any_gray_array[:,:,0] = 150
any_gray_array[:,:,1] = 150
any_gray_array[:,:,2] = 150

_ = plt.imshow(any_gray_array)


# In[ ]:


any_gray_array = copy.deepcopy(ones_array)
any_gray_array[:,:,0] = 100
any_gray_array[:,:,1] = 100
any_gray_array[:,:,2] = 100

_ = plt.imshow(any_gray_array)


# ### Now let's come back to the modelling pipeline

# # Splitting Input and Target Data

# In[ ]:


x = dataset.drop(['character'],axis = 1)
y_text = dataset.character


# In[ ]:


x.shape


# ## Target Label Binarization

# In[ ]:


from sklearn.preprocessing import LabelBinarizer
binencoder = LabelBinarizer()
y = binencoder.fit_transform(y_text)


# In[ ]:


y[0]


# ## Input Image Reshaping

# In[ ]:


x = x.values.reshape(x.shape[0],32,32,1)


# In[ ]:


x.shape


# # Image Normalisation

# ## Normalization is a process that changes the range of pixel intensity values

# In[ ]:


print(x.max())
print(x.mean())
print(x.sum())


# In[ ]:


x = x/255.0


# In[ ]:


print(x.max())
print(x.mean())
print(x.sum())


# ## Splitting into Training and Testing set

# In[ ]:


x.shape


# In[ ]:


y.shape


# In[ ]:


(unique, counts) = np.unique(y, return_counts=True, axis = 0)
frequencies = np.asarray((binencoder.inverse_transform(unique), counts)).T


# In[ ]:


frequencies


# ### Stratified split

# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.33, random_state=42, stratify = y)


# ### Shape check

# In[ ]:


x_train.shape


# In[ ]:


x_test.shape


# In[ ]:


y_train.shape


# In[ ]:


y_test.shape


# ### Frequency check

# In[ ]:


(unique, counts) = np.unique(y_train, return_counts=True, axis = 0)
frequencies = np.asarray((binencoder.inverse_transform(unique), counts)).T


# In[ ]:


frequencies


# In[ ]:


(unique, counts) = np.unique(y_test, return_counts=True, axis = 0)
frequencies = np.asarray((binencoder.inverse_transform(unique), counts)).T


# In[ ]:


frequencies


# ### So the train data has 1340 examples and test has 660 examples of each of the 46 classes.

# # Modelling

# In[ ]:


from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau


# In[ ]:


model = Sequential()

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu', input_shape = (32,32,1)))
model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))


model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))


model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(46, activation = "softmax"))


# In[ ]:


# Define the optimizer
optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)


# In[ ]:


# Compile the model
model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])


# In[ ]:


model.summary()


# In[ ]:


# Set a learning rate annealer
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)


# In[ ]:


epochs = 7
batch_size = 86


# In[ ]:


history = model.fit(x_train, y_train, batch_size = batch_size, epochs = epochs, 
       validation_data = (x_test, y_test), verbose = 2)


# # Evaluation

# In[ ]:


scores = model.evaluate(x_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))


# In[ ]:


# Plot the loss and accuracy curves for training and validation 
fig, ax = plt.subplots(2,1)
ax[0].plot(history.history['loss'], color='b', label="Training loss")
ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])
legend = ax[0].legend(loc='best', shadow=True)

ax[1].plot(history.history['accuracy'], color='b', label="Training accuracy")
ax[1].plot(history.history['val_accuracy'], color='r',label="Validation accuracy")
legend = ax[1].legend(loc='best', shadow=True)


# ## Updating With More Details Very Soon

# ### Thanks! Do upvote if you found it helpful. 
