#!/usr/bin/env python
# coding: utf-8

# # Contents
# 
# 1. Introduction
# 
# 2. Importing Libraries
# 
# 3. Reading and Manipulating Data
# 
# 4. Creating x and y Labels
# 
# 5. Creating CNN Model with Keras
# 
# 6. Applying CNN Model
# 
# 7. Test Model
# 
# 8. Conclusion

# # 1. Introduction
# 
# ![](http://www.wikizero.biz/index.php?q=aHR0cHM6Ly91cGxvYWQud2lraW1lZGlhLm9yZy93aWtpcGVkaWEvY29tbW9ucy82LzY5L01hbGFyaWFfUGFyYXNpdGVfQ29ubmVjdGluZ190b19IdW1hbl9SZWRfQmxvb2RfQ2VsbF8lMjgzNDAzNDE0MzQ4MyUyOS5qcGc)
# 
# Malaria is a mosquito-borne infectious disease that affects humans and other animals. Malaria causes symptoms that typically include fever, tiredness, vomiting, and headaches. In severe cases it can cause yellow skin, seizures, coma, or death. Symptoms usually begin ten to fifteen days after being bitten by an infected mosquito. If not properly treated, people may have recurrences of the disease months later. In those who have recently survived an infection, reinfection usually causes milder symptoms. This partial resistance disappears over months to years if the person has no continuing exposure to malaria.
# 
# Source: wikipedia.com

# # 2. Importing Libraries
# 
# I used pyplot from matplotlib for showing test results, Pillow(PIL) library for manipulating images.

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from PIL import Image
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# # 3. Reading and Manipulating Data

# There is two different folders named uninfected and parasitized. I took file names from each folder with below code.

# In[2]:


parasitized = os.listdir("../input/cell_images/cell_images/Parasitized")
uninfected = os.listdir("../input/cell_images/cell_images/Uninfected")


# I removed "Thumbs.db" filenames from our parasitized and uninfected lists. If not, next processes will be broken due to wrong file extension.

# In[3]:


parasitized.remove("Thumbs.db")
uninfected.remove("Thumbs.db")


# Now we have all image names in "parasitized" and "uninfected" lists. But images don't have same pixel rates. We need to resize all pictures and I choose 50x50 for this kernel.

# In[4]:


parasitized_images = []
for p in parasitized:
    img = Image.open("../input/cell_images/cell_images/Parasitized/"+p)
    img = img.resize((50,50))
    parasitized_images.append(img)

uninfected_images = []
for u in uninfected:
    img = Image.open("../input/cell_images/cell_images/Uninfected/"+u)
    img = img.resize((50,50))
    uninfected_images.append(img)


# We can see what parasitized and uninfected cells looks like below.

# In[5]:


rndm = np.random.randint(len(parasitized_images)-1,size = 10)
plt.figure(1, figsize=(15,7))
for i in range(1,11):
        plt.subplot(2,5,i)
        if i < 6:
            plt.imshow(parasitized_images[rndm[i-1]])
            plt.axis("off")
            plt.title("Parasitized")
        else:
            plt.imshow(uninfected_images[rndm[i-1]])
            plt.axis("off")
            plt.title("Uninfected")


# # 4. Creating x and y Labels

# Now we have to create x(pixels) and y(class) axis for each images. For x labels we need (27558(total sample), 50(horizontal pixel qty), 50(vertical pixel qty), 3(RGB)) array.
# Also for Keras, we should feed models with integers if we want to implement RGB images.

# In[7]:


x_array = np.empty((len(parasitized_images)+len(uninfected_images), 50, 50, 3))
x_array = x_array.astype(int)


# Filling empty numpy array with image values.

# In[12]:


index = 0
for i in range(x_array.shape[0]):
    if i < len(parasitized_images):
        x_array[i] = np.array(parasitized_images[i])
    else:
        x_array[i] = np.array(uninfected_images[index])
        index += 1


# When I create y label, I will consider parasitized as 1 and uninfected as 0.

# In[13]:


y_array = np.append(np.ones(len(parasitized_images)), np.zeros(len(uninfected_images)))


# In[15]:


from keras.utils.np_utils import to_categorical
y_array = to_categorical(y_array, num_classes = 2)


# Split data for train and test with sklearn library.

# In[16]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_array, y_array, random_state = 42, test_size = 0.1)
print("x_train shape: ",x_train.shape)
print("x_test shape: ",x_test.shape)
print("y_train shape: ",y_train.shape)
print("y_test shape: ",y_test.shape)


# In[17]:


plt.imshow(x_train[1991])
plt.axis("off")
plt.title("Sample")
plt.show()


# # 5. Creating CNN Model with Keras

# In[18]:


from sklearn.metrics import confusion_matrix
import itertools

from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization
from keras.optimizers import RMSprop,Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau

model = Sequential()

model.add(Conv2D(filters = 32, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu', input_shape = (50,50,3)))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(BatchNormalization(axis = -1))
model.add(Dropout(0.2))

model.add(Conv2D(filters = 32, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(BatchNormalization(axis = -1))
model.add(Dropout(0.2))

model.add(Conv2D(filters = 32, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(BatchNormalization(axis = -1))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(512, activation = "relu"))
model.add(BatchNormalization(axis = -1))
model.add(Dropout(0.5))
model.add(Dense(2, activation = "softmax"))


# In[19]:


model.compile(optimizer = "Adam" , loss = "categorical_crossentropy", metrics=["accuracy"])


# In[20]:


epochs = 20
batch_size = 32


# In[22]:


datagen = ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        rotation_range=0.5,
        zoom_range = 0.5,
        width_shift_range=0.5,
        height_shift_range=0.5,
        horizontal_flip=False,
        vertical_flip=False)

datagen.fit(x_train)


# # 6. Applying CNN Model

# In[23]:


history = model.fit(x_train,y_train,epochs=epochs, batch_size=batch_size)


# In[24]:


plt.plot(history.history['acc'], color='r', label="accuracies")
plt.title("Train Accuracies")
plt.xlabel("Number of Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()


# We trained our model with train datas. Now time to test our model.

# # 7. Test Model

# In[30]:


print("Test accuracy: {} %".format(round(model.evaluate(x_test,y_test)[1]*100,2)))


# # 8. Conclusion
# 
# When we test our model with test data we hit nearly 95% accuracy. So we can say CNN method with this dataset succesful.
# 
# Days before I hit nearly 60% test accuracy with Logistic Regression method. Also with ANN algorithm mean accuracy with this method was 65%.
# 
# From here we could say CNN is the better method when processing images.
# 
# Thank you for reading this Kernel. I hope you enjoyed it.
