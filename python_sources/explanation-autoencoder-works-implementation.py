#!/usr/bin/env python
# coding: utf-8

# In[ ]:



import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import os

print(os.listdir("../input"))


# > **Why   this  kernel  ?******

# In[ ]:


from IPython.display import Image
Image("../input/sabihaprova/oka.png")
from IPython.display import Image
Image("../input/provathebeauty/ok.png")


# **PROBLEM SATEMENT:**
# > suppose you are given a dirty document(like above image)/noisy image, you have to make a clean document/image, so how will you do that?
# 
# **SOLUTION**
# > you can use** AutoEncoder** to solve this problem

# In[ ]:


from IPython.display import Image
Image("../input/provathebeauty/okkk.png")


# > 
# 
# > 1. **An auto encoder is neural network consisting of hidden layer and it has two part encoder and decoder**
# > > 2. **It is simply like an identity function like f(x)=x where it makes same size output as like as imput by removing noise**

# **How denosing Auto Encoder works
# ?**

# In[ ]:


from IPython.display import Image
Image("../input/provathebeauty/o.png")


# **Quick explanation**:
# 
# >       Lets suppose:
#                   x=  input
#                   y=  hidden layer output
#                   z=  output of autoencoder
#                   L=  Loss function
#                   so, it first set missing  some  nodes in input layers. then forwarded to hidden layer then hidden layer output goes to loss function. then by calculation loss function ,backpropagate and update weights.then by using weights it approximate the output

# **Question:**
# 
# >           what is the difference between dropout and autoencoder as both of them set off the nodes? 
# 
# **Answer:**
# 
# >          Main difference is in dropout(regularization technique for reducing overfitting)use dropout(set off) nodes            in every layer in neural network,where autoencoder set off some nodes in only in input layer
#            

# **How   to  implement  Denosing  Auto  Encoder?**
# 
# >          we know that in CNN, frist few layers are used for feature extraction like: finding shape of object, edge detection. and after convolution layer we used pooling layer. so by using these layer we extract just features by removing the redundant information.and these feature represent the input object. so here we just find compact representation of this input object. this is like encoding.
# 
# >        so in the Next layers we just perform the reverse operation of pooling .this reverse pooling is called Upsampling. this is called decodidng.
# 
# >  so Denosing Auto encoder consist of Two part like:
#                                 1. Encoder
#                                 2. Decoder.
#                                 
#                                 
# > Implementation in keras is given below:
# 
# >     def autoencoder(input_img):
# >     #encoder
# >     #input = 28 x 28 x 1 (wide and thin)
# >     conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img) #28 x 28 x 32
# >     pool1 = MaxPooling2D(pool_size=(2, 2))(conv1) #14 x 14 x 32
# >     conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1) #14 x 14 x 64
# >     pool2 = MaxPooling2D(pool_size=(2, 2))(conv2) #7 x 7 x 64
# >     conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2) #7 x 7 x 128 (small and thick)
# > 
# >     #decoder
# >     conv4 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3) #7 x 7 x 128
# >     up1 = UpSampling2D((2,2))(conv4) # 14 x 14 x 128
# >     conv5 = Conv2D(64, (3, 3), activation='relu', padding='same')(up1) # 14 x 14 x 64
# >     up2 = UpSampling2D((2,2))(conv5) # 28 x 28 x 64
# >     decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(up2) # 28 x 28 x 1
# >     return decoded
#                                 
#                                 
#                                 
# 
# 

# **Source**:
#       *Medical image denosing using convolutional denosing encoder by Lovedeep Gondara*

# **Now  solve the Denosing Dirty Document problem:**

# In[ ]:


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import cv2


# In[ ]:


#lets define a function for rading train and test images


# In[ ]:


get_ipython().system('pip install python-resize-image')


# In[ ]:


print(os.listdir("../input"))


# In[ ]:


img = cv2.imread('../input/denoising-dirty-documents/test/1.png', 0)
plt.imshow(img,cmap='gray')


# In[ ]:


from PIL import Image
from resizeimage import resizeimage

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        if filename == "train":
            continue
        if filename == "test":
            continue
        if filename == "train_cleaned":
            continue
        img = cv2.imread(os.path.join(folder,filename))
        img = np.array(img)
        s = img.shape
        s = np.array(s)
        if  s[0] == 258:
            img1 = Image.open(os.path.join(folder,filename))
            new1 = resizeimage.resize_contain(img1, [540, 420, 3])
            new1 = np.array(new1, dtype='uint8')
            images.append(new1)
        else:
            img1 = Image.open(os.path.join(folder,filename))
            images.append(img)
    return images

train = load_images_from_folder("../input/denoising-dirty-documents/train")
test = load_images_from_folder("../input/denoising-dirty-documents/test")
train_cleaned = load_images_from_folder("../input/denoising-dirty-documents/train_cleaned")


# In[ ]:



#now convert these image list into array and then convert values in range o-1
train = np.array(train)
test = np.array(test)
train_cleaned = np.array(train_cleaned)

train = train.astype('float32') / 255
test = test.astype('float32') / 255
train_cleaned = train_cleaned.astype('float32') / 255


# In[ ]:


import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding
from keras.layers import SpatialDropout1D, Conv2D, MaxPooling2D, UpSampling2D


# In[ ]:


model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(420, 540, 3,))) 
model.add(MaxPooling2D((2, 2), padding='same'))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2), padding='same'))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(3, (3, 3), activation='sigmoid', padding='same'))

model.summary() 

model.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=["accuracy"])


# In[ ]:


model.fit(train, train_cleaned, epochs=100, batch_size=52, shuffle=True, validation_data=(train, train_cleaned))


# In[ ]:


pred=model.predict(test)


# In[ ]:


array=np.array(pred)


# In[ ]:


array.shape


# In[ ]:


for img in array:
    plt.show()
    plt.imshow(img,cmap='gray')

