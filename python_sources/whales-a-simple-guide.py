#!/usr/bin/env python
# coding: utf-8

# We all like to play around with data and get things done. In this kernel I'll show you how you can do it yourself.
# ## Full Video Explanation of this Notebook
# [KAGGLE KERNELS 2019](https://www.youtube.com/watch?v=AXcTm4gFerE)
# 
# I also have a full explanation on how to work with large Image datasets ( like this one :D ) 
# [How to Deal with Large Image Datasets](https://www.youtube.com/watch?v=myYMrZXpn6U) and you can check out the [Kernel](https://www.kaggle.com/jhonatansilva31415/loading-all-whales-into-memory)
# 
# ## Notebook Content
# 1. [Resources](#zeroth-bullet)
# 2. [Some libraries we need to get things done](#first-bullet)
# 3. [How to load the dataset](#second-bullet)
# 4. [Looking at 5 random beauties](#third-bullet)
# 5. [Preprocessing the data](#forth-bullet)<br/>
#      5.1 [Using python OpenCV](#forth1-bullet)<br/>
#      5.2 [Using torchvision](#forth2-bullet)<br/>
# 6. [Cleaning the Data](#fifth-bullet)
# 7. [Encoding](#fifth-bullet)
# 8. [Handling the dataset](#sixth-bullet)
# 9. [Building a very simple sequential model](#seventh-bullet)
# 10. [Conclusion](#eighth-bullet)

# ### Resources <a class="anchor" id="zeroth-bullet"></a>
# I'm currently making a video series explaining step by step this kernel, if this sounds interesting, here are the links 
# 1. [Introduction](https://www.youtube.com/watch?v=pD_IR72g5tE&t=1s)
# 2. [Libraries](https://www.youtube.com/watch?v=2iRIPjXTGeY&t=1s) 

# ### Some libraries we need to get things done <a class="anchor" id="first-bullet"></a>

# In[ ]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import cv2
import os
from PIL import Image

from matplotlib.pyplot import imshow
from IPython.display import HTML


# ### Working with files
# It's always a pain in the ass to work with paths, when I was starting I almost have all my paths hardcoded. When working with teams I saw that this approach isn't get me anywhere, it's always, ok, in most cases, a great idea to store your general paths into variables.

# In[ ]:


print(os.listdir('../input'))


# In[ ]:


img_train_path = os.path.abspath('../input/train')
img_test_path = os.path.abspath('../input/test')
csv_train_path = os.path.abspath('../input/train.csv')
csv_train_path


# ### How to load the dataset <a class="anchor" id="second-bullet"></a>
# We'll use here the [Pandas](https://pandas.pydata.org/pandas-docs/stable/) to load the dataset into memory

# In[ ]:


df = pd.read_csv(csv_train_path)
df.head()


# We can see that we have the paths of the images and the labels associated with the whales. To easy the image reading process we can create a aditional column to the dataset with the global path of the images

# In[ ]:


df['Image_path'] = [os.path.join(img_train_path,whale) for whale in df['Image']]


# In[ ]:


df.head()


# ### Looking at 5 random beauties  <a class="anchor" id="third-bullet"></a>
# It's a great deal of fun to explore the data and play around with *matplotlib*

# In[ ]:


full_path_random_whales = np.random.choice(df['Image_path'],5)


# In[ ]:


full_path_random_whales


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
for whale in full_path_random_whales:
    img = Image.open(whale)
    plt.imshow(img)
    plt.show()


# ### Preprocessing the data <a class="anchor" id="forth-bullet"></a>
# I could find some cool resources to help me put all this together. You'll find it extremely usefull
# * [DATA LOADING AND PROCESSING TUTORIAL](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html)
# * [Lecture Notes: Basic Image Processing](https://www.cs.virginia.edu/~vicente/recognition/notebooks/image_processing_lab.html)
# * [PyTorch quick start: Classifying an image](http://blog.outcome.io/pytorch-quick-start-classifying-an-image/)
# 
# 
# Here we're going to use 2 approaches, basic OpenCv and PyTorch.

# In[ ]:


from torchvision import transforms


# #### Using python OpenCV <a class="anchor" id="forth1-bullet"></a>
# OpenCV is a great, great, computer vision library. Here I just use the basics of it, but you can go wild with OpenCv. We are going to use to scale the images down and convert to grayscale

# In[ ]:


img = cv2.imread(full_path_random_whales[0])
img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
res = cv2.resize(img, dsize=(128, 128), interpolation=cv2.INTER_CUBIC)
plt.imshow(res,cmap='gray')
plt.show()


# #### Using torchvision <a class="anchor" id="forth2-bullet"></a>
# PyTorch is a library developed by Facebook, the torchvision module has some convenient features, like we're using here
# * Convert to grayscale
# * Resize
# * Corp
# * Transform to tensor
# * Normalize

# In[ ]:


normalize = transforms.Normalize(
   mean=[0.485, 0.456, 0.406],
   std=[0.229, 0.224, 0.225]
)
preprocess = transforms.Compose([
   transforms.Grayscale(num_output_channels=1),
   transforms.Resize(128),
   transforms.CenterCrop(128),
   transforms.ToTensor(),
   normalize
])
imgs = [Image.open(whale) for whale in full_path_random_whales]
imgs_tensor = [preprocess(whale) for whale in imgs]


# In[ ]:


imgs_tensor[0].shape


# In[ ]:


img = imgs_tensor[0]
plt.imshow(img[0],cmap='gray')
plt.show()


# ### Cleaning the Data <a class="anchor" id="fifth-bullet"></a>
# [Why removing new_whale is a good idea](https://www.kaggle.com/suicaokhoailang/removing-class-new-whale-is-a-good-idea)
# 
# Working with biases datasets is a huge problem, you can look more in this blog I posted a while ago also using data from a Kaggle competition 
# 
# [Why you should care about bias.](https://jhonatandasilva.com/bias-in-ai/)

# In[ ]:


df.Id.value_counts().head()


# We can create a new dataframe just for testing purposes without the new_whale class

# In[ ]:


I_dont_want_new_whales = df['Id'] != 'new_whale'
df = df[I_dont_want_new_whales]
df.Id.value_counts().head()


# ### Encoding <a class="anchor" id="sixth-bullet"></a>
# To further use torchvision we need to encode our data, here's how you can do it

# In[ ]:


unique_classes = pd.unique(df['Id'])
encoding = dict(enumerate(unique_classes))
encoding = {value: key for key, value in encoding.items()}
df = df.replace(encoding)


# In[ ]:


df.head()


# ### Handling the dataset <a class="anchor" id="sixth-bullet"></a>
# (Don't do this in your personal computer, this isn't a great way to open your images, just for test purposes)

# In[ ]:


import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch import nn
from torch import optim
from torch.utils.data import Dataset, DataLoader


# #### Simple model
# As we are going to construct a simple sequencial linear model we will load just 1000 images to test it out

# In[ ]:


test = df['Image_path'][:1000]
imgs = [Image.open(whale) for whale in test]
imgs_tensor = torch.stack([preprocess(whale) for whale in imgs])


# In[ ]:


labels = torch.tensor(df['Id'][:1000].values)
max_label = int(max(labels)) +1
max_label


# In[ ]:


plt.imshow(imgs_tensor[0].reshape(128,128),cmap='gray')


# ### Building a very simple sequential model <a class="anchor" id="seventh-bullet"></a>
# 
# This is a great way to play around if you are a begginner in the area. If you don't know much from building Neural Networks I have a few resources 
# 
# 1. [Creating a Perceptron](https://jhonatandasilva.com/build-your-own-perceptron/)
# 2. [What are the building blocks of Deep Learning](https://jhonatandasilva.com/perceptrons/) 
# 3. [Play around with Neural Nets](https://jhonatandasilva.com/play-with-nn/)
# 4. [Training your Neural Net](https://jhonatandasilva.com/training-your-neural-networks/)
# 5. [When all comes together](https://jhonatandasilva.com/mnist-pytorch/) 
# 
# Exploring more on the Vision side there's also
# 
# 1. [How Neural Nets sees the world ](https://jhonatandasilva.com/how-nn-sees-the-world/)
# 2. [How to build your CNN with Keras](https://youtu.be/lkvzqfhgITM)

# ### I know I know
# We are all busy people and don't have time to waste in a bunch of blogs posts, so I'll give it to you straight. I've created some animations to explain better how the model bellow works ( I like art, sue me )
# 
# #### THE VISION SIDE
# We are **NOT** using Convolutional Neural Networks here, so we need to feed our Neural Net a flatten vector, but what does that mean? We get our 128x128 our what side do you choose and transform into a one dimensional vector
# 
# <img src="https://jhonatandasilva.com/wp-content/uploads/2018/12/flattening.gif" alt="drawing" width="200"/>
# 
# ### THE NEURAL NET SIDE
# 
# Ok, now we have our one dimensional vector, what do we do? We feed one by one into our neural net and it gives out a probability for the whale class
# 
# <img src="https://jhonatandasilva.com/wp-content/uploads/2018/12/nn.gif" alt="drawing" width="600"/>
# 
# 

# In[ ]:


model = nn.Sequential(nn.Linear(128*128, 256),
                      nn.Sigmoid(),
                      nn.Linear(256, 128),
                      nn.Sigmoid(),
                      nn.Linear(128, max_label),
                      nn.LogSoftmax(dim=1))

optimizer = optim.SGD(model.parameters(), lr=0.01)
criterion = nn.NLLLoss()

model


# In[ ]:


epochs = 5
batch_size = 10
iters = int(len(imgs_tensor)/batch_size)
next_batch = 0
for e in range(epochs):
    running_loss = 0
    next_batch = 0
    for n in range(iters):
        batch_images = imgs_tensor[next_batch:next_batch+batch_size] 
        batch_images = batch_images.view(batch_images.shape[0], -1)
        batch_labels = labels[next_batch:next_batch+batch_size]
        
        optimizer.zero_grad()
        
        output = model(batch_images)
        loss = criterion(output, batch_labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        next_batch += batch_size
        
    print(running_loss)


# ### Huge Loss, but very simple code to play around <a class="anchor" id="eighth-bullet"></a>
# This is a huuuuge loss (considering we just used 1000 images), but this tutorial is made to make things simple and feel the data. From here you can play around with MLPs or CNNs. Convolutional Neural Nets are great, instead of losing all the spatial information when flattennig the image, we can understand images much better, it solves this problem by working with the weights and biases.
# 
# <img src="https://jhonatandasilva.com/wp-content/uploads/2018/12/cnns.gif" alt="drawing" width="400"/>
# 
# You can Look it up more resources on CNNs here
# 
# * [CNNs made it easy](https://jhonatandasilva.com/cnns-made-it-easy/) 
# * [How the layers of CNNs works](https://jhonatandasilva.com/cnns-layers/)
# * [How to build your CNN with Keras](https://youtu.be/lkvzqfhgITM)
# 
# 
