#!/usr/bin/env python
# coding: utf-8

# ![](https://software.intel.com/sites/default/files/managed/fc/32/expanding-possibilities-computer-vision-with-ai-wallpaper.jpg)

# **Hello Everyone !**
# 
# **This is a gentle introductory kernel meant to be a starter's guide to the field of Computer Vision and Image processing in particular.**
# 
# **A brief listing of the things we are going to cover in the following kernel -**
# 
# **----  Section A  ----**
# 
# **1.How images are stored in systems and why they are stored the way they are.**
# 
# **2.How to read images and view their sizes,view the images etc.**
# 
# **3.How to resize the images in order to fit in specific requirements for training purposes.**
# 
# **----  Section B  ----**
# 
# **1.The Basics of kernel transformation and multiplication.**
# 
# **2.We came across terms like Kernel, Filter and Padding.**
# 
# **3.We had an idea about wha Convolutions actually are and how they take place**
# 
# **4.We got an idea about how the number of channels of any image can be altered for the purpose of learning important features.**
# 
# **----  Section C  ----**
# 
# **1.We learn about making a really basic Computer Vision model architecture.**
# 
# **2.We then move forward to training the model for the classification problem.**

# 

# **---- SECTION A ----**

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import cv2

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        (os.path.join(dirname, filename));

# Any results you write to the current directory are saved as output.


# In[ ]:


train = pd.read_csv('../input/flower-recognition-he/he_challenge_data/data/train.csv')


# **IMAGES**

# The way we see the images and the way machines store images in their memories are very different.
# 
# An image is basically a **collection of various pixels** (Picture Elements), which are the smallest parts and which collectively make up the image as we see it.

# **Primary Colours** - Red, Green and Blue

# ![](https://upload.wikimedia.org/wikipedia/commons/thumb/c/c2/AdditiveColor.svg/2000px-AdditiveColor.svg.png)

# The most common way of storing these images are in form of **arrays of numbers**, where these numbers represent the different levels of activations of the primary colours,i.e, Red, Green and Blue (RGB).
# 
# The images are usually stored in the **RGB format** because these are the primary colours and any colour can then be generated from these colours by varying the activations of these colours.
# 
# In a nutshell, **there are 3 different 2D arrays, each corresponding to one colour out of RGB.**
# 
# PS: There are other ways of storing/representing an image too, which we can consider while going into the depths of image processing.

# **Reading an Image**

# Here we use the **OpenCV library** to read in the images and then view them and we try to make small tweaks and manipulations to it.

# In[ ]:


img=cv2.imread('../input/flower-recognition-he/he_challenge_data/data/train/12.jpg')


# Lets have a look at the **shape** of the image.

# In[ ]:


img.shape


# So the image we have here is of the shape **(500,500,3)** , which means, there are **three 2D matrices of size (500,500)** , each for every colour from Red, Green and Blue.

# Lets have a look at the image.

# In[ ]:


plt.imshow(img)


# Now, we can **resize** the images as per our own needs using the resize() function of OpenCV.
# 
# The **bigger the size of the images, more is the level of detailing** in them and hence, more is the computational expense required to process these images.
# 
# Thus, due to resource limitations, generally we work with images of sizes below (300,300) and images bigger than these sizes are resized to fit the model.

# In[ ]:


plt.imshow(cv2.resize(img, (50,50)))


# This is the same image which has been aggressively scaled down to a lower size, leading to huge **information/detail loss** as is clearly visible from the image above.

# Just a recap of what we have already covered -
# 
# 1.How images are stored in systems and why they are stored the way they are.
# 
# 2.How to read images and view their sizes,view the images etc.
# 
# 3.How to resize the images in order to fit in specific requirements for training purposes.
# 
# 
# Now, we will move forward to making a very basic Convolution architecture for training on these images and to get going towards solving a classification problem.

# **---- SECTION B ----**

# **CNN Basics**

# Now, before diving right into making a model and classifying the images, lets have a brief idea about how CNN models actually work.

# So, as we have discussed above, there are 2D matrices which we are going to refer to as '**channels**' from here forth.
# 
# Now, every channel undergoes an affine transformation using a kernel which is another 2D matrix of a smaller size. 
# 
# **The steps of the transformation are as follows -** 
# 
# 1.The kernel gets **multiplied with a matrix** of the same size from the channel, giving **one scalar value** as output. 
# 
# 2.This kernel then **shifts sidewards/downwards** to another set of numbers from the channel of the same size as the kernel and **another scalar is generated.**
# 
# 3.This repeatedly takes place until all such possible smaller matrices are multiplied with the kernel to produce a **new matrix** made from the scalars generated.
# 
# A visual representation of the entire procedure is shown below -

# ![](https://mlnotebook.github.io/img/CNN/convExample.png)

# A complete animated visual of the process for clearer understanding is given below.
# 

# ![](https://miro.medium.com/max/526/1*GcI7G-JLAQiEoCON7xFbhg.gif)

# **Gentle Introduction to the Concept of 'Padding' - **

# The most obvious observation here is that the **size of the resulting matrix is smaller than the matrix we started out with**.
# 
# We started out with a (5,5) matrix and ended up having a (3,3) matrix as output after the operation.
# 
# This leads to some **information loss** and here comes into the picture a technique known as "**Padding**", which is used to conserve the size of the original matrix.
# 
# What padding does is it adds additional cells across the border of the original matrix so that when the kernel multiplication takes place, we still end up with the same size of output matrix.
# 
# A visual representation of the padding concept is given below - 
# 

# ![](http://deeplearning.net/software/theano_versions/dev/_images/same_padding_no_strides.gif)

# So, now we have a matrix of the same size as our input matrix.

# **Extending this knowledge to multiple channels**

# Moving forward to the next most important thing, we discussed above the way one particular channel undergoes the matrix multiplication process to generate an output channel.
# 
# Now, as we know, an image is generally consists of more than one channels ( unless it is a greyscale image, in which case it has 1 channel only ).
# 
# So, now we need to extend the above knowledge of kernel multiplication to Three and greater dimensions.

# **Introduction to Filters**

# Here comes another term known as 'Filter', which is quite frequently used with the term 'kernel'.
# 
# For clarification, the multiple kernels combine to form what is known as a 'filter'.
# 
# So, for the processing of say a matrix which has 3 channels, we would need 3 kernels, one for each.
# 
# Now, these 3 kernels collectively make One Filter.

# Lets look at the entire process and break them step wise.

# **Step 1 -
# **
# 
# These 3 kernels act on the 3 different channels of the image, giving out 3 matrices as output as shown in the representation below -

# ![](https://miro.medium.com/max/1166/1*8dx6nxpUh2JqvYWPadTwMQ.gif)

# **Step 2 -**
# 
# All the elements in the layers formed now are added up to give one channel 

# ![](https://jie-tao.com/wp-content/uploads/2019/02/stardard-convolution-multi-channel2.gif)

# Thus, as we can see, **one filter yields us one channel**.
# 
# So, in case we want a Nf number of output channels, all we need to do is, take an image with say Ni input channels, and make Nf filters each consisting of multiple kernels.

# One last thing before we move forward to making our first model, since in our case we are trying to classify the images into 102 clases, we would want our model to finally give us out 102 values which we would then give as an input to the **SoftMax**^ layer, which would then give as output the probabilities of the image belonging to each class.
# 
# **^** - For now, we can see the SoftMax layer as being a layer which takes as input the numbers generated from the model and converts them into **probabilities** of each of the classes.

# **Beore going any further, lets have a recap of what we have discussed in this section of the kernel -**
# 
# 1.The Basics of kernel transformation and multiplication.
# 
# 2.We came across terms like Kernel, Filter and Padding.
# 
# 3.We had an idea about wha Convolutions actually are and how they take place
# 
# 4.We got an idea about how the number of channels of any image can be altered for the purpose of learning important features.

# **---- SECTION C ----**

# **GETTING OUR HANDS DIRTY MAKING A VERY BASIC MODEL**

# Making a very simple model using FastAI

# In[ ]:


#Importing the libraries
from fastai.vision import *
from fastai.vision.models import *
from fastai.vision.learner import model_meta


# Here, the parameters represent these in the same order - 
# 
# 1.Number of input channels
# 
# 2.Number of Output Channels
# 
# 3.Kernel Size (3,3 or 5,5 are the general used sizes)
# 
# 4.Stride (We haven't discussed this yet, so we can consider this to be a default parameter )
# 
# 5.Padding.

# In[ ]:


model = nn.Sequential(
        nn.Conv2d(3,8,kernel_size=3,stride=2,padding=2), #8*250*250
        nn.BatchNorm2d(8),
        nn.ReLU(),
        
        nn.Conv2d(8,16,kernel_size=3,stride=2,padding=2), #16*125*125
        nn.BatchNorm2d(16),
        nn.ReLU(),
        
        nn.Conv2d(16,32,kernel_size=3,stride=2,padding=2), #32*63*63
        nn.BatchNorm2d(32),
        nn.ReLU(),
    
        nn.Conv2d(32,64,kernel_size=3,stride=2,padding=2), #64*32*32
        nn.BatchNorm2d(64),
        nn.ReLU(),
    
        nn.Conv2d(64,128,kernel_size=3,stride=2,padding=2), #128*16*16
        nn.BatchNorm2d(128),
        nn.ReLU(),
            
        nn.Conv2d(128,256,kernel_size=3,stride=2,padding=2), #256*8*8
        nn.BatchNorm2d(256),
        nn.ReLU(),
    
        nn.Conv2d(256,128,kernel_size=3,stride=2,padding=2), #128*4*4
        nn.BatchNorm2d(128),
        nn.ReLU(),
        
        nn.Conv2d(128,102,kernel_size=3,stride=2,padding=2), #102*2*2
        nn.BatchNorm2d(102),
        nn.ReLU(),
    
        nn.Conv2d(102,102,kernel_size=3,stride=2,padding=2), #102*1*1
        nn.BatchNorm2d(102),
        Flatten()        
)


# Lets have a look at the model...

# In[ ]:


model


# Setting up the path to the images folder and initialising the ImageDataBunch.
# 
# Now, as we know the images are stored in a separate folder and the labels are stored separately in a CSV named 'train.csv'.
# 
# **So, the prime function of the ImageDataBunch class is to -**
# 
# 1.Read the rows from the train.csv
# 
# 2.Pick up the image corresponding to the name in the train.csv from the train images directory and send it to the model for training purposes.
# 
# 3.Once an epoch is complete, it again restarts the epoch and keeps on providing the model a continued flow of images throughout.

# In[ ]:


path = pathlib.Path('../input/flower-recognition-he/he_challenge_data/data/');path.ls()
np.random.seed(20)
data = ImageDataBunch.from_csv(path, folder='train', csv_labels='train.csv',suffix='.jpg',
                               valid_pct=0.15, test='test',
                               size=128,bs = 64)


# **Parameters - 
# **
# 
# 1.'**valid_pct**' represents that we are making 15% of our total train set as our validation set.
# 
# 2.**Size** corresponds to the size of the input images.
# 
# The image is **scaled** to the mentioned size before passing it into the model for training purposes.
# 
# 3.**bs is the Batch Size**, i.e, how many number of images go into the model for training at once.
# 
# Since due to computational resources limitation, the model can't work on all the images at once, so, we have **batches** of images which are sequentially generated by the ImageDataBunch class and sent to the model for training purposes.
# 

# Defining a learner object in order to train the model and giving it the Data and the Model to work with and giving it a metric for evaluation purposes.

# In[ ]:


learn = Learner(data,model,metrics=[accuracy])


# Here's a summary of the model we are going to train on the dataset.

# In[ ]:


learn.summary()


# **Training the Model...**

# Now we can finally call the fit() function in order to start training the model on our images.
# 
# We can see the performance of our model using the "**Validation Loss**" as well as the "**Accuracy**" column in the training report generated.

# In[ ]:


learn.fit(3)


# In[ ]:





# Thus, this was all about the basics of the functioning of CNN and making a very basic CNN model for Image Classification.

# **Now that we have reached the end of the kernel, I am assuming you liked the kernel, since you didnt close it mid-way.**
# 
# **If you did like it, please UPVOTE the kernel. That keeps me going !**
# 
# **Any suggestions and criticism are welcome.**
# 
# **Also, feel free to ask for clarifications in case some part of the kernel is not clear as you might want it to be**
# 
# **Cheers !**

# **PS: After going through this basic tutorial, MNIST dataset would be a really great place to start off for beginners.**

# **PPS: For a more advanced approach towards Image classification and using State of the Art techniques do have a look at this kernel -
# **
# 
# https://www.kaggle.com/sandeeppat/ship-classification-top-3-5-kernel

# In[ ]:




