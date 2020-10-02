#!/usr/bin/env python
# coding: utf-8

# **If you like this Kernel enough to fork it, please remember to upvote as well!**

# This is the first notebook from version 3 of the fast.ai course! For more information you can visit https://www.fast.ai/.

# # Lesson 1 - What's your pet

# In this lesson we will build our first image classifier from scratch, and see if we can achieve world-class results. Let's dive in!
# 
# Every notebook starts with the following three lines; they ensure that any edits to libraries you make are reloaded here automatically, and also that any charts or images displayed are shown in this notebook.

# In[1]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# We import all the necessary packages. We are going to work with the [fastai V1 library](http://www.fast.ai/2018/10/02/fastai-ai/) which sits on top of [Pytorch 1.0](https://hackernoon.com/pytorch-1-0-468332ba5163). The fastai library provides many useful functions that enable us to quickly and easily build neural networks and train our models.

# In[2]:


from fastai import *
from fastai.vision import *


# If you're using a computer with an unusually small GPU, you may get an out of memory error when running this notebook. If this happens, click Kernel->Restart, uncomment the 2nd line below to use a smaller *batch size* (you'll learn all about what this means during the course), and try again.

# In[3]:


bs = 8
# bs = 16   # uncomment this line if you run out of memory even after clicking Kernel->Restart


# ## Looking at the data

# We are going to use the [Oxford-IIIT Pet Dataset](http://www.robots.ox.ac.uk/~vgg/data/pets/) by [O. M. Parkhi et al., 2012](http://www.robots.ox.ac.uk/~vgg/publications/2012/parkhi12a/parkhi12a.pdf) which features 12 cat breeds and 25 dogs breeds. Our model will need to learn to differentiate between these 37 distinct categories. According to their paper, the best accuracy they could get in 2012 was 59.21%, using a complex model that was specific to pet detection, with separate "Image", "Head", and "Body" models for the pet photos. Let's see how accurate we can be using deep learning!
# 
# We are going to use the `untar_data` function to which we must pass a URL as an argument and which will download and extract the data.

# In[4]:


help(untar_data)


# In[5]:


path = untar_data(URLs.PETS); path


# In[6]:


URLs.PETS


# In[7]:


path.ls()


# In[8]:


path_anno = path/'annotations'
path_img = path/'images'
path_anno


# The first thing we do when we approach a problem is to take a look at the data. We _always_ need to understand very well what the problem is and what the data looks like before we can figure out how to solve it. Taking a look at the data means understanding how the data directories are structured, what the labels are and what some sample images look like.
# 
# The main difference between the handling of image classification datasets is the way labels are stored. In this particular dataset, labels are stored in the filenames themselves. We will need to extract them to be able to classify the images into the correct categories. Fortunately, the fastai library has a handy function made exactly for this, `ImageDataBunch.from_name_re` gets the labels from the filenames using a [regular expression](https://docs.python.org/3.6/library/re.html).

# In[9]:


fnames = get_image_files(path_img)
fnames[:5]


# In[10]:


np.random.seed(2)
pat = re.compile(r'/([^/]+)_\d+.jpg$')


# In[11]:


data = ImageDataBunch.from_name_re(path_img, fnames, pat, ds_tfms=get_transforms(), size=224, bs=bs, num_workers=4
                                  ).normalize(imagenet_stats)


# In[12]:


data.show_batch(rows=3, figsize=(7,6))


# In[13]:


print(data.classes)
len(data.classes),data.c


# ## Training: resnet34

# Now we will start training our model. We will use a [convolutional neural network](http://cs231n.github.io/convolutional-networks/) backbone and a fully connected head with a single hidden layer as a classifier. Don't know what these things mean? Not to worry, we will dive deeper in the coming lessons. For the moment you need to know that we are building a model which will take images as input and will output the predicted probability for each of the categories (in this case, it will have 37 outputs).
# 
# We will train for 4 epochs (4 cycles through all our data).

# In[14]:


learn = cnn_learner(data, models.resnet34, metrics=error_rate)


# In[15]:


learn.fit_one_cycle(4)


# In[16]:


learn.recorder.plot_losses()


# In[ ]:


learn.recorder.plot_lr()

